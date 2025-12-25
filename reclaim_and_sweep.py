#!/usr/bin/env python3
"""Reclaim SOL from *empty token-account rent* + **Pump creator rewards** for all
keypairs under ./keys, then sweep remaining SOL back into the deposit wallet.

Flow per wallet (excluding deposit wallet):
  1) Optional topup from deposit (only when needed to pay fees for non-sweep work)
  2) Close empty token accounts (reclaim rent)
  3) Collect Pump creator rewards (if any)
  4) Sweep remaining SOL back to deposit

Safety:
  - Defaults to DRY RUN. Use --execute to broadcast transactions.

Env:
  - DEPOSIT_WALLET_KEYPAIR_PATH (default: ./keys/deposit_wallet.json)
  - HELIUS_API_KEY (optional)
  - RPC_URL or SOLANA_URL (optional override)
  - KEYS_DIR (optional default override)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.system_program import ID as SYS_PROGRAM_ID
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction


# ---------------- Minimal base58 (no external dependency) ----------------
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def b58decode(s: str) -> bytes:
    s_bytes = s.encode("ascii")
    n = 0
    for ch in s_bytes:
        try:
            n = n * 58 + _B58_INDEX[ch]
        except KeyError as e:
            raise ValueError(f"Invalid base58 character: {chr(ch)!r}") from e
    # Convert int -> bytes
    out = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    # Add leading zeros
    pad = 0
    for ch in s_bytes:
        if ch == _B58_ALPHABET[0]:
            pad += 1
        else:
            break
    return b"\x00" * pad + out


# ---------------- Program IDs / constants ----------------
LAMPORTS_PER_SOL = 1_000_000_000

TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
TOKEN_2022_PROGRAM_ID = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")

PUMP_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")

# Anchor discriminator for Pump collect_creator_fee
PUMP_COLLECT_CREATOR_FEE_DISCRIMINATOR = bytes([20, 22, 86, 123, 198, 28, 219, 132])

# SPL Token CloseAccount instruction index
CLOSE_ACCOUNT_IX = bytes([9])


# ---------------- Data structures ----------------
@dataclass(slots=True)
class WalletScan:
    pubkey_str: str
    key_path: Path

    native_lamports: int = 0
    rent_claimable_lamports: int = 0
    pump_claimable_lamports: int = 0

    # List of (token_program_id_str, token_account_pubkey_str)
    closable_token_accounts: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def grand_total_lamports(self) -> int:
        return self.native_lamports + self.rent_claimable_lamports + self.pump_claimable_lamports


# ---------------- RPC helpers ----------------
def _default_rpc_url() -> str:
    override = (os.getenv("RPC_URL") or os.getenv("SOLANA_URL") or "").strip()
    if override:
        return override
    api_key = (os.getenv("HELIUS_API_KEY") or "").strip()
    if api_key:
        return f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    print("WARNING: HELIUS_API_KEY not set; using public mainnet RPC (slower).")
    return "https://api.mainnet-beta.solana.com"


def rpc_call(rpc_url: str, method: str, params: list, *, max_retries: int = 8) -> dict:
    """Raw JSON-RPC helper with basic 429/backoff handling."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(rpc_url, data=data, headers={"Content-Type": "application/json"})

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out = json.loads(resp.read().decode("utf-8"))
                if "error" in out:
                    raise RuntimeError(f"RPC error: {out['error']}")
                return out
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(min(2 * attempt, 10))
                continue
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise RuntimeError(f"RPC HTTPError {e.code} {e.reason}: {body}") from e
        except Exception as e:
            last_err = e
            time.sleep(min(1.25 * attempt, 8))
            continue

    raise RuntimeError(f"RPC call failed after retries: {method} {params} (last={last_err})")


def _decode_account_data(data_field) -> bytes:
    if isinstance(data_field, list) and data_field:
        raw = data_field[0]
        if isinstance(raw, str) and raw:
            return base64.b64decode(raw)
        return b""
    if isinstance(data_field, str) and data_field:
        return base64.b64decode(data_field)
    return b""


def _rent_exempt_lamports(rpc_url: str, data_len: int, rent_cache: Dict[int, int]) -> int:
    """Cache getMinimumBalanceForRentExemption by data_len."""
    if data_len not in rent_cache:
        resp = rpc_call(rpc_url, "getMinimumBalanceForRentExemption", [int(data_len)])
        rent_cache[data_len] = int(resp.get("result", 0) or 0)
    return int(rent_cache[data_len])


def get_balance(rpc_url: str, pubkey_str: str) -> int:
    resp = rpc_call(rpc_url, "getBalance", [pubkey_str, {"commitment": "confirmed"}])
    return int(resp.get("result", {}).get("value", 0) or 0)


def _get_latest_blockhash(rpc_url: str) -> Hash:
    resp = rpc_call(rpc_url, "getLatestBlockhash", [{"commitment": "confirmed"}])
    bh = resp.get("result", {}).get("value", {}).get("blockhash")
    if not bh:
        raise RuntimeError(f"getLatestBlockhash failed: {resp}")
    return Hash.from_string(str(bh))


def _confirm_signature(rpc_url: str, sig: str, *, timeout_sec: int = 60) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = rpc_call(
            rpc_url,
            "getSignatureStatuses",
            [[sig], {"searchTransactionHistory": True}],
        )
        val = (resp.get("result", {}).get("value") or [None])[0]
        if val is None:
            time.sleep(0.8)
            continue
        err = val.get("err")
        if err:
            raise RuntimeError(f"Transaction {sig} failed: {err}")
        status = (val.get("confirmationStatus") or "").lower()
        if status in {"confirmed", "finalized"}:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for confirmation: {sig}")


def send_transaction(
    rpc_url: str,
    payer: Keypair,
    instructions: Sequence[Instruction],
    *,
    skip_preflight: bool = False,
    max_retries: int = 3,
) -> str:
    if not instructions:
        raise ValueError("No instructions to send")

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            bh = _get_latest_blockhash(rpc_url)
            msg = Message.new_with_blockhash(list(instructions), payer.pubkey(), bh)
            tx = Transaction.new_unsigned(msg)
            tx.sign([payer], bh)
            tx_b64 = base64.b64encode(bytes(tx)).decode("utf-8")

            resp = rpc_call(
                rpc_url,
                "sendTransaction",
                [
                    tx_b64,
                    {
                        "encoding": "base64",
                        "skipPreflight": bool(skip_preflight),
                        "preflightCommitment": "confirmed",
                    },
                ],
            )
            sig = resp.get("result")
            if not sig:
                raise RuntimeError(f"sendTransaction returned no signature: {resp}")
            _confirm_signature(rpc_url, str(sig))
            return str(sig)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"send_transaction failed after retries: {last_exc}") from last_exc


def estimate_fee_for_message(
    rpc_url: str,
    payer_pubkey: Pubkey,
    instructions: Sequence[Instruction],
) -> int:
    bh = _get_latest_blockhash(rpc_url)
    msg = Message.new_with_blockhash(list(instructions), payer_pubkey, bh)
    msg_b64 = base64.b64encode(bytes(msg)).decode("utf-8")
    resp = rpc_call(rpc_url, "getFeeForMessage", [msg_b64, {"commitment": "confirmed"}])
    fee = resp.get("result", {}).get("value")
    if fee is None:
        raise RuntimeError(f"getFeeForMessage failed: {resp}")
    return int(fee)


# ---------------- Keypair loading ----------------
def _keypair_from_64b_secret(raw64: bytes) -> Keypair:
    if hasattr(Keypair, "from_bytes"):
        try:
            return Keypair.from_bytes(raw64)
        except Exception:
            pass
    # Fallback: solders always supports base58-string constructor
    # (requires encoding)
    return Keypair.from_base58_string(_b58encode(raw64))


def _b58encode(b: bytes) -> str:
    if not b:
        return ""
    n = int.from_bytes(b, "big")
    out = bytearray()
    while n > 0:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem])
    # leading zeros
    pad = 0
    for ch in b:
        if ch == 0:
            pad += 1
        else:
            break
    out.extend(_B58_ALPHABET[0] for _ in range(pad))
    out.reverse()
    return out.decode("ascii")


def load_keypair_any(path: Path) -> Keypair:
    """Load a solana-keygen style keypair from .json or .json.gz.

    Supports:
      - JSON array of 64 ints (Solana CLI default)
      - base58 string of 64 raw bytes (optional)
    """
    if path.name.endswith(".json.gz") or path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

    if isinstance(payload, list):
        raw = bytes(int(x) for x in payload)
    elif isinstance(payload, str):
        raw = b58decode(payload)
    else:
        raise ValueError(f"Unsupported keypair format: {path}")

    if len(raw) != 64:
        raise ValueError(f"Keypair must be 64 bytes (got {len(raw)}): {path}")
    return _keypair_from_64b_secret(raw)


def get_pubkey_from_keyfile(path: Path) -> str | None:
    try:
        if path.name.endswith(".gz") or path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            raw = bytes(int(x) for x in payload)
        elif isinstance(payload, str):
            raw = b58decode(payload)
        else:
            return None
        if len(raw) != 64:
            return None
        return str(_keypair_from_64b_secret(raw).pubkey())
    except Exception:
        return None


def iter_key_files(keys_dir: Path) -> Iterable[Path]:
    for root, _dirs, files in os.walk(keys_dir):
        for name in files:
            if name.endswith(".json") or name.endswith(".json.gz"):
                yield (Path(root) / name).resolve()


# ---------------- Pump PDA helpers ----------------
def pump_creator_vault_pda(creator: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([b"creator-vault", bytes(creator)], PUMP_PROGRAM_ID)[0]


def pump_event_authority() -> Pubkey:
    return Pubkey.find_program_address([b"__event_authority"], PUMP_PROGRAM_ID)[0]


# ---------------- Instruction builders ----------------
def build_transfer_ix(from_pubkey: Pubkey, to_pubkey: Pubkey, lamports: int) -> Instruction:
    return transfer(TransferParams(from_pubkey=from_pubkey, to_pubkey=to_pubkey, lamports=int(lamports)))


def build_close_token_account_ix(
    token_program_id: Pubkey,
    token_account: Pubkey,
    destination: Pubkey,
    authority: Pubkey,
) -> Instruction:
    return Instruction(
        program_id=token_program_id,
        accounts=[
            AccountMeta(pubkey=token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=destination, is_signer=False, is_writable=True),
            AccountMeta(pubkey=authority, is_signer=True, is_writable=False),
        ],
        data=CLOSE_ACCOUNT_IX,
    )


def build_pump_collect_ix(creator: Pubkey) -> Instruction:
    creator_vault = pump_creator_vault_pda(creator)
    return Instruction(
        program_id=PUMP_PROGRAM_ID,
        accounts=[
            AccountMeta(pubkey=creator, is_signer=True, is_writable=True),
            AccountMeta(pubkey=creator_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=pump_event_authority(), is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_PROGRAM_ID, is_signer=False, is_writable=False),
        ],
        data=PUMP_COLLECT_CREATOR_FEE_DISCRIMINATOR,
    )


# ---------------- Rent scan helpers ----------------
def get_closeable_empty_token_accounts(
    rpc_url: str,
    owner_pubkey_str: str,
) -> Tuple[List[Tuple[str, str]], int]:
    """Return ([(token_program_id_str, token_account_pubkey_str), ...], total_rent_lamports)."""
    total_rent = 0
    closable: List[Tuple[str, str]] = []

    token_program_ids = [str(TOKEN_PROGRAM_ID), str(TOKEN_2022_PROGRAM_ID)]

    for program_id in token_program_ids:
        resp = rpc_call(
            rpc_url,
            "getTokenAccountsByOwner",
            [owner_pubkey_str, {"programId": program_id}, {"encoding": "jsonParsed"}],
        )
        for acc in resp.get("result", {}).get("value", []) or []:
            try:
                info = acc["account"]["data"]["parsed"]["info"]
                lamports = int(acc["account"]["lamports"])
                raw_amount = info["tokenAmount"].get("amount")
                close_auth = info.get("closeAuthority")
                state = info.get("state")

                # Close if empty, not frozen, and closeAuthority is unset or owned by wallet.
                if raw_amount == "0" and (close_auth in (None, owner_pubkey_str)) and state != "frozen":
                    closable.append((program_id, acc["pubkey"]))
                    total_rent += lamports
            except Exception:
                continue

    return closable, total_rent


# ---------------- Formatting ----------------
def fmt_sol(lamports: int) -> str:
    return f"{lamports / LAMPORTS_PER_SOL:.9f}".rstrip("0").rstrip(".") or "0"


# ---------------- Main ----------------
def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Reclaim rent + Pump creator rewards for all wallets in keys dir, then sweep SOL to deposit."
    )
    ap.add_argument("--keys-dir", default=os.getenv("KEYS_DIR", "./keys"), help="Keys root (default: ./keys)")
    ap.add_argument(
        "--deposit-keypair",
        default=os.getenv("DEPOSIT_WALLET_KEYPAIR_PATH", ""),
        help="Deposit wallet keypair path (default: env DEPOSIT_WALLET_KEYPAIR_PATH or ./keys/deposit_wallet.json)",
    )
    ap.add_argument("--rpc-url", default="", help="RPC URL override (default: RPC_URL/SOLANA_URL/HELIUS_API_KEY)")

    ap.add_argument("--batch-size", type=int, default=50, help="Batch size for getMultipleAccounts (default: 50)")
    ap.add_argument("--close-batch-size", type=int, default=8, help="CloseAccount instructions per tx (default: 8)")
    ap.add_argument("--sleep-ms", type=int, default=100, help="Sleep between tx sends (default: 100ms)")
    ap.add_argument("--scan-sleep-ms", type=int, default=100, help="Sleep between scan batches (default: 100ms)")

    ap.add_argument("--execute", action="store_true", help="Broadcast transactions (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N wallets (0 = all)")

    ap.add_argument(
        "--topup-min-lamports",
        type=int,
        default=120_000,
        help="Minimum lamports to topup when needed (default: 120000)",
    )
    ap.add_argument(
        "--topup-buffer-lamports",
        type=int,
        default=25_000,
        help="Extra lamports to add on top of estimated fees *when wallet needs non-sweep work* (default: 25000)",
    )

    args = ap.parse_args()

    keys_dir = Path(args.keys_dir).expanduser().resolve()
    if not keys_dir.exists():
        print(f"ERROR: keys dir not found: {keys_dir}")
        return 2

    deposit_path = (args.deposit_keypair or "").strip()
    deposit_keypair_path = (
        Path(deposit_path).expanduser().resolve()
        if deposit_path
        else (keys_dir / "deposit_wallet.json").resolve()
    )
    if not deposit_keypair_path.exists():
        print(f"ERROR: deposit wallet keypair not found: {deposit_keypair_path}")
        return 2

    rpc_url = (args.rpc_url or "").strip() or _default_rpc_url()

    deposit_kp = load_keypair_any(deposit_keypair_path)
    deposit_pub = deposit_kp.pubkey()
    deposit_pub_str = str(deposit_pub)

    deposit_balance = get_balance(rpc_url, deposit_pub_str)

    print(f"RPC: {rpc_url}")
    print(f"Keys dir: {keys_dir}")
    print(f"Deposit wallet: {deposit_pub_str} ({deposit_keypair_path})")
    print(f"Deposit balance: {fmt_sol(deposit_balance)} SOL ({deposit_balance} lamports)")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("-" * 80)

    # Build pubkey->file map (skip deposit wallet)
    pubkey_to_file: Dict[str, Path] = {}
    for path in iter_key_files(keys_dir):
        if path.resolve() == deposit_keypair_path:
            continue
        pk = get_pubkey_from_keyfile(path)
        if not pk:
            continue
        if pk == deposit_pub_str:
            continue
        pubkey_to_file.setdefault(pk, path)

    pubkeys = list(pubkey_to_file.keys())
    if args.limit and args.limit > 0:
        pubkeys = pubkeys[: args.limit]

    print(f"Found {len(pubkeys)} unique non-deposit keypairs under {keys_dir}")

    # Estimate a baseline fee (1-signature legacy transfer)
    try:
        base_fee = estimate_fee_for_message(
            rpc_url,
            deposit_pub,
            [build_transfer_ix(deposit_pub, deposit_pub, 1)],
        )
    except Exception:
        base_fee = 5_000
    print(f"Baseline fee estimate: {base_fee} lamports ({fmt_sol(base_fee)} SOL)")
    print("-" * 80)

    scans: Dict[str, WalletScan] = {}

    total_native = 0
    total_rent = 0
    total_pump = 0
    rent_cache: Dict[int, int] = {}

    batch_size = max(int(args.batch_size), 1)
    scanned = 0

    for i in range(0, len(pubkeys), batch_size):
        batch_pubkeys = pubkeys[i : i + batch_size]

        # 1) Native balances (batched)
        resp = rpc_call(rpc_url, "getMultipleAccounts", [batch_pubkeys, {"encoding": "jsonParsed"}])
        infos = resp.get("result", {}).get("value", []) or []
        for j, info in enumerate(infos):
            pk = batch_pubkeys[j]
            lamports = int(info.get("lamports", 0)) if info else 0
            scans[pk] = WalletScan(pubkey_str=pk, key_path=pubkey_to_file[pk], native_lamports=lamports)
            total_native += lamports

        # 2) Pump creator rewards (batched)
        solders_pubkeys = [Pubkey.from_string(pk) for pk in batch_pubkeys]
        pump_vaults = [pump_creator_vault_pda(pk) for pk in solders_pubkeys]
        pump_resp = rpc_call(
            rpc_url,
            "getMultipleAccounts",
            [[str(pk) for pk in pump_vaults], {"encoding": "base64"}],
        )
        pump_infos = pump_resp.get("result", {}).get("value", []) or []
        for j, info in enumerate(pump_infos):
            if not info:
                continue
            data_bytes = _decode_account_data(info.get("data"))
            rent = _rent_exempt_lamports(rpc_url, len(data_bytes), rent_cache)
            claimable = max(int(info.get("lamports", 0)) - rent, 0)
            if claimable:
                scans[batch_pubkeys[j]].pump_claimable_lamports = claimable
                total_pump += claimable

        # 3) Claimable rent (token accounts) (per wallet)
        for pk in batch_pubkeys:
            closable, rent_lamports = get_closeable_empty_token_accounts(rpc_url, pk)
            if rent_lamports:
                scans[pk].rent_claimable_lamports = rent_lamports
                scans[pk].closable_token_accounts = closable
                total_rent += rent_lamports

        scanned += len(batch_pubkeys)
        print(
            f"Scanned {scanned}/{len(pubkeys)}... "
            f"Native: {fmt_sol(total_native)} | Rent: {fmt_sol(total_rent)} | Pump: {fmt_sol(total_pump)}"
        )
        time.sleep(max(args.scan_sleep_ms, 0) / 1000)

    # Final scan report (non-deposit wallets only)
    grand_total = total_native + total_rent + total_pump
    print("-" * 80)
    print("SCAN REPORT (excluding deposit wallet)")
    print(f"Total Native SOL (Liquid):      {fmt_sol(total_native)} SOL")
    print(f"Total Claimable Rent (Empty):   {fmt_sol(total_rent)} SOL")
    print(f"Total Creator Rewards (Pump):   {fmt_sol(total_pump)} SOL")
    print("=" * 30)
    print(f"GRAND TOTAL RECOVERABLE:        {fmt_sol(grand_total)} SOL")

    actionable: List[WalletScan] = [
        s
        for s in scans.values()
        if (s.native_lamports > 0 or s.rent_claimable_lamports > 0 or s.pump_claimable_lamports > 0)
    ]
    actionable.sort(key=lambda s: s.grand_total_lamports, reverse=True)

    print("-" * 80)
    print(f"Actionable wallets: {len(actionable)}")

    if not args.execute:
        print("DRY RUN: not broadcasting any transactions.")
        for s in actionable[:25]:
            print(
                f"wallet={s.pubkey_str} native={fmt_sol(s.native_lamports)} rent={fmt_sol(s.rent_claimable_lamports)} "
                f"pump={fmt_sol(s.pump_claimable_lamports)} total={fmt_sol(s.grand_total_lamports)} file={s.key_path}"
            )
        if len(actionable) > 25:
            print(f"... ({len(actionable) - 25} more)")
        return 0

    # ---------------- EXECUTION PHASE ----------------
    tx_sent = 0
    topups_sent = 0

    for idx, s in enumerate(actionable, start=1):
        print("-" * 80)
        print(
            f"[{idx}/{len(actionable)}] wallet={s.pubkey_str} "
            f"native={fmt_sol(s.native_lamports)} rent={fmt_sol(s.rent_claimable_lamports)} pump={fmt_sol(s.pump_claimable_lamports)}"
        )

        try:
            kp = load_keypair_any(s.key_path)
        except Exception as exc:
            print(f"  ERROR: failed to load keypair {s.key_path}: {exc}")
            continue

        wallet_pub = kp.pubkey()
        if str(wallet_pub) != s.pubkey_str:
            print("  WARN: pubkey mismatch after loading; skipping")
            continue

        try:
            current_balance = get_balance(rpc_url, s.pubkey_str)
        except Exception as exc:
            print(f"  ERROR: failed to fetch current balance: {exc}")
            continue

        close_tx_count = (
            math.ceil(len(s.closable_token_accounts) / max(args.close_batch_size, 1))
            if s.closable_token_accounts
            else 0
        )
        pump_tx_count = 1 if s.pump_claimable_lamports > 0 else 0
        sweep_tx_count = 1

        needs_non_sweep_work = (close_tx_count + pump_tx_count) > 0

        # Fee budget only matters if we have non-sweep work to do.
        # For sweep-only wallets, topups are usually a net loss (extra tx fee from deposit).
        if needs_non_sweep_work:
            est_wallet_fees = (close_tx_count + pump_tx_count + sweep_tx_count) * base_fee
            est_wallet_fees += int(args.topup_buffer_lamports)

            if current_balance < est_wallet_fees:
                needed = est_wallet_fees - current_balance
                topup_amt = max(int(args.topup_min_lamports), int(needed))
                print(
                    f"  topup needed: balance={current_balance} est_fees={est_wallet_fees} "
                    f"-> sending {topup_amt} lamports ({fmt_sol(topup_amt)} SOL)"
                )
                try:
                    sig = send_transaction(
                        rpc_url,
                        deposit_kp,
                        [build_transfer_ix(deposit_pub, wallet_pub, topup_amt)],
                    )
                    tx_sent += 1
                    topups_sent += 1
                    time.sleep(max(args.sleep_ms, 0) / 1000)
                    print(f"  topup sig: {sig}")
                except Exception as exc:
                    print(f"  ERROR: topup failed, skipping wallet: {exc}")
                    continue

        # 1) Close empty token accounts (rent)
        if s.closable_token_accounts:
            print(f"  closing {len(s.closable_token_accounts)} empty token accounts...")
            batch_n = max(int(args.close_batch_size), 1)
            for j in range(0, len(s.closable_token_accounts), batch_n):
                batch = s.closable_token_accounts[j : j + batch_n]
                ixes: List[Instruction] = []
                for program_id_str, token_acc_str in batch:
                    ixes.append(
                        build_close_token_account_ix(
                            Pubkey.from_string(program_id_str),
                            Pubkey.from_string(token_acc_str),
                            destination=wallet_pub,
                            authority=wallet_pub,
                        )
                    )
                try:
                    sig = send_transaction(rpc_url, kp, ixes)
                    tx_sent += 1
                    time.sleep(max(args.sleep_ms, 0) / 1000)
                    print(f"    close tx {j//batch_n + 1}: {sig}")
                except Exception as exc:
                    print(f"    WARN: batch close failed ({exc}); retrying individually...")
                    for program_id_str, token_acc_str in batch:
                        try:
                            sig = send_transaction(
                                rpc_url,
                                kp,
                                [
                                    build_close_token_account_ix(
                                        Pubkey.from_string(program_id_str),
                                        Pubkey.from_string(token_acc_str),
                                        destination=wallet_pub,
                                        authority=wallet_pub,
                                    )
                                ],
                            )
                            tx_sent += 1
                            time.sleep(max(args.sleep_ms, 0) / 1000)
                            print(f"      closed {token_acc_str}: {sig}")
                        except Exception as exc2:
                            print(f"      ERROR: failed to close {token_acc_str}: {exc2}")

        # 2) Collect Pump creator rewards
        if s.pump_claimable_lamports > 0:
            print(f"  collecting Pump creator rewards (~{fmt_sol(s.pump_claimable_lamports)} SOL)...")
            try:
                sig = send_transaction(rpc_url, kp, [build_pump_collect_ix(wallet_pub)])
                tx_sent += 1
                time.sleep(max(args.sleep_ms, 0) / 1000)
                print(f"  pump collect sig: {sig}")
            except Exception as exc:
                print(f"  ERROR: pump collect failed: {exc}")

        # 3) Sweep all SOL back to deposit
        try:
            bal = get_balance(rpc_url, s.pubkey_str)
            try:
                sweep_fee = estimate_fee_for_message(
                    rpc_url,
                    wallet_pub,
                    [build_transfer_ix(wallet_pub, deposit_pub, 1)],
                )
            except Exception:
                sweep_fee = base_fee
            sweep_amt = bal - sweep_fee
            if sweep_amt <= 0:
                print(f"  sweep: skipping (balance={bal} fee={sweep_fee})")
            else:
                sig = send_transaction(
                    rpc_url,
                    kp,
                    [build_transfer_ix(wallet_pub, deposit_pub, sweep_amt)],
                )
                tx_sent += 1
                time.sleep(max(args.sleep_ms, 0) / 1000)
                print(f"  swept {fmt_sol(sweep_amt)} SOL -> deposit, sig: {sig}")
        except Exception as exc:
            print(f"  ERROR: sweep failed: {exc}")

    print("-" * 80)
    print("DONE")
    print(f"Transactions sent: {tx_sent}")
    print(f"Topups sent:       {topups_sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Generate reproducible ESMFold2 complex ensembles through the Biohub API.

Key features
------------
* Full ESMFold2 quality defaults: 20 trunk loops and 100 ODE sampling steps.
* Configurable inference-time ensemble over LM dropout, LM masking, MSA depth,
  and MSA column masking while keeping each individual prediction high quality.
* Optional per-chain A3M MSAs and chain-order replicas.
* PAE and pair-chain iPTM output, interface-PAE summaries, and a reproducible
  run-plan CSV written before API submission.
* Multi-key rolling rate-limit accounting compatible with the original tool.
* Backward-compatible core options and stdout behaviour: stdout contains only
  the final summary CSV path.

Install the current SDK with:
    pip install 'esm@git+https://github.com/Biohub/esm.git@main'
"""

import argparse
import dataclasses
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from collections import OrderedDict
from itertools import permutations
from pathlib import Path

try:
    from esm.sdk import esmfold2_client
    from esm.sdk.api import FoldingConfig, ESMProteinError
    from esm.utils.structure import input_builder
    from esm.utils.msa import MSA
except ImportError:
    print(
        "Error: the current 'esm' SDK is required. Install it with: "
        "pip install 'esm@git+https://github.com/Biohub/esm.git@main'",
        file=sys.stderr,
    )
    sys.exit(1)

MAX_CALLS_PER_MINUTE = 20
MAX_CALLS_PER_24H = 100
MINUTE_SECONDS = 60.0
DAY_SECONDS = 86400.0


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def diagnostic_payload(value):
    """Return a JSON-safe diagnostic view without assuming SDK error fields."""
    payload = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "str": str(value),
        "repr": repr(value),
    }
    try:
        if dataclasses.is_dataclass(value):
            payload["dataclass"] = dataclasses.asdict(value)
    except Exception as exc:
        payload["dataclass_inspection_error"] = repr(exc)
    try:
        attrs = vars(value)
    except TypeError:
        attrs = None
    if attrs:
        payload["attributes"] = {
            str(k): diagnostic_json_value(v) for k, v in attrs.items()
            if not str(k).lower().endswith(("token", "api_key", "authorization"))
        }
    for name in ("error_code", "code", "status", "status_code", "message", "msg", "detail", "reason"):
        if hasattr(value, name):
            try:
                payload[name] = diagnostic_json_value(getattr(value, name))
            except Exception as exc:
                payload[f"{name}_inspection_error"] = repr(exc)
    return payload


def diagnostic_json_value(value):
    """Best-effort conversion for diagnostic JSON; never serializes huge tensors."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): diagnostic_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json_value(v) for v in value[:100]]
    return repr(value)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=repr) + "\n")


def csv_values(text, cast, name, allow_none=False):
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if allow_none and raw.lower() in {"none", "null", "off"}:
            values.append(None)
            continue
        try:
            values.append(cast(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid value {raw!r} in {name}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


API_MAX_SEQUENCE_LENGTH = 768


def local_command_recommendation(input_path, total_residues, args):
    """Build a shell-safe recommendation for the companion local runner."""
    import shlex
    local_script = Path(__file__).with_name("run_esmfold2.py")
    loops = min(args.num_loops, 10)
    steps = min(args.num_sampling_steps, 68)
    parts = [
        "python", str(local_script), "-i", str(input_path),
        "--model", "biohub/ESMFold2-Fast",
        "--num-loops", str(loops),
        "--num-sampling-steps", str(steps),
        "--msa_max_depth", "1",
        "--ensemble", "1", "--verbose",
    ]
    return " ".join(shlex.quote(x) for x in parts)


def explain_api_length_limit(input_path, sequences, args):
    total = sum(map(len, sequences.values()))
    longest_id, longest_seq = max(sequences.items(), key=lambda kv: len(kv[1]))
    if total <= API_MAX_SEQUENCE_LENGTH:
        return False
    eprint(
        f"Error: Biohub API input length is {total} residues, exceeding its "
        f"{API_MAX_SEQUENCE_LENGTH}-residue limit by {total - API_MAX_SEQUENCE_LENGTH}."
    )
    eprint(
        "Changing --num-loops, --num-sampling-steps, dropout, masking, MSA depth, "
        "or selecting esmfold2-fast does not change the API validation limit."
    )
    eprint(f"Longest input record: {longest_id} ({len(longest_seq)} residues).")
    if len(sequences) == 1:
        eprint(
            "Options: (1) run the complete sequence locally; or (2) crop/split at a "
            "biologically justified domain/linker boundary. Arbitrary overlapping chunks "
            "will not preserve a reliable full-length inter-domain arrangement."
        )
    else:
        eprint(
            "Options: (1) run the complete complex locally; or (2) submit a biologically "
            "justified subset whose combined length is <=768. Splitting chains removes "
            "the omitted interfaces from the prediction."
        )
    eprint("Recommended companion local command (VRAM-conservative starting point):")
    eprint("  " + local_command_recommendation(input_path, total, args))
    eprint(
        "If that is stable and more accuracy is required, increase in this order: "
        "--num-loops 20, then --num-sampling-steps 100, then MSA depth. "
        "Loops/steps affect compute and quality, not supported length; MSA depth increases memory."
    )
    return True


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Submit a high-quality, diverse ESMFold2 protein-complex ensemble to Biohub.",
    )
    p.add_argument("-i", "--inputfile", help="Multi-record protein FASTA. Required unless --test-availability is used.")
    p.add_argument("--test-availability", action="store_true", help="Summarize how many calls for each API token can be made based on local timestamp records and exit.")
    p.add_argument("--tag", default="esmf", help="Output filename tag.")
    p.add_argument("--outdir", default=None, help="Output directory; defaults to the FASTA directory.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--refresh", action="store_true", help="Recalculate even if a non-empty summary exists.")
    p.add_argument("--dry-run", action="store_true", help="Write the run plan without making API calls.")
    p.add_argument("--num-ensemble", "--n-poses", dest="num_ensemble", type=int, default=10,
                   help="Total API predictions to request.")
    p.add_argument("--api-token", required=True,
                   help="One or more Biohub tokens separated by colons.")
    p.add_argument("--model", default="esmfold2-2026-05",
                   choices=("esmfold2-2026-05", "esmfold2-fast-2026-05"))
    p.add_argument("--num-loops", type=int, default=20, help="ESMFold2 trunk loops; API range 0-20.")
    p.add_argument("--num-sampling-steps", type=int, default=100,
                   help="Diffusion ODE steps; API range 1-100. Kept constant across poses.")
    p.add_argument("--lm-dropouts", default="0.20,0.30",
                   help="Comma-separated LM pair-embedding dropout probabilities.")
    p.add_argument("--lm-mask-pcts", default="0.0",
                   help="Comma-separated sequence-mask fractions; use 'none' for model default.")
    p.add_argument("--msa-depths", default="1024",
                   help="Comma-separated MSA subsampling depths; use 'none' to disable subsampling.")
    p.add_argument("--msa-column-mask-rates", default="0.05,0.10,0.15",
                   help="Comma-separated non-query MSA column-mask fractions.")
    p.add_argument("--msa", action="append", default=[], metavar="CHAIN=FILE.a3m",
                   help="Per-chain A3M. Repeat for multiple chains, e.g. --msa H=H.a3m --msa A=A.a3m.")
    p.add_argument("--msa-load-max-sequences", type=int, default=16384,
                   help="Maximum A3M rows loaded before server-side/inference subsampling.")
    p.add_argument("--chain-order-mode", choices=("canonical", "reverse", "canonical,reverse", "all"),
                   default="canonical,reverse",
                   help="Input-chain serialization replicas. 'all' is capped by --max-chain-orders.")
    p.add_argument("--max-chain-orders", type=int, default=6)
    p.add_argument("--include-distogram", action="store_true")
    p.add_argument("--include-embeddings", action="store_true")
    p.add_argument("--no-pae", action="store_true", help="Do not request PAE (not recommended for complexes).")
    p.add_argument("--no-pair-chains-iptm", action="store_true",
                   help="Do not request pair-chain iPTM (not recommended for complexes).")
    p.add_argument("--spread", type=int, default=0,
                   help="Calls per key/bucket before rotating keys; 0 disables inter-bucket 24h spreading.")
    p.add_argument("--ignore-limit", action="store_true",
                   help="Disable local 20/min and 100/rolling-24h accounting.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-base-seconds", type=float, default=15.0)
    p.add_argument("--ranking-iptm-weight", type=float, default=0.55)
    p.add_argument("--ranking-ptm-weight", type=float, default=0.25)
    p.add_argument("--ranking-plddt-weight", type=float, default=0.20)
    args = p.parse_args()

    if not args.test_availability and not args.inputfile:
        p.error("-i/--inputfile is required unless --test-availability is used.")

    if args.num_ensemble < 1:
        p.error("--num-ensemble must be >= 1")
    if args.spread < 0:
        p.error("--spread must be >= 0")
    if not 0 <= args.num_loops <= 20:
        p.error("--num-loops must be in [0, 20]")
    if not 1 <= args.num_sampling_steps <= 100:
        p.error("--num-sampling-steps must be in [1, 100]")
    if args.max_chain_orders < 1:
        p.error("--max-chain-orders must be >= 1")
    if args.msa_load_max_sequences < 1 or args.msa_load_max_sequences > 16384:
        p.error("--msa-load-max-sequences must be in [1, 16384]")

    args.lm_dropouts = csv_values(args.lm_dropouts, float, "--lm-dropouts")
    args.lm_mask_pcts = csv_values(args.lm_mask_pcts, float, "--lm-mask-pcts", allow_none=True)
    args.msa_depths = csv_values(args.msa_depths, int, "--msa-depths", allow_none=True)
    args.msa_column_mask_rates = csv_values(
        args.msa_column_mask_rates, float, "--msa-column-mask-rates"
    )
    for name, values in (("--lm-dropouts", args.lm_dropouts),
                         ("--lm-mask-pcts", [x for x in args.lm_mask_pcts if x is not None]),
                         ("--msa-column-mask-rates", args.msa_column_mask_rates)):
        if any(x < 0 or x > 1 for x in values):
            p.error(f"{name} values must be in [0, 1]")
    if any(x is not None and not 1 <= x <= 16384 for x in args.msa_depths):
        p.error("--msa-depths values must be in [1, 16384] or 'none'")
    if args.model.endswith("-fast-2026-05") and args.msa:
        p.error("Per-chain MSAs require --model esmfold2-2026-05, not the Fast model")
    return args


def read_fasta(path):
    records = OrderedDict()
    current = None
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                if not current:
                    raise ValueError(f"Empty FASTA identifier at line {line_no}")
                if current in records:
                    raise ValueError(f"Duplicate FASTA identifier: {current}")
                records[current] = []
            elif current is None:
                raise ValueError(f"Sequence before first FASTA header at line {line_no}")
            else:
                seq = re.sub(r"\s+", "", line).upper()
                if not re.fullmatch(r"[A-Z*.-]+", seq):
                    raise ValueError(f"Invalid FASTA sequence characters at line {line_no}")
                records[current].append(seq)
    sequences = OrderedDict((k, "".join(v).replace("*", "")) for k, v in records.items())
    if not sequences or any(not seq for seq in sequences.values()):
        raise ValueError("FASTA must contain at least one non-empty sequence")
    return sequences


def parse_msa_specs(specs, chain_ids, max_sequences):
    paths = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --msa {spec!r}; expected CHAIN=FILE.a3m")
        chain, filename = spec.split("=", 1)
        chain, path = chain.strip(), Path(filename).expanduser()
        if chain not in chain_ids:
            raise ValueError(f"MSA chain {chain!r} is not in FASTA chains: {', '.join(chain_ids)}")
        if chain in paths:
            raise ValueError(f"Duplicate --msa for chain {chain}")
        if not path.is_file():
            raise FileNotFoundError(f"MSA file not found: {path}")
        paths[chain] = path
    return {
        chain: MSA.from_a3m(path=str(path), remove_insertions=True, max_sequences=max_sequences)
        for chain, path in paths.items()
    }, paths


def chain_orders(chain_ids, mode, maximum):
    canonical = tuple(chain_ids)
    orders = [canonical]
    if mode in {"reverse", "canonical,reverse"}:
        orders = [tuple(reversed(canonical))] if mode == "reverse" else [canonical, tuple(reversed(canonical))]
    elif mode == "all":
        orders = list(permutations(canonical))
    unique = []
    for order in orders:
        if order not in unique:
            unique.append(order)
    return unique[:maximum]


def build_run_plan(args, sequences):
    orders = chain_orders(list(sequences), args.chain_order_mode, args.max_chain_orders)
    configs = []
    for order in orders:
        for dropout in args.lm_dropouts:
            for mask_pct in args.lm_mask_pcts:
                for depth in args.msa_depths:
                    for column_mask in args.msa_column_mask_rates:
                        configs.append({
                            "chain_order": order,
                            "lm_dropout": dropout,
                            "lm_mask_pct": mask_pct,
                            "msa_max_depth": depth,
                            "msa_column_mask_rate": column_mask,
                        })
    if not configs:
        raise ValueError("The ensemble parameter grid is empty")
    plan = []
    for index in range(args.num_ensemble):
        cfg = dict(configs[index % len(configs)])
        cfg["sample"] = index + 1
        cfg["grid_cycle"] = index // len(configs) + 1
        plan.append(cfg)
    return plan


def safe_scalar(value):
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def tensor_to_nested(value):
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    except (TypeError, RuntimeError):
        pass
    return value if isinstance(value, (list, tuple)) else None


def pair_chain_iptm_json(result):
    for attr in ("pair_chains_iptm", "pair_chain_iptm", "pairwise_iptm"):
        raw = getattr(result, attr, None)
        nested = tensor_to_nested(raw)
        if nested is not None:
            return json.dumps(nested, separators=(",", ":"))
    return ""


def interface_pae_summary(result, order, sequences):
    """Return mean/max inter-chain PAE from a square residue-level PAE matrix."""
    matrix = tensor_to_nested(getattr(result, "pae", None))
    if not matrix or not isinstance(matrix, list):
        return "", ""
    lengths = [len(sequences[c]) for c in order]
    total = sum(lengths)
    if len(matrix) != total or any(not isinstance(row, list) or len(row) != total for row in matrix):
        return "", ""
    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    values = []
    for i, (a0, a1) in enumerate(bounds):
        for j, (b0, b1) in enumerate(bounds):
            if i == j:
                continue
            for a in range(a0, a1):
                for b in range(b0, b1):
                    try:
                        x = float(matrix[a][b])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(x):
                        values.append(x)
    if not values:
        return "", ""
    return f"{sum(values) / len(values):.4f}", f"{max(values):.4f}"


def parse_api_keys(text):
    keys = [x.strip() for x in text.split(":") if x.strip()]
    if not keys:
        raise ValueError("No API tokens were supplied")
    return keys


def key_alias(key):
    # Used for display and output CSV matching the first 10 chars of SHA-256
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def get_rate_file(key):
    # Generates a path in $HOME formatted as biohub.api.<md5_hash>.txt
    md5_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return Path.home() / f"biohub.api.{md5_hash}.txt"


def read_timestamps(path, now):
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text().splitlines():
            try:
                ts = float(line)
                if 0 <= now - ts < DAY_SECONDS:
                    out.append(ts)
            except ValueError:
                continue
    except OSError:
        return []
    return sorted(out)


def write_timestamps(path, timestamps):
    path.write_text("".join(f"{x:.6f}\n" for x in sorted(timestamps)), encoding="utf-8")


def wait_needed(timestamps, now):
    waits = []
    recent = [x for x in timestamps if now - x < MINUTE_SECONDS]
    if len(recent) >= MAX_CALLS_PER_MINUTE:
        waits.append(MINUTE_SECONDS - (now - recent[-MAX_CALLS_PER_MINUTE]) + 0.05)
    if len(timestamps) >= MAX_CALLS_PER_24H:
        waits.append(DAY_SECONDS - (now - timestamps[-MAX_CALLS_PER_24H]) + 0.05)
    return max([0.0] + waits)


def select_key_for_call(keys, files, call_index, selector, args):
    """Select one key per API call, allowing an ensemble to span keys/days.

    With ``--spread N``, each consecutive bucket of N planned poses prefers
    one key before rotating. A full preferred key spills to another key with
    rolling-24h capacity. If every key is full, wait for the earliest slot.
    """
    offset = int(hashlib.sha256(selector.encode()).hexdigest(), 16) % len(keys)
    bucket_size = args.spread if args.spread > 0 else MAX_CALLS_PER_24H
    preferred = (offset + call_index // bucket_size) % len(keys)
    if args.ignore_limit:
        return preferred, keys[preferred]

    while True:
        now = time.time()
        states = []
        for rank in range(len(keys)):
            idx = (preferred + rank) % len(keys)
            timestamps = read_timestamps(files[keys[idx]], now)
            recent = sum(now - ts < MINUTE_SECONDS for ts in timestamps)
            states.append((len(timestamps) < MAX_CALLS_PER_24H, rank, recent,
                           len(timestamps), idx, timestamps))
        eligible = [state for state in states if state[0]]
        if eligible:
            _, _, _, _, idx, _ = min(
                eligible,
                key=lambda state: (state[1] != 0, state[3], state[2], state[1]),
            )
            return idx, keys[idx]

        sleep_seconds, _, _ = min(
            (max(0.0, timestamps[0] + DAY_SECONDS + 0.05 - now), rank, idx)
            for _, rank, _, _, idx, timestamps in states
        )
        if args.verbose:
            eprint(
                "All API keys have reached the local rolling-24h limit; "
                f"sleeping {sleep_seconds:.2f}s until one call slot is free"
            )
        time.sleep(sleep_seconds)

def reserve_call_on_key(key, rate_file, args):
    """Reserve one call on an already selected ensemble key."""
    if args.ignore_limit:
        return
    while True:
        now = time.time()
        timestamps = read_timestamps(rate_file, now)
        wait = wait_needed(timestamps, now)
        if wait > 0:
            if args.verbose:
                eprint(f"Selected ensemble key is locally limited; sleeping {wait:.2f}s")
            time.sleep(wait)
            continue
        timestamps.append(time.time())
        write_timestamps(rate_file, timestamps)
        return


def make_config(args, item):
    return FoldingConfig(
        include_distogram=args.include_distogram,
        include_pae=not args.no_pae,
        include_pair_chains_iptm=not args.no_pair_chains_iptm,
        num_sampling_steps=args.num_sampling_steps,
        num_loops=args.num_loops,
        lm_dropout=item["lm_dropout"],
        lm_mask_pct=item["lm_mask_pct"],
        msa_max_depth=item["msa_max_depth"],
        msa_column_mask_rate=item["msa_column_mask_rate"],
        include_embeddings=args.include_embeddings,
    )


def write_plan(path, plan, args, msa_paths):
    fields = ["sample", "grid_cycle", "model", "chain_order", "num_loops",
              "num_sampling_steps", "lm_dropout", "lm_mask_pct", "msa_max_depth",
              "msa_column_mask_rate", "msa_files"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in plan:
            writer.writerow({
                "sample": item["sample"], "grid_cycle": item["grid_cycle"],
                "model": args.model, "chain_order": ":".join(item["chain_order"]),
                "num_loops": args.num_loops, "num_sampling_steps": args.num_sampling_steps,
                "lm_dropout": item["lm_dropout"], "lm_mask_pct": item["lm_mask_pct"],
                "msa_max_depth": item["msa_max_depth"],
                "msa_column_mask_rate": item["msa_column_mask_rate"],
                "msa_files": json.dumps({k: str(v.resolve()) for k, v in msa_paths.items()}, sort_keys=True),
            })


def main():
    args = parse_args()
    
    # Handle the --test-availability early exit flag
    if args.test_availability:
        api_keys = parse_api_keys(args.api_token)
            
        now = time.time()
        print(f"{'Token Alias':<15} | {'1-Min Capacity':<16} | {'24-Hour Capacity':<18} | {'Rate File'}")
        print("-" * 80)
        
        total_min_avail = 0
        total_min_max = 0
        total_day_avail = 0
        total_day_max = 0
        
        for key in api_keys:
            alias = key_alias(key)
            rate_file = get_rate_file(key)
            
            timestamps = read_timestamps(rate_file, now)
            recent = [x for x in timestamps if now - x < MINUTE_SECONDS]
            
            min_avail = max(0, MAX_CALLS_PER_MINUTE - len(recent))
            day_avail = max(0, MAX_CALLS_PER_24H - len(timestamps))
            
            total_min_avail += min_avail
            total_min_max += MAX_CALLS_PER_MINUTE
            total_day_avail += day_avail
            total_day_max += MAX_CALLS_PER_24H
            
            print(f"{alias:<15} | {min_avail:>2} / {MAX_CALLS_PER_MINUTE:<11} | {day_avail:>3} / {MAX_CALLS_PER_24H:<12} | {rate_file.name}")
        
        print("-" * 80)
        print(f"{'TOTAL':<15} | {total_min_avail:>2} / {total_min_max:<11} | {total_day_avail:>3} / {total_day_max:<12} |")
        return

    input_path = Path(args.inputfile).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input FASTA not found: {input_path}")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    summary_path = outdir / f"{stem}.{args.tag}.csv"
    plan_path = outdir / f"{stem}.{args.tag}.run_plan.csv"
    if summary_path.exists() and summary_path.stat().st_size > 0 and not args.refresh:
        print(str(summary_path.resolve()))
        return

    sequences = read_fasta(input_path)
    complex_id = stem
    if explain_api_length_limit(input_path, sequences, args):
        raise SystemExit(2)
    msa_by_chain, msa_paths = parse_msa_specs(
        args.msa, list(sequences), args.msa_load_max_sequences
    )
    plan = build_run_plan(args, sequences)
    write_plan(plan_path, plan, args, msa_paths)
    if args.verbose:
        eprint(f"Run plan: {plan_path}")
        eprint(f"Complex {complex_id}: {len(sequences)} chains, {sum(map(len, sequences.values()))} residues")
        eprint(f"Model: {args.model}; poses: {len(plan)}; loops/steps: {args.num_loops}/{args.num_sampling_steps}")
    if args.dry_run:
        print(str(plan_path.resolve()))
        return

    api_keys = parse_api_keys(args.api_token)
    clients = {key: esmfold2_client(model=args.model, token=key) for key in api_keys}
    
    # Store and map local limits accurately to $HOME with the biohub.api.<md5>.txt format
    rate_files = {key: get_rate_file(key) for key in api_keys}
    records = []

    if not args.ignore_limit and args.verbose:
        total_capacity = len(api_keys) * MAX_CALLS_PER_24H
        eprint(
            f"Local rolling-24h capacity: {total_capacity} calls across "
            f"{len(api_keys)} key(s); large ensembles will span keys and, "
            "when necessary, rolling-24h windows"
        )

    for item in plan:
        sample = item["sample"]
        order = item["chain_order"]
        config = make_config(args, item)
        ensemble_key_idx, ensemble_key = select_key_for_call(
            api_keys, rate_files, call_index=sample - 1,
            selector=f"{complex_id}:{len(plan)}", args=args,
        )
        ensemble_client = clients[ensemble_key]
        ensemble_rate_file = rate_files[ensemble_key]
        protein_inputs = [
            input_builder.ProteinInput(id=chain, sequence=sequences[chain], msa=msa_by_chain.get(chain))
            for chain in order
        ]
        structure_input = input_builder.StructurePredictionInput(sequences=protein_inputs)
        if args.verbose:
            eprint(
                f"[{sample}/{len(plan)}] key={ensemble_key_idx + 1}/{len(api_keys)} order={':'.join(order)} "
                f"dropout={item['lm_dropout']} mask={item['lm_mask_pct']} "
                f"msa_depth={item['msa_max_depth']} msa_col_mask={item['msa_column_mask_rate']}"
            )

        result = None
        last_error = None
        error_log_path = outdir / f"{stem}.{args.tag}.errors.jsonl"
        for attempt in range(args.max_retries + 1):
            attempt_started = time.monotonic()
            try:
                # Every submission consumes quota, including retry attempts.
                reserve_call_on_key(ensemble_key, ensemble_rate_file, args)
                if args.verbose:
                    eprint(f"  API attempt {attempt + 1}/{args.max_retries + 1}: submitting")
                candidate = ensemble_client.fold_all_atom(structure_input, config=config)
                elapsed = time.monotonic() - attempt_started
                if isinstance(candidate, ESMProteinError):
                    last_error = candidate
                    event = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "complex_id": complex_id,
                        "sample": sample,
                        "attempt": attempt + 1,
                        "elapsed_seconds": round(elapsed, 3),
                        "api_key_alias": key_alias(ensemble_key),
                        "model": args.model,
                        "input_residues": sum(len(sequences[c]) for c in order),
                        "chain_order": list(order),
                        "config": {
                            "num_loops": args.num_loops,
                            "num_sampling_steps": args.num_sampling_steps,
                            "lm_dropout": item["lm_dropout"],
                            "lm_mask_pct": item["lm_mask_pct"],
                            "msa_max_depth": item["msa_max_depth"],
                            "msa_column_mask_rate": item["msa_column_mask_rate"],
                        },
                        "error": diagnostic_payload(candidate),
                    }
                    append_jsonl(error_log_path, event)
                    eprint(
                        f"  API returned ESMProteinError after {elapsed:.2f}s: "
                        f"{json.dumps(event['error'], sort_keys=True, default=repr)}"
                    )
                    error_text = f"{candidate!s} {candidate!r}"
                    if "exceeds maximum allowed sequence length" in error_text:
                        eprint("  This is a non-retryable API input-length error.")
                        explain_api_length_limit(input_path, sequences, args)
                        break
                elif not hasattr(candidate, "complex"):
                    last_error = TypeError(
                        f"Unexpected API result type {type(candidate).__module__}."
                        f"{type(candidate).__qualname__}; missing 'complex'"
                    )
                    event = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "complex_id": complex_id, "sample": sample, "attempt": attempt + 1,
                        "elapsed_seconds": round(elapsed, 3),
                        "api_key_alias": key_alias(ensemble_key),
                        "result": diagnostic_payload(candidate),
                    }
                    append_jsonl(error_log_path, event)
                    eprint(f"  {last_error}: {json.dumps(event['result'], sort_keys=True, default=repr)}")
                else:
                    result = candidate
                    if args.verbose:
                        eprint(f"  API attempt {attempt + 1} succeeded in {elapsed:.2f}s")
                    break
            except Exception as exc:
                elapsed = time.monotonic() - attempt_started
                last_error = exc
                event = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "complex_id": complex_id, "sample": sample, "attempt": attempt + 1,
                    "elapsed_seconds": round(elapsed, 3),
                    "api_key_alias": key_alias(ensemble_key),
                    "exception": diagnostic_payload(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_log_path, event)
                eprint(f"  API attempt {attempt + 1} raised {type(exc).__name__}: {exc}")
                if args.verbose:
                    eprint(event["traceback"].rstrip())
            if attempt < args.max_retries:
                delay = args.retry_base_seconds * (2 ** attempt) * random.uniform(0.8, 1.2)
                eprint(f"  Retrying after {delay:.2f}s; diagnostics: {error_log_path}")
                time.sleep(delay)
        if result is None:
            eprint(
                f"Error: sample {sample} failed after {args.max_retries + 1} attempts; "
                f"last error={last_error!r}; diagnostics={error_log_path}"
            )
            continue

        try:
            cif_text = result.complex.to_mmcif()
        except Exception as exc:
            event = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "complex_id": complex_id, "sample": sample,
                "stage": "to_mmcif", "api_key_alias": key_alias(ensemble_key),
                "exception": diagnostic_payload(exc), "result": diagnostic_payload(result),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(error_log_path, event)
            eprint(f"Error: sample {sample} could not be serialized to mmCIF: {exc!r}; diagnostics={error_log_path}")
            if args.verbose:
                eprint(event["traceback"].rstrip())
            continue
        md5sum = hashlib.md5(cif_text.encode("utf-8")).hexdigest()
        cif_path = outdir / f"{stem}.{args.tag}.pose_{sample:04d}.{md5sum[:10]}.cif"
        cif_path.write_text(cif_text, encoding="utf-8")
        iptm = safe_scalar(getattr(result, "iptm", None))
        ptm = safe_scalar(getattr(result, "ptm", None))
        plddt = safe_scalar(getattr(result, "plddt", None))
        pae = safe_scalar(getattr(result, "pae", None))
        interface_mean, interface_max = interface_pae_summary(result, order, sequences)
        components = ((args.ranking_iptm_weight, iptm),
                      (args.ranking_ptm_weight, ptm),
                      (args.ranking_plddt_weight, plddt))
        available_weight = sum(w for w, value in components if value is not None)
        rank_score = (sum(w * value for w, value in components if value is not None) / available_weight
                      if available_weight else None)
        records.append({
            "Name": f"ESMF-{complex_id}-{md5sum}", "md5sum": md5sum,
            "structure": str(cif_path.resolve()), "ComplexID": complex_id,
            "Chains": ":".join(order), "Sample": sample, "grid_cycle": item["grid_cycle"],
            "model": args.model, "api_key_alias": key_alias(ensemble_key),
            "num_loops": args.num_loops, "num_sampling_steps": args.num_sampling_steps,
            "lm_dropout": item["lm_dropout"], "lm_mask_pct": item["lm_mask_pct"],
            "msa_max_depth": item["msa_max_depth"],
            "msa_column_mask_rate": item["msa_column_mask_rate"],
            "msa_files": json.dumps({k: str(v.resolve()) for k, v in msa_paths.items()}, sort_keys=True),
            "sequences": ":".join(sequences[c] for c in order),
            "Amino Acids": ":".join(str(len(sequences[c])) for c in order),
            "iptm": "" if iptm is None else f"{iptm:.6f}",
            "pair_chains_iptm": pair_chain_iptm_json(result),
            "ptm": "" if ptm is None else f"{ptm:.6f}",
            "mean_plddt": "" if plddt is None else f"{plddt:.6f}",
            "mean_pae": "" if pae is None else f"{pae:.6f}",
            "mean_interface_pae": interface_mean, "max_interface_pae": interface_max,
            "rank_score": "" if rank_score is None else f"{rank_score:.6f}",
        })

    fields = [
        "Name", "md5sum", "structure", "ComplexID", "Chains", "Sample", "grid_cycle",
        "model", "api_key_alias", "num_loops", "num_sampling_steps", "lm_dropout",
        "lm_mask_pct", "msa_max_depth", "msa_column_mask_rate", "msa_files", "sequences",
        "Amino Acids", "iptm", "pair_chains_iptm", "ptm", "mean_plddt", "mean_pae",
        "mean_interface_pae", "max_interface_pae", "rank_score",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: float(r["rank_score"] or "-inf"), reverse=True))
    if args.verbose:
        eprint(f"Completed {len(records)}/{len(plan)} predictions; summary: {summary_path}")
    print(str(summary_path.resolve()))


if __name__ == "__main__":
    main()

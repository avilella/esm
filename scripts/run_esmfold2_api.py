#!/usr/bin/env python3
import argparse
import csv
import sys
import hashlib
import time
import random
from pathlib import Path

# Import the ESM SDK as demonstrated in the esmfold2.py notebook
try:
    from esm.sdk import esmfold2_client #
    from esm.sdk.api import FoldingConfig, ESMProteinError #
    from esm.utils.structure import input_builder #
except ImportError:
    print(
        "Error: The 'esm' SDK package is not installed. "
        "Please install it using: pip install esm@git+https://github.com/Biohub/esm.git@main",
        file=sys.stderr,
    )
    sys.exit(1)


def eprint(*args, **kwargs):
    """Prints strictly to sys.stderr."""
    print(*args, file=sys.stderr, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit ESMFold2 protein complex folding tasks using the biohub.ai client."
    )

    parser.add_argument(
        "-i", "--inputfile", required=True, help="Input FASTA file containing protein chains of the complex."
    )
    parser.add_argument(
        "--tag", default="esmf", help="Tag for the output file (default: esmf)."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: same directory as inputfile).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and processing logs to sys.stderr.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recalculate tasks even if non-empty output CSV already exists.",
    )
    parser.add_argument(
        "--num-ensemble",
        type=int,
        default=5,
        help="Number of CIF complex structures to generate in the ensemble (default: 5).",
    )
    parser.add_argument(
        "--api-token",
        required=True,
        help="biohub.ai API key/token string(s) for authentication. Separate multiple keys with colons (key1:key2:key3).",
    )
    parser.add_argument(
        "--spread",
        type=int,
        default=0,
        help="Spread calls over 24h. Uses NN calls per key before semi-randomly waiting and rotating keys.",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=10,
        help="Number of refinement loops (default: 10).",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=100,
        help="Number of diffusion sampling steps (default: 100).",
    )
    parser.add_argument(
        "--ignore-limit",
        action="store_true",
        help="Ignore the default API rate limits (20 calls/min, 100 calls/24h).",
    )

    return parser.parse_args()


def read_fasta(file_path):
    """Parses a FASTA file into a dictionary of ID -> sequence."""
    sequences = {}
    with open(file_path, "r", encoding="utf-8") as f:
        curr_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                curr_id = line[1:].split()[0]
                sequences[curr_id] = []
            elif curr_id is not None:
                sequences[curr_id].append(line)

    for seq_id in sequences:
        sequences[seq_id] = "".join(sequences[seq_id])

    return sequences


def extract_metric(obj, attr_name):
    """Safely extracts and averages tensor metrics returned by the esm client."""
    val = getattr(obj, attr_name, None)
    if val is None:
        return ""
    if hasattr(val, "mean"):
        val = val.mean()
    if hasattr(val, "item"):
        val = val.item()
    try:
        return f"{float(val):.4f}"
    except (ValueError, TypeError):
        return str(val)


MAX_CALLS_PER_MINUTE = 20
MAX_CALLS_PER_24H = 100
MINUTE_SECONDS = 60.0
DAY_SECONDS = 86400.0


def parse_api_keys(api_token_arg):
    """Return a cleaned list of colon-separated API keys."""
    return [key.strip() for key in api_token_arg.split(":") if key.strip()]


def read_rate_limit_timestamps(rate_limit_file, now, verbose=False):
    """Read retained call timestamps for one key, keeping only the last 24h."""
    timestamps = []
    if not rate_limit_file.exists():
        return timestamps

    try:
        with open(rate_limit_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = float(line)
                except ValueError:
                    continue
                if now - ts < DAY_SECONDS:
                    timestamps.append(ts)
    except Exception as e:
        if verbose:
            eprint(f"    Warning: Error reading rate limit file '{rate_limit_file}': {e}")

    timestamps.sort()
    return timestamps


def write_rate_limit_timestamps(rate_limit_file, timestamps, verbose=False):
    """Persist retained timestamps for one key."""
    try:
        with open(rate_limit_file, "w") as f:
            for ts in sorted(timestamps):
                f.write(f"{ts}\n")
    except Exception as e:
        if verbose:
            eprint(f"    Warning: Error writing rate limit file '{rate_limit_file}': {e}")


def rate_limit_wait_seconds(timestamps, now):
    """Return the sleep required before this key can be used, plus the reason."""
    recent = [ts for ts in timestamps if now - ts < MINUTE_SECONDS]
    sleep_time = 0.0
    reason = ""

    if len(timestamps) >= MAX_CALLS_PER_24H:
        sleep_time = max(0.0, DAY_SECONDS - (now - timestamps[0]))
        reason = f"{MAX_CALLS_PER_24H} calls/24h"

    if len(recent) >= MAX_CALLS_PER_MINUTE:
        minute_sleep = max(0.0, MINUTE_SECONDS - (now - recent[0]))
        if minute_sleep > sleep_time:
            sleep_time = minute_sleep
            reason = f"{MAX_CALLS_PER_MINUTE} calls/min"

    return sleep_time, reason


def remaining_24h_capacity(timestamps):
    """Call slots left within the 24h window for one key."""
    return max(0, MAX_CALLS_PER_24H - len(timestamps))


def shuffled_key_indices(api_keys, complex_id, bucket_id):
    """
    Deterministically shuffle the key order for a complex/bucket.

    This keeps all samples inside the same --spread bucket pinned to the same
    preferred key across reruns, while distributing different inputs/buckets
    semi-randomly across the supplied key pool.
    """
    seed_material = f"{complex_id}|bucket:{bucket_id}|keys:{len(api_keys)}"
    seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    indices = list(range(len(api_keys)))
    rng.shuffle(indices)
    return indices


def select_key_for_bucket(api_keys, rate_limit_files, complex_id, bucket_id, bucket_calls, args):
    """
    Pick one key for the whole spread bucket.

    The function tries to choose a key that can accommodate the entire bucket
    inside its remaining 24h capacity. If every key is exhausted for that bucket,
    it sleeps only until the earliest key can accept the bucket, rather than
    getting stuck on the first supplied key.
    """
    while True:
        now = time.time()
        preferred_order = shuffled_key_indices(api_keys, complex_id, bucket_id)
        key_states = []

        for idx in preferred_order:
            key = api_keys[idx]
            timestamps = read_rate_limit_timestamps(rate_limit_files[key], now, verbose=args.verbose)
            day_capacity = remaining_24h_capacity(timestamps)
            wait_time, wait_reason = rate_limit_wait_seconds(timestamps, now)
            key_states.append((idx, key, timestamps, day_capacity, wait_time, wait_reason))

        # Prefer keys with enough remaining 24h quota for the complete bucket.
        day_eligible = [state for state in key_states if state[3] >= bucket_calls]
        if day_eligible:
            # Among day-eligible keys, prefer one not currently minute-limited.
            immediately_available = [state for state in day_eligible if state[4] <= 0]
            selected = immediately_available[0] if immediately_available else min(day_eligible, key=lambda state: state[4])
            idx, key, _, day_capacity, wait_time, wait_reason = selected
            if wait_time > 0:
                if args.verbose:
                    eprint(
                        f"    Selected Key {idx + 1}/{len(api_keys)} for spread bucket {bucket_id + 1}, "
                        f"but it is temporarily limited ({wait_reason}). Sleeping for {wait_time:.2f} seconds..."
                    )
                time.sleep(wait_time)
            return idx, key

        # No key can fit the whole bucket right now. Sleep until the earliest key
        # can fit the complete bucket, preserving bucket-to-key consistency.
        waits = []
        for idx, key, timestamps, day_capacity, _, _ in key_states:
            needed = bucket_calls - day_capacity
            if needed <= 0:
                waits.append((0.0, idx, key))
            elif len(timestamps) >= needed:
                waits.append((max(0.0, DAY_SECONDS - (now - timestamps[needed - 1])), idx, key))

        if not waits:
            raise RuntimeError("Could not compute API key availability from rate limit state.")

        sleep_time, idx, _ = min(waits, key=lambda item: item[0])
        if args.verbose:
            eprint(
                f"    No API key currently has enough 24h quota for this {bucket_calls}-call spread bucket. "
                f"Sleeping for {sleep_time:.2f} seconds until Key {idx + 1}/{len(api_keys)} can accept it..."
            )
        time.sleep(sleep_time)


def reserve_call_for_key(api_key, rate_limit_file, args):
    """Reserve exactly one API call slot for the selected key."""
    if args.ignore_limit:
        return

    while True:
        now = time.time()
        timestamps = read_rate_limit_timestamps(rate_limit_file, now, verbose=args.verbose)
        sleep_time, reason = rate_limit_wait_seconds(timestamps, now)

        if sleep_time > 0:
            if args.verbose:
                eprint(f"    Selected API key is rate limited ({reason}). Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            continue

        timestamps.append(now)
        write_rate_limit_timestamps(rate_limit_file, timestamps, verbose=args.verbose)
        return


def spread_bucket_pause(api_key_count, spread):
    """Semi-random pause between spread buckets over the pooled 24h capacity."""
    if spread <= 0:
        return 0.0
    total_buckets = (api_key_count * float(MAX_CALLS_PER_24H)) / float(spread)
    if total_buckets <= 0:
        return 0.0
    base_wait = DAY_SECONDS / total_buckets
    return random.uniform(base_wait * 0.8, base_wait * 1.2)


def main():
    args = parse_args()

    input_path = Path(args.inputfile)
    if not input_path.is_file():
        eprint(f"Error: Input file '{args.inputfile}' does not exist.")
        sys.exit(1)

    # Determine target output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = input_path.parent

    outdir.mkdir(parents=True, exist_ok=True)

    # Construct output name: <input_stem>.<tag>.csv
    out_file = outdir / f"{input_path.stem}.{args.tag}.csv"

    # Refresh check: if output exists and is non-empty, skip execution
    if not args.refresh:
        if out_file.exists() and out_file.stat().st_size > 0:
            if args.verbose:
                eprint(f"Output file '{out_file}' already exists and is non-empty. Skipping computation.")
            # Print ONLY the resolved path to STDOUT and exit cleanly
            print(str(out_file.resolve()))
            sys.exit(0)

    if args.verbose:
        eprint(f"Reading sequences from '{input_path}'...")

    sequences = read_fasta(input_path)

    if not sequences:
        eprint(f"Error: No FASTA sequences found in '{input_path}'.")
        sys.exit(1)

    api_keys = parse_api_keys(args.api_token)
    if not api_keys:
        eprint("Error: No non-empty API keys found in --api-token.")
        sys.exit(1)

    if args.verbose:
        eprint(
            f"Parsed {len(sequences)} chain(s) for complex folding: {list(sequences.keys())}. "
            f"Found {len(api_keys)} API key(s). Initializing Biohub API clients..."
        )

    # Initialize a Biohub client SDK and rate limit file for each key
    clients = {}
    rate_limit_files = {}
    for key in api_keys:
        clients[key] = esmfold2_client(
            model="esmfold2-2026-05",
            url="https://biohub.ai",
            token=key
        ) #

        # Hash the API token to create a unique, secure filename for rate limiting
        token_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        rate_limit_files[key] = Path.home() / f"biohub.api.{token_hash}.txt"

    complex_id = input_path.stem
    chains_str = "+".join(sequences.keys())
    seqs_str = ":".join(sequences.values())
    amino_acids_str = seqs_str.replace(":", "")
    records = []

    spread_size = args.spread if args.spread and args.spread > 0 else 1
    active_bucket_id = None
    active_key_idx = None
    active_key = None

    if args.verbose:
        eprint(f"\nFolding complex '{complex_id}' with {len(sequences)} chains...")
        if args.spread > 0:
            eprint(
                f"    Spread mode: assigning each {args.spread}-sample bucket to one key from the pooled "
                f"{len(api_keys)}-key quota. Different buckets/inputs are distributed semi-randomly."
            )

    for i in range(1, args.num_ensemble + 1):
        bucket_id = (i - 1) // spread_size
        first_sample_in_bucket = bucket_id * spread_size + 1
        last_sample_in_bucket = min(args.num_ensemble, (bucket_id + 1) * spread_size)
        bucket_calls = last_sample_in_bucket - first_sample_in_bucket + 1

        if bucket_id != active_bucket_id:
            if active_bucket_id is not None and args.spread > 0:
                wait_time = spread_bucket_pause(len(api_keys), args.spread)
                if args.verbose:
                    eprint(
                        f"    Spread bucket {active_bucket_id + 1} complete. "
                        f"Sleeping for {wait_time:.2f} seconds before selecting the next pooled API key..."
                    )
                time.sleep(wait_time)

            if args.ignore_limit:
                preferred = shuffled_key_indices(api_keys, complex_id, bucket_id)[0]
                active_key_idx = preferred
                active_key = api_keys[active_key_idx]
            else:
                active_key_idx, active_key = select_key_for_bucket(
                    api_keys=api_keys,
                    rate_limit_files=rate_limit_files,
                    complex_id=complex_id,
                    bucket_id=bucket_id,
                    bucket_calls=bucket_calls,
                    args=args,
                )
            active_bucket_id = bucket_id

            if args.verbose:
                eprint(
                    f"    Spread bucket {bucket_id + 1}: samples {first_sample_in_bucket}-{last_sample_in_bucket} "
                    f"assigned to Key {active_key_idx + 1}/{len(api_keys)}."
                )

        # Select the active client and rate-limit file for this iteration.
        current_key = active_key
        current_key_idx = active_key_idx
        client = clients[current_key]
        rate_limit_file = rate_limit_files[current_key]

        # Start with the original num_loops and num_sampling_steps,
        # and increment num_sampling_steps by one for subsequent calls.
        current_num_loops = args.num_loops
        current_num_sampling_steps = args.num_sampling_steps + (i - 1)

        # Setup folding parameters
        config = FoldingConfig(
            num_loops=current_num_loops,
            num_sampling_steps=current_num_sampling_steps,
            include_pae=True
        ) #

        # Build multi-chain protein complex input INSIDE the loop
        # to ensure it is independent and not mutated by previous client calls
        protein_inputs = [
            input_builder.ProteinInput(id=seq_id, sequence=seq)
            for seq_id, seq in sequences.items()
        ] #
        complex_input = input_builder.StructurePredictionInput(sequences=protein_inputs) #

        if args.verbose:
            eprint(f"  Sampling complex fold {i}/{args.num_ensemble} via Biohub API (Key {current_key_idx + 1}/{len(api_keys)})...")
            eprint(f"    Parameters: num_loops={current_num_loops}, num_sampling_steps={current_num_sampling_steps}")

        # Cross-execution file-based throttling logic (20 calls/min, 100 calls/24h).
        # The key has already been selected from the pool; here we reserve one slot
        # for that selected key without falling back to the first input key.
        reserve_call_for_key(current_key, rate_limit_file, args)

        try:
            # Call remote prediction service for all-atom complex structure
            fold_result = client.fold_all_atom(complex_input, config=config) #

            # Check if the SDK returned an explicit ESMProteinError object
            if type(fold_result).__name__ == "ESMProteinError":
                error_msg = getattr(fold_result, "error_msg", str(fold_result))
                raise Exception(f"API Error: {error_msg}")

            # Export to mmCIF string format
            cif_content = fold_result.complex.to_mmcif() #

            # Compute MD5 Hash
            md5_hash = hashlib.md5(cif_content.encode('utf-8')).hexdigest()

            # Save CIF locally
            cif_path = outdir / f"{complex_id}_complex_sample_{i}.cif"
            with open(cif_path, "w", encoding="utf-8") as f:
                f.write(cif_content)

            resolved_cif_path = str(cif_path.resolve())

            # Safely extract metrics
            iptm_val = extract_metric(fold_result, 'iptm') #
            ptm_val = extract_metric(fold_result, 'ptm') #
            plddt_val = extract_metric(fold_result, 'plddt') #
            pae_val = extract_metric(fold_result, 'pae') #

            # Append as a single record
            records.append({
                "Name": f"ESMF-{complex_id}-{md5_hash}",
                "md5sum": md5_hash,
                "structure": resolved_cif_path,
                "ComplexID": complex_id,
                "Chains": chains_str,
                "Sample": str(i),
                "num_loops": str(current_num_loops),
                "num_sampling_steps": str(current_num_sampling_steps),
                "sequences": seqs_str,
                "Amino Acids": amino_acids_str,
                "iptm": iptm_val,
                "ptm": ptm_val,
                "mean_plddt": plddt_val,
                "mean_pae": pae_val
            })

            if args.verbose:
                eprint(f"    Saved complex structure to: {resolved_cif_path}")
                eprint(f"    MD5: {md5_hash} | ipTM: {iptm_val} | pTM: {ptm_val}")

        except Exception as e:
            eprint(f"    Error processing complex sample {i}: {e}")

    if args.verbose:
        eprint(f"\nWriting summary CSV to '{out_file}'...")

    # Write output CSV mapping each CIF file to a unique row
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Name", "md5sum", "structure", "ComplexID", "Chains", "Sample",
            "num_loops", "num_sampling_steps", "sequences", "Amino Acids",
            "iptm", "ptm", "mean_plddt", "mean_pae"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    if args.verbose:
        eprint("All tasks complete.")

    # STDOUT output reserved exclusively for output CSV path
    print(str(out_file.resolve()))


if __name__ == "__main__":
    main()

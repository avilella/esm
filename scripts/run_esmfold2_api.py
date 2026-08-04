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

    api_keys = args.api_token.split(':')

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
    
    current_key_idx = 0
    calls_in_bucket = 0

    if args.verbose:
        eprint(f"\nFolding complex '{complex_id}' with {len(sequences)} chains...")

    for i in range(1, args.num_ensemble + 1):
        
        # Implement '--spread NN' logic: Switch keys and wait semi-randomly if bucket is full
        if args.spread > 0 and calls_in_bucket >= args.spread:
            # Calculate wait time to spread the total expected calls evenly across 24h (86400 seconds)
            total_buckets = (len(api_keys) * 100.0) / args.spread
            base_wait = 86400.0 / total_buckets
            wait_time = random.uniform(base_wait * 0.8, base_wait * 1.2)
            
            if args.verbose:
                eprint(f"    Spread bucket of {args.spread} calls reached.")
                eprint(f"    Sleeping for {wait_time:.2f} seconds before switching to the next API key...")
            
            time.sleep(wait_time)
            
            current_key_idx = (current_key_idx + 1) % len(api_keys)
            calls_in_bucket = 0

        # Select the active client and rate limit file for this iteration
        current_key = api_keys[current_key_idx]
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

        # Cross-execution file-based throttling logic (20 calls/min, 100 calls/24h)
        if not args.ignore_limit:
            while True:
                current_time = time.time()
                timestamps = []
                
                # Read existing timestamps from the shared file
                if rate_limit_file.exists():
                    try:
                        with open(rate_limit_file, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    ts = float(line)
                                    # Only retain timestamps from the last 24 hours (86400 seconds)
                                    if current_time - ts < 86400.0:
                                        timestamps.append(ts)
                    except Exception as e:
                        if args.verbose:
                            eprint(f"    Warning: Error reading rate limit file: {e}")
                
                timestamps.sort()
                recent_timestamps = [ts for ts in timestamps if current_time - ts < 60.0]
                
                sleep_time = 0.0
                limit_reason = ""
                
                # Check 24-hour limit (100 calls)
                if len(timestamps) >= 100:
                    sleep_time = 86400.0 - (current_time - timestamps[0])
                    limit_reason = "100 calls/24h"
                    
                # Check 1-minute limit (20 calls)
                if len(recent_timestamps) >= 20:
                    minute_sleep = 60.0 - (current_time - recent_timestamps[0])
                    if minute_sleep > sleep_time:
                        sleep_time = minute_sleep
                        limit_reason = "20 calls/min"

                if sleep_time > 0:
                    if args.verbose:
                        eprint(f"    API rate limit ({limit_reason}) reached for this key. Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    # Limits not reached; record current call and overwrite file
                    timestamps.append(current_time)
                    try:
                        with open(rate_limit_file, "w") as f:
                            for ts in timestamps:
                                f.write(f"{ts}\n")
                    except Exception as e:
                        if args.verbose:
                            eprint(f"    Warning: Error writing to rate limit file: {e}")
                    break

        try:
            # Call remote prediction service for all-atom complex structure
            fold_result = client.fold_all_atom(complex_input, config=config) #
            
            # Increment bucket counter on successful API execution
            calls_in_bucket += 1

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

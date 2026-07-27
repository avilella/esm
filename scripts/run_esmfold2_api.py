#!/usr/bin/env python3
import argparse
import csv
import sys
import hashlib
from pathlib import Path

# Import the ESM SDK as demonstrated in the esmfold2.py notebook
try:
    from esm.sdk import esmfold2_client #[cite: 3]
    from esm.sdk.api import FoldingConfig, ESMProteinError #[cite: 3]
    from esm.utils.structure import input_builder #[cite: 3]
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
        "--tag", default="tool", help="Tag for the output file (default: tool)."
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
        help="biohub.ai API key/token string for authentication.",
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

    if args.verbose:
        eprint(
            f"Parsed {len(sequences)} chain(s) for complex folding: {list(sequences.keys())}. "
            "Initializing Biohub API client..."
        )

    # Initialize the Biohub client SDK
    client = esmfold2_client(
        model="esmfold2-2026-05", 
        url="https://biohub.ai", 
        token=args.api_token
    ) #[cite: 3]

    complex_id = input_path.stem
    chains_str = "+".join(sequences.keys())
    seqs_str = ":".join(sequences.values())
    records = []

    if args.verbose:
        eprint(f"\nFolding complex '{complex_id}' with {len(sequences)} chains...")

    for i in range(1, args.num_ensemble + 1):
        
        # Start with the original num_loops and num_sampling_steps, 
        # and increment num_sampling_steps by one for subsequent calls.
        current_num_loops = args.num_loops
        current_num_sampling_steps = args.num_sampling_steps + (i - 1)

        # Setup folding parameters
        config = FoldingConfig(
            num_loops=current_num_loops,
            num_sampling_steps=current_num_sampling_steps,
            include_pae=True
        ) #[cite: 3]

        # Build multi-chain protein complex input INSIDE the loop 
        # to ensure it is independent and not mutated by previous client calls
        protein_inputs = [
            input_builder.ProteinInput(id=seq_id, sequence=seq)
            for seq_id, seq in sequences.items()
        ] #[cite: 3]
        complex_input = input_builder.StructurePredictionInput(sequences=protein_inputs) #[cite: 3]

        if args.verbose:
            eprint(f"  Sampling complex fold {i}/{args.num_ensemble} via Biohub API...")
            eprint(f"    Parameters: num_loops={current_num_loops}, num_sampling_steps={current_num_sampling_steps}")

        try:
            # Call remote prediction service for all-atom complex structure
            fold_result = client.fold_all_atom(complex_input, config=config) #[cite: 3]

            # Check if the SDK returned an explicit ESMProteinError object
            if isinstance(fold_result, ESMProteinError): #[cite: 3]
                error_msg = getattr(fold_result, "error_msg", str(fold_result))
                raise Exception(f"API Error: {error_msg}")

            # Export to mmCIF string format
            cif_content = fold_result.complex.to_mmcif() #[cite: 3]
            
            # Compute MD5 Hash
            md5_hash = hashlib.md5(cif_content.encode('utf-8')).hexdigest()

            # Save CIF locally
            cif_path = outdir / f"{complex_id}_complex_sample_{i}.cif"
            with open(cif_path, "w", encoding="utf-8") as f:
                f.write(cif_content)

            resolved_cif_path = str(cif_path.resolve())

            # Safely extract metrics
            iptm_val = extract_metric(fold_result, 'iptm') #[cite: 3]
            ptm_val = extract_metric(fold_result, 'ptm') #[cite: 3]
            plddt_val = extract_metric(fold_result, 'plddt') #[cite: 3]
            pae_val = extract_metric(fold_result, 'pae') #[cite: 3]

            # Append as a single record
            records.append({
                "Name": f"ESF2-{md5_hash}",
                "md5sum": md5_hash,
                "structure": resolved_cif_path,
                "ComplexID": complex_id,
                "Chains": chains_str,
                "Sample": str(i),
                "num_loops": str(current_num_loops),
                "num_sampling_steps": str(current_num_sampling_steps),
                "sequences": seqs_str,
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
            "num_loops", "num_sampling_steps", "sequences", 
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

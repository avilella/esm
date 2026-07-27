#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

# Import the ESM SDK as demonstrated in the notebook
try:
    from esm.sdk import esmfold2_client
    from esm.sdk.api import FoldingConfig
    from esm.utils.structure import input_builder
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
        description="Submit ESMFold2 folding tasks using the biohub.ai client."
    )

    parser.add_argument(
        "-i", "--inputfile", required=True, help="Input FASTA file containing protein sequences."
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
        help="Number of CIF structures to generate per sequence (default: 5).",
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
        help="num-loops.",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=100,
        help="num-sampling steps.",
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

    # Construct name: <input_stem>.<tag>.csv
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
    
    if args.verbose:
        eprint(f"Parsed {len(sequences)} sequence(s). Initializing Biohub API client...")

    # Initialize the Biohub client SDK
    client = esmfold2_client(
        model="esmfold2-2026-05", 
        url="https://biohub.ai", 
        token=args.api_token
    )

    # Setup folding parameters
    config = FoldingConfig(num_loops=args.num_loops, num_sampling_steps=args.num_sampling_steps, include_pae=True)

    results = []

    for seq_id, seq in sequences.items():
        if args.verbose:
            eprint(f"\nProcessing sequence '{seq_id}' (Length: {len(seq)} aa)...")

        # Create ESM structure input
        protein_input = input_builder.ProteinInput(id=seq_id, sequence=seq)
        structure_input = input_builder.StructurePredictionInput(sequences=[protein_input])

        cif_files = []
        for i in range(1, args.num_ensemble + 1):
            if args.verbose:
                eprint(f"  Sampling fold {i}/{args.num_ensemble} via Biohub API...")

            try:
                # Call remote prediction service
                fold_result = client.fold_all_atom(structure_input, config=config)

                # Export to mmCIF string format
                cif_content = fold_result.complex.to_mmcif()

                # Save CIF locally
                cif_path = outdir / f"{seq_id}_sample_{i}.cif"
                with open(cif_path, "w", encoding="utf-8") as f:
                    f.write(cif_content)

                resolved_cif_path = str(cif_path.resolve())
                cif_files.append(resolved_cif_path)

                if args.verbose:
                    eprint(f"    Saved structure to: {resolved_cif_path}")

            except Exception as e:
                eprint(f"    Error processing sample {i} for '{seq_id}': {e}")

        results.append([seq_id] + cif_files)

    if args.verbose:
        eprint(f"\nWriting summary CSV to '{out_file}'...")

    # Write output CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["SequenceID"] + [f"CIF_File_{i}" for i in range(1, args.num_ensemble + 1)]
        writer.writerow(header)
        for row in results:
            writer.writerow(row)

    if args.verbose:
        eprint("All tasks complete.")

    # STDOUT output reserved exclusively for output CSV path
    print(str(out_file.resolve()))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import sys
import os
import datetime
import json
import tempfile
import time
import numpy as np

# Suppress HuggingFace transformers verbosity at the environment level
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ---------------------------------------------------------------------------
# Distogram-conditioning helpers
#
# ESMFold2 exposes per-chain distogram conditioning through
# StructurePredictionInput(distogram_conditioning=[DistogramConditioning(...)]).
# The builder (esm/utils/structure/input_builder.py::compute_distogram_conditioning)
# expects an [n_tokens_in_chain, n_tokens_in_chain] matrix of *distances in
# Angstrom* between the model's representative atoms, which for protein tokens
# is CB (CA for glycine). It is bucketized into 64 bins spanning 2.0-22.0 A and
# injected into the trunk pair representation with an intra-chain-only mask.
#
# Consequences worth remembering:
#   * distances above 22 A saturate in the top bin, so this pins local/domain
#     geometry firmly but only weakly restrains inter-domain motion;
#   * the mask is intra-chain, so naming two chains pins each chain's own fold
#     but NOT their relative orientation;
#   * it is a soft prior, not a hard constraint.
# ---------------------------------------------------------------------------

def chain_token_slices(complex_obj):
    """
    Group token indices of a MolecularComplex by chain, preserving the order in
    which chains first appear.

    Returns a list of (chain_id_string, token_index_array) tuples. The chain id
    string is resolved through complex.metadata.chain_lookup so that it matches
    the chain ids ESMFold2 assigns internally (which is what
    DistogramConditioning.chain_id must key on).
    """
    cids = np.asarray(complex_obj.chain_id)
    try:
        lookup = {int(k): str(v) for k, v in dict(complex_obj.metadata.chain_lookup).items()}
    except Exception:
        lookup = {}

    seen = []
    for c in cids.tolist():
        if c not in seen:
            seen.append(c)

    out = []
    for asym in seen:
        idx = np.flatnonzero(cids == asym)
        out.append((lookup.get(int(asym), str(asym)), idx))
    return out


def token_atom_bounds(complex_obj, token_index):
    """Return (start, end) atom indices for a token. token_to_atoms is [L, 2]."""
    t2a = np.asarray(complex_obj.token_to_atoms).reshape(-1, 2)
    return int(t2a[token_index][0]), int(t2a[token_index][1])


def atom_coords_by_name(complex_obj, token_indices, preferred, fallback=None,
                        ordinal_fallback=None):
    """
    Pull one atom coordinate per token, selected by atom name.

    preferred / fallback are atom names (e.g. "CB" then "CA"). If the complex
    carries no atom_names array, fall back to a positional index into the
    token's atom block (CCD conformer order for standard residues is
    N, CA, C, O, CB, ...), which is what ordinal_fallback selects.

    Returns an [n_tokens, 3] float64 array; missing atoms are NaN.
    """
    pos = np.asarray(complex_obj.atom_positions, dtype=np.float64)
    names = getattr(complex_obj, "atom_names", None)
    coords = np.full((len(token_indices), 3), np.nan, dtype=np.float64)

    for k, ti in enumerate(token_indices):
        start, end = token_atom_bounds(complex_obj, int(ti))
        if end <= start:
            continue

        if names is not None:
            block = [str(x).strip().upper() for x in np.asarray(names)[start:end]]
            sel = None
            if preferred in block:
                sel = start + block.index(preferred)
            elif fallback is not None and fallback in block:
                sel = start + block.index(fallback)
            if sel is not None:
                coords[k] = pos[sel]
                continue

        if ordinal_fallback is not None:
            off = min(ordinal_fallback, end - start - 1)
            coords[k] = pos[start + off]

    return coords


def representative_atom_coords(complex_obj, token_indices):
    """
    CB coordinate per token, falling back to CA (glycine, or any residue whose
    CB is absent). Mirrors ESMFold2's compute_representative_atoms().
    """
    return atom_coords_by_name(
        complex_obj, token_indices, preferred="CB", fallback="CA", ordinal_fallback=1
    )


def ca_coords(complex_obj, token_indices):
    """CA coordinate per token (used only for the verification RMSD)."""
    return atom_coords_by_name(
        complex_obj, token_indices, preferred="CA", fallback="CB", ordinal_fallback=1
    )


def pairwise_distance_matrix(coords):
    """Symmetric [N, N] Euclidean distance matrix in Angstrom."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def kabsch_rmsd(mobile, target):
    """
    RMSD between two [N, 3] coordinate sets after optimal superposition.
    Used to check that a conditioned chain kept the fold it was given, which is
    a question about internal geometry, not about the global frame.
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape:
        return float("nan")

    ok = ~(np.isnan(mobile).any(axis=-1) | np.isnan(target).any(axis=-1))
    P, Q = mobile[ok], target[ok]
    if len(P) < 3:
        return float("nan")

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = (R @ Pc.T).T - Qc
    return float(np.sqrt(np.sum(diff * diff) / len(P)))


def calculate_iplddt(atom_positions, token_to_atoms, chain_ids, plddts, cutoff=4.0):
    """
    Calculates the Interface pLDDT (I-pLDDT) score by computing the average pLDDT
    value for residues within a specified distance cutoff (default 4.0 A) across
    different chains.
    """
    try:
        coords = np.array(atom_positions, dtype=np.float64)
        cids = np.array(chain_ids)
        plddt_arr = np.array(plddts, dtype=np.float64)
        t2a = np.asarray(token_to_atoms).reshape(-1, 2)

        L = len(cids)
        interface_mask = np.zeros(L, dtype=bool)

        # Per-token atom blocks. token_to_atoms is [L, 2] of (start, end) indices,
        # so the atoms of token i are coords[start:end] -- indexing with the pair
        # itself would pick up exactly two atoms and run off the end of the array
        # on the final token.
        blocks = []
        centroids = np.full((L, 3), np.nan)
        radii = np.zeros(L)
        for i in range(L):
            start, end = int(t2a[i][0]), int(t2a[i][1])
            atoms_i = coords[start:end]
            blocks.append(atoms_i)
            if len(atoms_i):
                centroids[i] = atoms_i.mean(axis=0)
                radii[i] = np.sqrt(np.max(np.sum((atoms_i - centroids[i]) ** 2, axis=-1)))

        # O(L^2) scan over cross-chain token pairs, with an exact bounding-sphere
        # prefilter so that the expensive all-atom distance block is only computed
        # for pairs that can possibly be within the cutoff.
        for i in range(L):
            atoms_i = blocks[i]
            if len(atoms_i) == 0:
                continue

            for j in range(i + 1, L):
                if cids[i] == cids[j]:
                    continue
                if interface_mask[i] and interface_mask[j]:
                    continue
                atoms_j = blocks[j]
                if len(atoms_j) == 0:
                    continue

                gap = np.linalg.norm(centroids[i] - centroids[j]) - radii[i] - radii[j]
                if gap > cutoff:
                    continue

                # Compute pairwise squared distances between all atoms of residue i and j
                dist_sq = np.sum((atoms_i[:, None, :] - atoms_j[None, :, :]) ** 2, axis=-1)
                min_dist = np.sqrt(np.min(dist_sq))

                if min_dist <= cutoff:
                    interface_mask[i] = True
                    interface_mask[j] = True

        if np.any(interface_mask):
            return float(np.mean(plddt_arr[interface_mask]))
        else:
            # Fallback for models lacking any interacting residues within the cutoff
            return 1.0
    except Exception:
        return 1.0


def convert_mmcif_to_pdb(mmcif_str, query_name, metrics=None, params=None, extra_remarks=None):
    """
    Parses the mmCIF output string generated by ESMFold2 and formats it
    into a standard 80-column PDB file with REMARK headers, including
    metrics and execution parameters.
    """
    lines = mmcif_str.split('\n')
    pdb_lines = []

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    pdb_lines.append(f"REMARK    QUERY NAME: {query_name}")
    pdb_lines.append(f"REMARK    STRUCTURE PREDICTED USING ESMFOLD2, {current_date}")

    if metrics:
        if 'pLDDT' in metrics:
            pdb_lines.append(f"REMARK    PLDDT: {metrics['pLDDT']:.5f}")
        if 'pTM' in metrics:
            pdb_lines.append(f"REMARK    PTM: {metrics['pTM']:.5f}")
        if 'ipTM' in metrics:
            pdb_lines.append(f"REMARK    IPTM: {metrics['ipTM']:.5f}")
        if 'ipLDDT' in metrics:
            pdb_lines.append(f"REMARK    I-PLDDT: {metrics['ipLDDT']:.5f}")

    if params:
        pdb_lines.append("REMARK    --- EXECUTION PARAMETERS ---")
        for key, val in params.items():
            pdb_lines.append(f"REMARK    {key}: {val}")

    if extra_remarks:
        for line in extra_remarks:
            pdb_lines.append(f"REMARK    {line}")

    atom_serial = 1
    chain_map = {}
    chain_letters = "HLABCDEFGIJKMNOPQRSTUVWXYZ"

    for line in lines:
        line = line.strip()
        if line.startswith("ATOM") or line.startswith("HETATM"):
            parts = line.split()
            if len(parts) < 18:
                continue

            record = parts[0]
            element = parts[1]
            atom_name = parts[2]
            res_name = parts[4]
            chain_id_full = parts[5]
            res_seq = int(parts[7])
            b_factor = float(parts[13])
            occupancy = float(parts[14])
            x = float(parts[15])
            y = float(parts[16])
            z = float(parts[17])

            if chain_id_full not in chain_map:
                if chain_id_full.endswith("_H") and "H" not in chain_map.values():
                    chain_map[chain_id_full] = "H"
                elif chain_id_full.endswith("_L") and "L" not in chain_map.values():
                    chain_map[chain_id_full] = "L"
                elif chain_id_full == "H" and "H" not in chain_map.values():
                    chain_map[chain_id_full] = "H"
                elif chain_id_full == "L" and "L" not in chain_map.values():
                    chain_map[chain_id_full] = "L"
                else:
                    for letter in chain_letters:
                        if letter not in chain_map.values():
                            chain_map[chain_id_full] = letter
                            break
                    else:
                        chain_map[chain_id_full] = "A"

            chain_id = chain_map[chain_id_full]

            if len(atom_name) < 4:
                name_fmt = f" {atom_name:<3}"
            else:
                name_fmt = f"{atom_name[:4]:<4}"

            pdb_line = f"{record:<6}{atom_serial:>5} {name_fmt} {res_name:>3} {chain_id:1}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}          {element:>2}"
            pdb_lines.append(pdb_line)
            atom_serial += 1

    pdb_lines.append("END")
    return "\n".join(pdb_lines)


def main():
    # --- STDOUT PROTECTION ---
    # Save the true standard output so we can explicitly route final payloads to it.
    # We redirect standard stdout to stderr to catch any rogue print() statements
    # from external libraries (like model checkpoint warnings) before they corrupt
    # piped commands or JSON mode standard outputs.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    parser = argparse.ArgumentParser(description="Run ESMFold2 and output a structure and CSV metrics.")
    parser.add_argument("-i", "--inputfile", default=None, 
                        help="Input FASTA file containing one or more protein sequences")
    parser.add_argument("--json", default=None, 
                        help="JSON payload with heavy, light, and antigen sequences to evaluate directly")
    parser.add_argument("--extra-sequences", default=None, metavar="NAME1:SEQ1::NAME2:SEQ2",
                        help="Double-colon-separated named extra protein sequence(s) to append to the input FASTA records "
                             "before validating --distogram-conditioning. Each item must be NAME:SEQUENCE. "
                             "Example: --extra-sequences nano1:QVQLVES::nano2:EVQLVES")
    parser.add_argument("--tag", default="esf2", 
                        help="Tag for output files (default: esf2)")
    parser.add_argument("--outdir", default=None, 
                        help="Output directory (default: same directory as the input file)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print detailed processing information to STDERR")
    parser.add_argument("--refresh", action="store_true", 
                        help="Force recalculation even if the output file already exists")
    parser.add_argument("--budget-min", type=float, default=None,
                        help="Compute budget in minutes (enforces a wall-clock limit on diffusion sampling loops)")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Produce a full ensemble of N predictions instead of one")
    parser.add_argument("--model", default="biohub/ESMFold2",
                        choices=("biohub/ESMFold2", "biohub/ESMFold2-Fast"),
                        help="Local Hugging Face model checkpoint")
    parser.add_argument("--num-loops", type=int, default=20,
                        help="Recurrent folding loops; reduce first to lower runtime")
    parser.add_argument("--num-sampling-steps", type=int, default=100,
                        help="Requested diffusion schedule steps; 68 is a strong efficient starting point")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Enable activation checkpointing on the model transformer blocks to reduce peak VRAM usage")
    parser.add_argument("--msa_max_depth", type=int, default=1024,
                        help="Maximum number of MSA sequences to randomly subsample each loop to prevent memory explosion (default: 1024)")
    parser.add_argument("--detailed-output", action="store_true",
                        help="Produce detailed per-residue/atom scores and matrices (PAE, distogram) in output files or JSON")
    parser.add_argument("--distogram-conditioning", default=None, metavar="FIRST:SECOND",
                        help="Colon-separated list of FASTA record names to fold FIRST, on their own. Their predicted "
                             "CB-CB distance matrices are then supplied to ESMFold2 as per-chain distogram "
                             "conditioning while the full complex is folded, so those chains keep the fold they were "
                             "given instead of being deformed by the other chains. "
                             "Example: --distogram-conditioning antigen, or --distogram-conditioning heavy:light. "
                             "Note the conditioning mask is intra-chain: naming several chains pins each of their "
                             "folds but not their relative orientation.")

    args = parser.parse_args()

    def vprint(*pargs, **kwargs):
        if args.verbose:
            print(*pargs, file=sys.stderr, **kwargs)

    def eprint(*pargs, **kwargs):
        print(*pargs, file=sys.stderr, **kwargs)

    if not args.inputfile and not args.json:
        eprint("Error: Must provide either -i/--inputfile or --json.")
        sys.exit(1)

    is_json_mode = False
    temp_fasta_path = None
    basename = "complex"

    # 1. Resolve Paths & Input Mode
    if args.json:
        is_json_mode = True
        try:
            payload = json.loads(args.json)
        except json.JSONDecodeError as e:
            eprint(f"Error parsing JSON payload: {e}")
            sys.exit(1)
        
        heavy = payload.get("heavy", "")
        light = payload.get("light", "")
        antigen = payload.get("antigen", "")
        basename = payload.get("name", "json_query")
        
        linker = "GGGSGGGSGGGSGGGS"
        mab_seq = heavy + linker + light
        
        # Create temporary FASTA
        fd, temp_fasta_path = tempfile.mkstemp(suffix=".fasta", text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(f">mAb\n{mab_seq}\n")
            f.write(f">antigen\n{antigen}\n")
        
        in_path = temp_fasta_path
    else:
        in_path = os.path.abspath(args.inputfile)
        basename = os.path.splitext(os.path.basename(in_path))[0]
        
        if args.outdir is not None:
            out_dir = os.path.abspath(args.outdir)
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(in_path)

        out_csv = os.path.join(out_dir, f"{basename}.{args.tag}.csv")

        # 2. Check Refresh using the CSV as the primary target (File Mode Only)
        if not args.refresh:
            if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
                vprint(f"Output file {out_csv} already exists. Skipping computation.")
                # Route exclusively to true STDOUT
                print(out_csv, file=original_stdout)
                sys.exit(0)

    # 3. Parse FASTA
    if not os.path.exists(in_path):
        eprint(f"Error: Input file '{in_path}' does not exist.")
        sys.exit(1)

    sequences = []
    with open(in_path, "r") as f:
        seq_id = ""
        seq_data = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id:
                    sequences.append((seq_id, "".join(seq_data)))
                seq_id = line[1:].split()[0]
                seq_data = []
            elif line:
                seq_data.append(line)
        if seq_id:
            sequences.append((seq_id, "".join(seq_data)))

    # 3a. Append command-line extra sequences before any downstream validation.
    #     Format: --extra-sequences NAME1:SEQ1::NAME2:SEQ2
    #     These named records are added to `sequences` before --distogram-conditioning
    #     is resolved, so they count as normal chains when deciding whether every chain
    #     has been conditioned.
    if args.extra_sequences:
        extra_specs = [part.strip() for part in args.extra_sequences.split("::") if part.strip()]
        if not extra_specs:
            eprint("Error: --extra-sequences was provided but no NAME:SEQUENCE pair(s) could be parsed.")
            sys.exit(1)

        existing_ids = {sid for sid, _ in sequences}
        added_extra = []
        for spec in extra_specs:
            if ":" not in spec:
                eprint(f"Error: invalid --extra-sequences entry '{spec}'. Expected NAME:SEQUENCE, "
                       "with entries separated by double colon, e.g. NAME1:SEQ1::NAME2:SEQ2.")
                sys.exit(1)

            seq_id, seq = spec.split(":", 1)
            seq_id = seq_id.strip()
            seq = seq.strip()

            if not seq_id:
                eprint(f"Error: invalid --extra-sequences entry '{spec}': NAME is empty.")
                sys.exit(1)
            if not seq:
                eprint(f"Error: invalid --extra-sequences entry '{spec}': SEQUENCE is empty.")
                sys.exit(1)
            if seq_id in existing_ids:
                eprint(f"Error: duplicate sequence name '{seq_id}' from --extra-sequences. "
                       "Extra sequence names must be unique and must not duplicate FASTA record names.")
                sys.exit(1)

            existing_ids.add(seq_id)
            sequences.append((seq_id, seq))
            added_extra.append(seq_id)

        vprint(f"Added {len(added_extra)} extra sequence(s) from --extra-sequences: {', '.join(added_extra)}")

    if not sequences:
        eprint("Error: No sequences found in the input FASTA or --extra-sequences.")
        sys.exit(1)

    # 3b. Resolve --distogram-conditioning against all parsed sequence record names,
    #     including named records appended from --extra-sequences.
    #     Validated here, before the model is loaded, so mistakes fail fast.
    dc_names = []
    if args.distogram_conditioning:
        raw = args.distogram_conditioning.replace(",", ":")
        dc_names = [n.strip() for n in raw.split(":") if n.strip()]
        if not dc_names:
            eprint("Error: --distogram-conditioning was given but no record names could be parsed.")
            sys.exit(1)

        available = [sid for sid, _ in sequences]
        if len(set(available)) != len(available):
            eprint("Error: --distogram-conditioning requires unique FASTA record names; "
                   f"the input contains duplicates: {sorted({s for s in available if available.count(s) > 1})}")
            sys.exit(1)

        missing = [n for n in dc_names if n not in available]
        if missing:
            eprint(f"Error: --distogram-conditioning name(s) not found in parsed input/extra sequences: {', '.join(missing)}")
            eprint(f"       Available record names, including --extra-sequences: {', '.join(available)}")
            sys.exit(1)

        seen_dc = []
        for n in dc_names:
            if n not in seen_dc:
                seen_dc.append(n)
        dc_names = seen_dc

        if len(dc_names) == len(available):
            eprint("Warning: --distogram-conditioning names every chain in the input, so there is no "
                   "unconditioned chain left to fold. Proceeding as a self-conditioning sanity check "
                   "(the reported RMSDs should be near zero if the conditioning is being applied).")
        elif len(dc_names) > 1:
            eprint("Warning: distogram conditioning is applied with an intra-chain mask, so naming "
                   f"{len(dc_names)} chains pins each of their folds but leaves their relative "
                   "orientation free in the second pass.")

    # 4. Process Budget Flag & Initialize Run Parameters
    budget_seconds = None
    if args.budget_min:
        budget_seconds = float(args.budget_min) * 60.0
        max_iterations = 999999  # Effectively infinite; breaks on time limit
        vprint(f"Time budget set to {args.budget_min} minutes. Will run sequential samples until budget is exhausted.")
    else:
        max_iterations = args.ensemble
        vprint(f"No budget provided. Defaulting to {max_iterations} diffusion sample(s).")
        
    num_loops = args.num_loops
    num_sampling_steps = args.num_sampling_steps
    total_residues = sum(len(seq.replace("|", "").replace(":", "")) for _, seq in sequences)
    vprint(f"Input: {len(sequences)} chain(s), {total_residues} total residues")
    if total_residues > 768:
        eprint(f"Note: {total_residues} residues exceeds the hosted API limit of 768; using local inference.")
        eprint("Memory-conservative order: ESMFold2-Fast, MSA depth 1, one ensemble member, "
               "10 loops/68 steps; then increase loops, steps, and MSA depth only after a successful run.")
    
    run_params = {
        "BUDGET_MIN": args.budget_min if args.budget_min is not None else "None",
        "ENSEMBLE_SIZE": args.ensemble,
        "NUM_LOOPS": num_loops,
        "NUM_SAMPLING_STEPS": num_sampling_steps,
        "MSA_MAX_DEPTH": args.msa_max_depth,
        "ACTIVATION_CHECKPOINTING": args.activation_checkpointing,
        "MODEL": args.model
    }
    if dc_names:
        run_params["DISTOGRAM_CONDITIONING"] = ":".join(dc_names)

    # 5. Load Model
    vprint("Loading ESM modules and biohub/ESMFold2 on GPU...")
    try:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput, ESMFold2InputBuilder
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from transformers.models.esmc.modeling_esmc import UnifiedTransformerBlock as TransformerBlock
        if dc_names:
            from esm.models.esmfold2 import DistogramConditioning
    except ImportError as e:
        eprint(f"Error importing ESM modules: {e}")
        if is_json_mode and temp_fasta_path and os.path.exists(temp_fasta_path):
            os.remove(temp_fasta_path)
        sys.exit(1)

    try:
        vprint(f"Loading local checkpoint {args.model}...")
        model = ESMFold2Model.from_pretrained(args.model).cuda().eval()

        if args.activation_checkpointing:
            vprint("Applying activation checkpointing to TransformerBlock layers to optimize memory...")
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda module: isinstance(module, TransformerBlock),
            )

        # 5b. PASS 1 (optional): fold the conditioning chains on their own and turn
        #     the result into per-chain distogram conditioning for pass 2.
        dc_entries = None
        dc_reference_ca = {}
        dc_chain_summary = []
        stage1_pdb_str = None

        if dc_names:
            dc_set = set(dc_names)
            stage1_inputs = [
                ProteinInput(id=sid, sequence=seq) for sid, seq in sequences if sid in dc_set
            ]
            expected_residues = sum(
                len(seq.replace("|", "").replace(":", "")) for sid, seq in sequences if sid in dc_set
            )

            vprint(f"[pass 1/2] Folding conditioning chain(s) {':'.join(dc_names)} on their own "
                   f"({expected_residues} residues)...")
            stage1_result = ESMFold2InputBuilder().fold(
                model,
                StructurePredictionInput(sequences=stage1_inputs),
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=1,
                seed=0,
                msa_max_depth=args.msa_max_depth
            )

            stage1_plddt = float(stage1_result.plddt.mean()) if stage1_result.plddt is not None else 0.0
            stage1_ptm = float(stage1_result.ptm) if stage1_result.ptm is not None else 0.0
            vprint(f"[pass 1/2] Reference fold: pLDDT {stage1_plddt:.3f}, pTM {stage1_ptm:.3f}")

            if getattr(stage1_result.complex, "atom_names", None) is None:
                eprint("Warning: pass-1 complex carries no atom names; falling back to CA rather than CB "
                       "as the representative atom for the conditioning distogram.")

            dc_entries = []
            total_tokens = 0
            for chain_id_str, tok_idx in chain_token_slices(stage1_result.complex):
                rep = representative_atom_coords(stage1_result.complex, tok_idx)
                if np.isnan(rep).any():
                    raise RuntimeError(
                        f"Could not resolve a representative atom for every residue of pass-1 chain "
                        f"'{chain_id_str}' ({int(np.isnan(rep).any(axis=-1).sum())} missing); refusing to "
                        f"build a distogram containing NaNs, since the builder masks every intra-chain pair."
                    )

                dist = pairwise_distance_matrix(rep)
                dc_entries.append(
                    DistogramConditioning(chain_id=chain_id_str, distogram=dist.astype(np.float32))
                )
                dc_reference_ca[chain_id_str] = ca_coords(stage1_result.complex, tok_idx)
                dc_chain_summary.append((chain_id_str, len(tok_idx)))
                total_tokens += len(tok_idx)
                offdiag = dist[dist > 0]
                if offdiag.size:
                    vprint(f"[pass 1/2] Chain '{chain_id_str}': {len(tok_idx)} tokens, "
                           f"CB-CB distances {offdiag.min():.2f}-{dist.max():.2f} A "
                           f"({float((dist > 22.0).mean() * 100):.1f}% above the 22 A top bin)")
                else:
                    vprint(f"[pass 1/2] Chain '{chain_id_str}': {len(tok_idx)} token(s)")

            if total_tokens != expected_residues:
                eprint(f"Warning: pass 1 produced {total_tokens} tokens for the conditioning chains but the "
                       f"FASTA records contain {expected_residues} residues. Distogram shapes are taken from "
                       f"the pass-1 prediction, so pass 2 will raise if the chain composition differs.")

            run_params["DC_STAGE1_CHAINS"] = ";".join(f"{c}:{n}" for c, n in dc_chain_summary)
            run_params["DC_STAGE1_PLDDT"] = f"{stage1_plddt:.5f}"

            stage1_pdb_str = convert_mmcif_to_pdb(
                stage1_result.complex.to_mmcif(),
                query_name=f"{basename}_dc_reference",
                metrics={"pLDDT": stage1_plddt, "pTM": stage1_ptm},
                params={"STAGE": "1 (distogram-conditioning reference)",
                        "NUM_LOOPS": num_loops,
                        "NUM_SAMPLING_STEPS": num_sampling_steps,
                        "SEED": 0}
            )

            if not is_json_mode:
                out_stage1_pdb = os.path.join(out_dir, f"{basename}.{args.tag}.dc_reference.pdb")
                with open(out_stage1_pdb, "w") as f:
                    f.write(stage1_pdb_str)
                vprint(f"[pass 1/2] Reference structure written to: {out_stage1_pdb}")

        # 6. Predict (Sequential Sampling with Wall-Clock Timer)
        vprint(f"{'[pass 2/2] ' if dc_names else ''}Folding {len(sequences)} sequence(s)...")
        protein_inputs = [ProteinInput(id=sid, sequence=seq) for sid, seq in sequences]
        spi = StructurePredictionInput(sequences=protein_inputs, distogram_conditioning=dc_entries)

        results = []
        start_time = time.time()
        
        for i in range(max_iterations):
            vprint(f"Generating diffusion sample {i+1}...")
            
            sample_result = ESMFold2InputBuilder().fold(
                model, 
                spi, 
                num_loops=num_loops, 
                num_sampling_steps=num_sampling_steps, 
                num_diffusion_samples=1, 
                seed=i,
                msa_max_depth=args.msa_max_depth
            )
            results.append(sample_result)
            
            if budget_seconds is not None:
                elapsed_seconds = time.time() - start_time
                if elapsed_seconds >= budget_seconds:
                    vprint(f"Time budget of {args.budget_min} minutes reached (Elapsed: {elapsed_seconds/60:.2f} min). Halting sampling.")
                    break

        # Log actual number of samples completed
        run_params["ACTUAL_DIFFUSION_SAMPLES"] = len(results)

        # Select the top N models by pTM
        vprint(f"Selecting the top {args.ensemble} model(s) out of {len(results)} sequential samples by pTM...")
        results.sort(key=lambda r: float(r.ptm) if r.ptm is not None else -1.0, reverse=True)
        top_results = results[:args.ensemble]

        # 7 & 8. Extract Metrics, Generate Output Files
        json_payloads = []
        csv_header = "query_name,tag,rank,pLDDT,pTM,ipTM,ipLDDT,structure"
        if dc_names:
            csv_header += ",dc_chains,dc_ca_rmsd_max"
        csv_lines = [csv_header + "\n"]
        
        for rank, result in enumerate(top_results, start=1):
            plddt_arr = result.plddt.cpu().numpy() if result.plddt is not None else np.zeros(len(result.complex.sequence))
            plddt = float(plddt_arr.mean())
            ptm = float(result.ptm) if result.ptm is not None else 0.0
            iptm = float(result.iptm) if result.iptm is not None else 0.0
            
            # Calculate the Interface pLDDT (I-pLDDT)
            iplddt = calculate_iplddt(
                atom_positions=result.complex.atom_positions,
                token_to_atoms=result.complex.token_to_atoms,
                chain_ids=result.complex.chain_id,
                plddts=plddt_arr,
                cutoff=4.0
            )
            
            metrics = {
                "pLDDT": plddt,
                "pTM": ptm,
                "ipTM": iptm,
                "ipLDDT": iplddt
            }
            
            vprint(f"Rank {rank} Metrics -> pLDDT: {plddt:.3f}, pTM: {ptm:.3f}, ipTM: {iptm:.3f}, ipLDDT: {iplddt:.3f}")

            # Verify that the conditioned chains actually kept their pass-1 fold.
            # This is a per-chain, superposition-based CA RMSD, so it measures
            # internal geometry and ignores the rigid placement of the chain in
            # the complex.
            dc_rmsd = {}
            extra_remarks = None
            if dc_names:
                for chain_id_str, tok_idx in chain_token_slices(result.complex):
                    if chain_id_str not in dc_reference_ca:
                        continue
                    dc_rmsd[chain_id_str] = kabsch_rmsd(
                        ca_coords(result.complex, tok_idx), dc_reference_ca[chain_id_str]
                    )

                finite = [v for v in dc_rmsd.values() if np.isfinite(v)]
                dc_rmsd_max = float(max(finite)) if finite else float("nan")
                metrics["dcCaRmsdMax"] = dc_rmsd_max

                extra_remarks = ["--- DISTOGRAM CONDITIONING ---",
                                 f"DC_CHAINS: {':'.join(dc_names)}"]
                for chain_id_str, val in dc_rmsd.items():
                    extra_remarks.append(f"DC_CA_RMSD[{chain_id_str}]: {val:.3f} A")
                extra_remarks.append(f"DC_CA_RMSD_MAX: {dc_rmsd_max:.3f} A")

                vprint("Rank {} conditioned-chain CA RMSD vs pass 1 -> {}".format(
                    rank,
                    ", ".join(f"{c}: {v:.3f} A" for c, v in dc_rmsd.items()) or "n/a"
                ))
                if np.isfinite(dc_rmsd_max) and dc_rmsd_max > 2.0:
                    eprint(f"Warning: rank {rank} conditioned chain(s) moved {dc_rmsd_max:.2f} A CA RMSD from "
                           f"the pass-1 reference. Distogram conditioning is a soft prior, not a hard "
                           f"constraint; consider docking against a fixed antigen if you need it rigid.")

            # Extract Detailed Data (if requested)
            detailed_data = None
            if args.detailed_output:
                vprint(f"Compiling detailed output metrics for Rank {rank}...")
                detailed_data = {
                    "sequence": np.array(result.complex.sequence),
                    "chain_id": result.complex.chain_id,
                    "plddt": plddt_arr,
                    "ptm": np.array(ptm),
                    "iptm": np.array(iptm),
                    "iplddt": np.array(iplddt),
                    "atom_positions": result.complex.atom_positions,
                    "atom_elements": result.complex.atom_elements,
                    "token_to_atoms": result.complex.token_to_atoms
                }
                if result.pae is not None:
                    detailed_data["pae"] = result.pae.cpu().numpy()
                if result.distogram is not None:
                    detailed_data["distogram"] = result.distogram.cpu().numpy()
                if result.pair_chains_iptm is not None:
                    detailed_data["pair_chains_iptm"] = result.pair_chains_iptm.cpu().numpy()
                if dc_names:
                    detailed_data["dc_chains"] = np.array(list(dc_rmsd.keys()))
                    detailed_data["dc_ca_rmsd"] = np.array(list(dc_rmsd.values()), dtype=np.float64)

            query_name_display = f"{basename}_rank{rank}" if args.ensemble > 1 else basename
            raw_mmcif_str = result.complex.to_mmcif()
            standard_pdb_str = convert_mmcif_to_pdb(
                raw_mmcif_str, 
                query_name=query_name_display, 
                metrics=metrics, 
                params=run_params,
                extra_remarks=extra_remarks
            )
            
            if is_json_mode:
                out_payload = {
                    "rank": rank,
                    "pdb": standard_pdb_str,
                    "metrics": metrics
                }
                if dc_names:
                    out_payload["distogram_conditioning"] = {
                        "chains": dc_names,
                        "stage1_chain_ids": [c for c, _ in dc_chain_summary],
                        "ca_rmsd_vs_stage1": dc_rmsd,
                        "reference_pdb": stage1_pdb_str
                    }
                if args.detailed_output:
                    json_detailed = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in detailed_data.items()}
                    out_payload["detailed"] = json_detailed
                
                json_payloads.append(out_payload)
            else:
                # File mode
                suffix = f"_rank{rank}" if args.ensemble > 1 else ""
                
                out_pdb = os.path.join(out_dir, f"{basename}.{args.tag}{suffix}.pdb")
                with open(out_pdb, "w") as f:
                    f.write(standard_pdb_str)
                
                csv_row = f"{basename},{args.tag},{rank},{plddt:.3f},{ptm:.3f},{iptm:.3f},{iplddt:.3f},{out_pdb}"
                if dc_names:
                    csv_row += f",{':'.join(dc_names)},{metrics['dcCaRmsdMax']:.3f}"
                csv_lines.append(csv_row + "\n")
                
                if args.detailed_output:
                    out_npz = os.path.join(out_dir, f"{basename}.{args.tag}{suffix}_detailed.npz")
                    np.savez_compressed(out_npz, **detailed_data)
                    
                    out_detailed_csv = os.path.join(out_dir, f"{basename}.{args.tag}{suffix}_per_residue.csv")
                    with open(out_detailed_csv, "w") as f:
                        f.write("token_index,chain_id,residue_name,plddt,tag\n")
                        seq = detailed_data["sequence"]
                        cids = detailed_data["chain_id"]
                        plddts = detailed_data["plddt"]
                        for idx in range(len(seq)):
                            f.write(f"{idx},{cids[idx]},{seq[idx]},{plddts[idx]:.4f},{args.tag}\n")
                            
                    vprint(f"Detailed outputs written to: \n  {out_npz}\n  {out_detailed_csv}")

        # 9. Output final master payload or CSV
        if is_json_mode:
            if args.ensemble == 1:
                print(json.dumps(json_payloads[0]), file=original_stdout)
            else:
                print(json.dumps(json_payloads), file=original_stdout)
        else:
            with open(out_csv, "w") as f:
                f.writelines(csv_lines)
            print(out_csv, file=original_stdout)

    except Exception as e:
        eprint(f"Error during processing or writing: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary files 
        if is_json_mode and temp_fasta_path and os.path.exists(temp_fasta_path):
            os.remove(temp_fasta_path)

if __name__ == "__main__":
    main()

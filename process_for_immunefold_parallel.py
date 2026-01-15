import pandas as pd
import subprocess
import sys
import re
import shutil
import multiprocessing
from pathlib import Path
from functools import partial

# Check for tqdm (optional but recommended for parallel scripts)
try:
    from tqdm import tqdm
except ImportError:
    # distinct shim if tqdm is missing
    def tqdm(iterable, **kwargs):
        return iterable

# =========================
# GLOBAL CONSTANTS
# =========================
HLA_A0201_ECD = (
    "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKV"
    "KAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWT"
    "AADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"
)

HLA_A0201_FULL = (
    "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKV"
    "KAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWT"
    "AADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"
    "LRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLP"
    "KPLTLRWE"
)

# =========================
# Trimming Helper
# =========================
def trim_tcr_for_structure(seq, chain_type, debug_id):
    """
    Trims sequence and prints REASON if it fails.
    """
    if not seq:
        # print(f"[{debug_id}] {chain_type} REJECTED: Sequence is empty.")
        return None
    
    original_len = len(seq)
    if original_len < 50:
        # print(f"[{debug_id}] {chain_type} REJECTED: Sequence too short (<50 AA).")
        return None

    # --- 1. Trim C-terminus (Remove Constant Region) ---
    j_motif = list(re.finditer(r'([FYW]G[A-Z]G)|(FARG)', seq))
    
    if len(j_motif) > 0:
        cut_point = j_motif[-1].start() + 15
        cut_point = min(cut_point, len(seq))
        seq = seq[:cut_point]
    else:
        # print(f"***[{debug_id}] {chain_type} ABNORMAL: J-motif ([FYW]GxG) or FARG not found.")
        # Optional: return None here if strict
        pass

    # --- 2. Trim N-terminus (Remove Leader Peptide) ---
    if seq.startswith("M"):
        found_start = False
        for i in range(15, 26):
            if i < len(seq) and seq[i] in ['Q', 'E', 'D', 'A']:
                seq = seq[i:]
                found_start = True
                break
        
        if not found_start:
            seq = seq[20:]
    
    # Final length check
    if len(seq) > 256:
        # print(f"[{debug_id}] {chain_type} REJECTED: Trimmed sequence too long ({len(seq)} > 256).")
        return None

    return seq


# =========================
# Stitchr wrapper
# =========================
def run_stitchr(v_gene, j_gene, cdr3, debug_id, chain_type):
    """Run stitchr and return sequence or None."""
    
    v_clean = v_gene.replace("_", "/")
    j_clean = j_gene.replace("_", "/")

    cmd = [
        "stitchr",
        "-v", v_clean,
        "-j", j_clean,
        "-cdr3", cdr3,
        "-s", "HUMAN",
        "-m", "aa", "-sw"
    ]

    try:
        # Use simple subprocess call
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            # We construct an error string to return so the main process can print it
            # to avoid interleaved printing in parallel
            return None, f"[{debug_id}] {chain_type} STITCHR FAILED: {result.stderr.strip()}"

        seq = ""
        for line in result.stdout.splitlines():
            if not line.startswith(">"):
                seq += line.strip()
        
        if not seq:
            return None, f"[{debug_id}] {chain_type} STITCHR OK but OUTPUT EMPTY."
            
        return seq, None

    except Exception as e:
        return None, f"[{debug_id}] {chain_type} EXCEPTION: {e}"


# =========================
# Worker Function (Runs in parallel)
# =========================
def process_single_row(args):
    """
    Worker function to process a single TCR entry.
    Args: (index, row_data_dict)
    Returns: Dictionary with results or failure info.
    """
    index, row = args
    rec_id = f"Row_{index}"
    final_id = f"TCR{index}"

    # Extract Data
    trav = row.get("TRAV_IMGT")
    traj = row.get("TRAJ_IMGT")
    trbv = row.get("TRBV_IMGT")
    trbj = row.get("TRBJ_IMGT")
    cdr3_a = row.get("cdr3_alpha_aa")
    cdr3_b = row.get("cdr3_beta_aa")
    peptide = row.get("epitope_aa")

    # Check for NaN
    if any(pd.isna([trav, traj, trbv, trbj, cdr3_a, cdr3_b, peptide])):
        return {'status': 'skip', 'msg': f"[{rec_id}] SKIPPED: Missing Data (NaN)"}

    # 1. Run Stitchr (Alpha)
    raw_alpha, err = run_stitchr(trav, traj, cdr3_a, rec_id, "ALPHA")
    if not raw_alpha:
        return {'status': 'fail', 'msg': err}

    # 2. Run Stitchr (Beta)
    raw_beta, err = run_stitchr(trbv, trbj, cdr3_b, rec_id, "BETA")
    if not raw_beta:
        return {'status': 'fail', 'msg': err}

    # Check for stop codons
    if '*' in raw_alpha or '*' in raw_beta:
        return {'status': 'fail', 'msg': f"[{rec_id}] Stop codon (*) found in stitchr output."}

    # Prepare Full Output Entry
    full_entry = (
        f">{final_id}_B_A_P_M\n"
        f"{raw_beta}:{raw_alpha}:{peptide}:{HLA_A0201_FULL}\n"
    )

    shortMHC_entry = (
        f">{final_id}_B_A_P_M\n"
        f"{raw_beta}:{raw_alpha}:{peptide}:{HLA_A0201_ECD}\n"
    )

    # 3. Trim Sequences
    clean_alpha = trim_tcr_for_structure(raw_alpha, "ALPHA", rec_id)
    clean_beta  = trim_tcr_for_structure(raw_beta, "BETA", rec_id)

    if not clean_alpha or not clean_beta:
        return {'status': 'fail', 'msg': f"[{rec_id}] Trimming failed (too short or format issues)."}

    # Prepare Final (Trimmed) Output Entry
    final_seq_str = f"{clean_beta}:{clean_alpha}:{peptide}:{HLA_A0201_ECD}"
    seq_len = len(final_seq_str)
    
    trimmed_entry = (
        f">{final_id}_B_A_P_M\n"
        f"{final_seq_str}\n"
    )

    log_msg = f"[{rec_id}] SUCCESS! Total length:{seq_len}"
    if not (415 < seq_len < 450):
        log_msg += " (SEQLEN CHECK FAIL?)"

    return {
        'status': 'success',
        'full_entry': full_entry,
        'trimmed_entry': trimmed_entry,
        'short_mhc': shortMHC_entry,
        'msg': log_msg
    }


# =========================
# Main Processor
# =========================
def process_tcr_csv(csv_file, output_fasta, max_rows=None, num_workers=None):
    
    # 1. Verify Stitchr exists
    if shutil.which("stitchr") is None:
        print("ERROR: 'stitchr' is not installed or not in your PATH.")
        return

    print(f"Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if max_rows:
        df = df.head(max_rows)

    # Filter HLA-A*02
    df = df[df["hla_long"].str.contains("A\\*02", na=False)]
    print(f"HLA-A*02 Rows to process: {len(df)}")
    
    if len(df) == 0:
        print("ERROR: No rows matched HLA-A*02.")
        return

    # Prepare data for parallel processing
    # Convert dataframe to list of tuples: (original_index, row_dict)
    # This preserves the original index for ID generation
    tasks = [(idx, row.to_dict()) for idx, row in df.iterrows()]

    # Determine workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Starting Parallel Processing with {num_workers} workers...")
    
    output_path = Path(output_fasta + '.fasta')
    full_output_path = Path(output_fasta + '_no_trim.fasta')
    shortMHC_output_path = Path(output_fasta + '_mhc_trim.fasta')
    
    success_count = 0
    fail_count = 0

    with open(output_path, "w") as fh, open(full_output_path, 'w') as fh_full, open(shortMHC_output_path, 'w') as fh_shortMHC:
        
        # Create Pool
        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap preserves order of input list in the output iterator
            # chunksize can be tweaked, but default is usually fine for subprocess calls
            results = pool.imap(process_single_row, tasks)
            
            # Iterate through results as they complete (in order)
            for res in tqdm(results, total=len(tasks), unit="seq"):
                
                if res['status'] == 'success':
                    fh.write(res['trimmed_entry'])
                    fh_full.write(res['full_entry'])
                    fh_shortMHC.write(res['short_mhc'])
                    success_count += 1
                    # print(res['msg']) # Uncomment if you want spammy success logs
                
                elif res['status'] == 'fail':
                    fail_count += 1
                    print(res['msg']) # Print failures to console
                
                elif res['status'] == 'skip':
                    fail_count += 1
                    # print(res['msg']) # Uncomment if you want skip logs

    print("\n" + "="*40)
    print(f"Total Successful: {success_count}")
    print(f"Total Failed:     {fail_count}")
    print(f"Output File:      {output_fasta}.fasta")
    print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("-o", "--output", default="TCR_B_A_P_M")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("-j", "--workers", type=int, default=None, help="Number of CPU cores to use")
    args = parser.parse_args()

    process_tcr_csv(args.input_csv, args.output, args.max_rows, args.workers)

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

import requests
import io
import re
from Bio import SeqIO

class HLAManager:
    def __init__(self):
        self.sequences = {}
        self.reference_map = {} # Maps 2-field (A*02:01) to full keys (HLA-A*02:01:01:01)

    def load_database(self):
        """Downloads and parses the IMGT/HLA database."""
        url = "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/fasta/hla_prot.fasta"
        print("Downloading IMGT/HLA database...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return

        fasta_file = io.StringIO(response.text)
        
        # Standard offsets
        SIGNAL_PEPTIDE_LEN = 24
        DOMAIN_LEN = 182 # Alpha 1 + Alpha 2
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Extract ID (e.g., HLA-A*02:01:01:01)
            desc_parts = record.description.split(" ")
            full_name = desc_parts[1] if len(desc_parts) > 1 else record.id
            
            # Filter for Classical Class I
            if not full_name.startswith(("HLA-A", "HLA-B", "HLA-C")):
                continue
            
            # Filter quality
            if len(record.seq) < (SIGNAL_PEPTIDE_LEN + DOMAIN_LEN) or full_name.endswith("N"):
                continue

            # Store Sequence
            seq_alpha1_2 = str(record.seq)[SIGNAL_PEPTIDE_LEN : SIGNAL_PEPTIDE_LEN + DOMAIN_LEN]
            self.sequences[full_name] = seq_alpha1_2
            
            # Build Reference Map for fuzzy matching
            # We map the 2-field resolution (e.g., HLA-A*02:01) to the full key
            # This allows O(1) lookup for short names later
            parts = full_name.split(":")
            if len(parts) >= 2:
                two_field_key = ":".join(parts[:2]) # e.g., HLA-A*02:01
                # Only store the first time we see this 2-field key (usually the reference allele)
                if two_field_key not in self.reference_map:
                    self.reference_map[two_field_key] = full_name

        print(f"Database loaded: {len(self.sequences)} alleles.")

    def normalize_name(self, raw_name):
        """
        Standardizes inputs like 'A*02:01', 'HLA-A02:01', 'a*02:01'
        into 'HLA-A*02:01'.
        """
        # 1. Force Uppercase and strip whitespace
        name = raw_name.upper().strip()
        
        # 2. Handle 'HLA' prefix variability
        # Remove 'HLA-' or 'HLA' if present to start clean
        name = re.sub(r^HLA-?, "", name)
        
        # 3. Ensure Gene and Fields are split by '*'
        # If input is 'A02:01', this splits A and 02:01
        if "*" not in name:
            # Assumes format "A02:01" -> Gene is first char
            gene = name[0]
            fields = name[1:]
            name = f"{gene}*{fields}"
        
        # 4. Re-add standard prefix
        standard_name = f"HLA-{name}"
        
        return standard_name

    def get_sequence(self, raw_name):
        """
        Takes a raw name, normalizes it, and finds the Alpha1/2 sequence.
        """
        search_key = self.normalize_name(raw_name)
        
        # Strategy 1: Exact Match (User provided full 8-digit ID)
        if search_key in self.sequences:
            return search_key, self.sequences[search_key]
            
        # Strategy 2: Lookup in Reference Map (User provided 4-digit ID)
        # e.g., User gave "A*02:01", we look up mapping to "HLA-A*02:01:01:01"
        if search_key in self.reference_map:
            full_key = self.reference_map[search_key]
            return full_key, self.sequences[full_key]

        # Strategy 3: Prefix Search (User provided 6-digit or weird format)
        # We iterate keys to find one that starts with the search key
        for db_key in self.sequences:
            if db_key.startswith(search_key):
                return db_key, self.sequences[db_key]
                
        return None, None

manager = HLAManager()
manager.load_database()

# =========================
# GLOBAL CONSTANTS
# =========================
'''
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
'''
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
        print(f"[{debug_id}] {chain_type} REJECTED: Trimmed sequence too long ({len(seq)} > 256).")

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
    hla = row.get("hla_long")

    _, mhc_seq = manager.get_sequence(hla)
    if mhc_seq is None:
        return {'status': 'fail', 'msg': f'No MHC sequence found for HLA allele {hla}'}

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

    shortMHC_entry = (
        f">{final_id}_B_A_P_M\n"
        f"{raw_beta}:{raw_alpha}:{peptide}:{mhc_seq}\n"
    )

    # 3. Trim Sequences
    clean_alpha = trim_tcr_for_structure(raw_alpha, "ALPHA", rec_id)
    clean_beta  = trim_tcr_for_structure(raw_beta, "BETA", rec_id)

    if not clean_alpha or not clean_beta:
        return {'status': 'fail', 'msg': f"[{rec_id}] Trimming failed (too short or format issues)."}

    # Prepare Final (Trimmed) Output Entry
    final_seq_str = f"{clean_beta}:{clean_alpha}:{peptide}:{mhc_seq}"
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
        'short_mhc': mhc_seq,
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
    
    if len(df) == 0:
        print("ERROR: No rows to process")
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
    #full_output_path = Path(output_fasta + '_no_trim.fasta')
    #shortMHC_output_path = Path(output_fasta + '_mhc_trim.fasta')
    
    success_count = 0
    fail_count = 0

    #with open(output_path, "w") as fh, open(full_output_path, 'w') as fh_full, open(shortMHC_output_path, 'w') as fh_shortMHC:
    with open(output_path, "w") as fh:
        
        # Create Pool
        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap preserves order of input list in the output iterator
            # chunksize can be tweaked, but default is usually fine for subprocess calls
            results = pool.imap(process_single_row, tasks)
            
            # Iterate through results as they complete (in order)
            for res in tqdm(results, total=len(tasks), unit="seq"):
                
                if res['status'] == 'success':
                    fh.write(res['trimmed_entry'])
                    #fh_full.write(res['full_entry'])
                    #fh_shortMHC.write(res['short_mhc'])
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

import os
import argparse
import multiprocessing as mp
import logging
import pandas as pd
import csv
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax

# Initialize logging to standard output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_list(name_idx, pdb_dir):
    """Yields full paths to PDB files based on the input list."""
    if not os.path.exists(name_idx):
        logger.error(f"Index file {name_idx} not found.")
        return
    
    names = list(pd.read_csv(name_idx, names=['name'], header=None)['name'])
    for name in names:
        pdb_file = os.path.join(pdb_dir, f"{name}.pdb")
        if os.path.exists(pdb_file):
            yield pdb_file
        else:
            logger.warning(f"File {pdb_file} missing, skipping.")

def combine_chains(pose, chains_indices):
    """Combine specified chains into a single pose. Indices are assumed to be 1-based (Rosetta standard)."""
    combined_pose = rosetta.core.pose.Pose()
    split_poses = pose.split_by_chain() # Returns a vector1, 1-based indexing
    
    for i, chain_index in enumerate(chains_indices):
        # vector1 indexing checks
        if chain_index > len(split_poses):
            raise IndexError(f"Chain index {chain_index} out of bounds for pose with {len(split_poses)} chains.")
            
        chain_pose = split_poses[chain_index]
        if i == 0:
            combined_pose = chain_pose.clone()
        else:
            combined_pose.append_pose_by_jump(chain_pose, combined_pose.total_residue())
            
    return combined_pose

def calc_interface_energy(pose, chain_ids):
    """Calculates dG_separated using InterfaceAnalyzerMover."""
    # Assuming first 2 are TCR, rest are Antigen as per original logic
    tcr_chain_ids = chain_ids[:2]
    antigen_chain_ids = chain_ids[2:]
    
    # Construct interface string (e.g., "AB_C")
    interface_tcr = ''.join(tcr_chain_ids)
    interface_antigen = ''.join(antigen_chain_ids)
    interface_str = f"{interface_tcr}_{interface_antigen}"

    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface_str)
    mover.set_scorefunction(pyrosetta.create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']

def calc_whole_energy(pose, chain_ids):
    """Calculates binding energy by manual subtraction of component energies."""
    # Indices logic preserved from original script
    # Note: These are likely 1-based indices for vector1
    peptide_chain_idx = 3
    if len(chain_ids) == 4:
        TCR_MHC_chain_indices = [1, 2, 4]
    elif len(chain_ids) == 3:
        TCR_MHC_chain_indices = [1, 2]
    else:
        # Fallback or error if unexpected chain count, assuming 3 is peptide
        TCR_MHC_chain_indices = [i for i in range(1, len(chain_ids) + 2) if i != peptide_chain_idx]

    # Create separate poses for components
    split_poses = pose.split_by_chain()
    if peptide_chain_idx > len(split_poses):
         logger.error(f"Cannot extract chain {peptide_chain_idx}, pose only has {len(split_poses)} chains.")
         return 9999.9 # Error code
         
    peptide_pose = split_poses[peptide_chain_idx]
    TCR_MHC_pose = combine_chains(pose, TCR_MHC_chain_indices)
    
    # Create combined pose for total score
    combined_pose = rosetta.core.pose.Pose(peptide_pose)
    combined_pose.append_pose_by_jump(TCR_MHC_pose, combined_pose.total_residue())
    
    scorefxn = rosetta.core.scoring.get_score_function()
    
    total_energy = scorefxn(combined_pose)
    energy_peptide = scorefxn(peptide_pose)
    energy_receptor = scorefxn(TCR_MHC_pose)
    
    binding_energy = total_energy - (energy_peptide + energy_receptor)
    return binding_energy

def process_pdb(pdb_file, args):
    """
    Worker function:
    1. Loads PDB.
    2. Performs N relax steps (sequentially).
    3. Calculates energy after each step.
    4. Returns dictionary of results.
    """
    pdb_name = os.path.basename(pdb_file).replace('.pdb', '')
    chain_ids = pdb_name.split('_')[1:]
    
    logger.info(f"Processing {pdb_name}...")
    
    try:
        pose = pyrosetta.pose_from_pdb(pdb_file)
    except Exception as e:
        logger.error(f"Failed to load {pdb_file}: {e}")
        return None

    scorefxn = pyrosetta.create_score_function('ref2015')
    relax = FastRelax(scorefxn)
    # The user requested '1 relax step at a time'. 
    # Standard FastRelax does multiple cycles. We rely on the outer loop for the requested 'repeats'.
    # We do NOT set -relax:default_repeats globally.
    
    result_row = {'PDB': pdb_name}

    for i in range(args.n_repeats):
        interface_label = f"interface_rep{i + 1}"
        whole_label = f"whole_rep{i + 1}"
        
        try:
            # 1. Relax (Perform 1 relax step/cycle)
            relax.apply(pose)
            
            # 2. Calculate Energy
            '''
            if args.mode == 'interface':
                energy = calc_interface_energy(pose, chain_ids)
            elif args.mode == 'whole':
                energy = calc_whole_energy(pose, chain_ids)
            else:
                energy = 0.0
            '''
            
            result_row[interface_label] = calc_interface_energy(pose, chain_ids)
            result_row[whole_label] = calc_whole_energy(pose, chain_ids)
            logger.info(f"{pdb_name} - {whole_label}: {result_row[whole_label]} - {interface_label}: {result_row[interface_label]}")
            
        except Exception as e:
            logger.error(f"Error during setp {i + 1} for {pdb_name}: {e}")
            result_row[interface_label] = None
            result_row[whole_label] = None

    # Dump the final relaxed structure once at the end
    output_pdb_path = os.path.join(args.pdb_dir, f"{pdb_name}_relaxed_final.pdb")
    try:
        pose.dump_pdb(output_pdb_path)
    except Exception as e:
        logger.warning(f"Failed to dump PDB {output_pdb_path}: {e}")

    return result_row

def main(args):
    # Prepare input list
    input_files = list(parse_list(args.name_idx, args.pdb_dir))
    if not input_files:
        logger.error("No valid PDB files found to process.")
        return

    # Run multiprocessing
    # using starmap to pass (pdb_file, args) tuples
    tasks = [(f, args) for f in input_files]
    
    with mp.Pool(args.cpus) as p:
        results = p.starmap(process_pdb, tasks)

    # Filter out None results (failed loads)
    clean_results = [r for r in results if r is not None]

    # Write to CSV
    if clean_results:
        # Determine columns dynamically based on the first successful result
        # keys will be 'PDB', 'mode_rep1', 'mode_rep2', ...
        cols = ['PDB'] + [f"interface_rep{i+1}" for i in range(args.n_repeats)] + [f"whole_rep{i+1}" for i in range(args.n_repeats)]
        
        df = pd.DataFrame(clean_results)
        # Ensure column order
        df = df[cols] 
        
        df.to_csv(args.output_file, index=False)
        logger.info(f"Successfully wrote results to {args.output_file}")
    else:
        logger.warning("No results were generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pdb_dir', type=str, required=True)
    parser.add_argument('-n', '--name_idx', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=True)
    parser.add_argument('-c', '--cpus', type=int, default=1)
    #parser.add_argument('-m', '--mode', type=str, required=True, choices=['whole', 'interface'])
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    # n_repeats now controls the number of sequential relax+score steps
    parser.add_argument('-r', '--n_repeats', type=int, default=2, help="Number of sequential relax steps")
    args = parser.parse_args()

    # Removed -relax:default_repeats flag to control stepping manually in the loop
    init_flags = '-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
        -ignore_zero_occupancy false -load_PDB_components true -relax:default_repeats 1 -no_fconfig'
    
    if not args.verbose:
        init_flags += ' -mute all'
        
    init(init_flags, silent=True)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    main(args)

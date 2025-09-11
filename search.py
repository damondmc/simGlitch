#!/home/hoitim.cheung/.conda/envs/glitch/bin/python
import os
import glob
import subprocess
from multiprocessing import Pool
import pandas as pd
import numpy as np
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def grid_size(m, T, factor=1):
    """Calculate grid sizes for frequency and its derivatives."""
    df = 2 * np.sqrt(3*m) / np.pi / T
    df1 = 12 * np.sqrt(5*m) / np.pi / T**2
    df2 = 20 * np.sqrt(7*m) / np.pi / T**3
    return [df*factor, df1*factor, df2*factor]

def find_sft_file(i, label, homedir):
    """Find the first .sft file in the data directory for given index and label."""
    sft_pattern = os.path.join(homedir, f'data/{label}/simCW{i}/*.sft')
    sft_files = glob.glob(sft_pattern)
    if not sft_files:
        raise FileNotFoundError(f"No .sft files found in {os.path.join(homedir, f'data/{label}/simCW{i}/')}")
    return sft_files[0]

def run_command(args):
    """Run a single lalpulsar_Weave command."""
    i, homedir, label, setup_file, df, dx = args
    try:
        sft_file = find_sft_file(i, label, homedir)
        command = (
            f"lalpulsar_Weave "
            f"--output-file={homedir}/results/{label}_CW{i}.fts "
            f"--sft-files={sft_file} "
            f"--setup-file={homedir}/metric/{setup_file} "
            f"--semi-max-mismatch=0.2 "
            f"--coh-max-mismatch=0.1 "
            f"--toplist-limit=1000 "
            f"--extra-statistics='coh2F_det,mean2F,coh2F_det,mean2F_det' "
            f"--alpha={df['alpha'][i]}/0 "
            f"--delta={df['delta'][i]}/0 "
            f"--freq={df['f0'][i]-dx[0]}/{2*dx[0]} "
            f"--f1dot={df['f1'][i]-dx[1]}/{2*dx[1]} "
            f"--f2dot={df['f2'][i]-dx[2]}/{2*dx[2]}"
        )
        print(command)
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Command {i} for {label} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command {i} for {label} failed: {e.stderr}")
        return None
    except FileNotFoundError as e:
        logger.error(f"Command {i} for {label} failed: {str(e)}")
        return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run lalpulsar_Weave commands with multiprocessing.")
    parser.add_argument('--label', default='no_glitch', choices=['no_glitch', 'with_glitch'],
                        help="Label for data directory (no_glitch or with_glitch)")
    parser.add_argument('--cpus', type=int, default=16,
                        help="Number of CPU cores to use for multiprocessing (default: all available)")
    parser.add_argument('--setup-file', default='metric_5d.fts',
                        help="Path to setup file relative to homedir (default: metric_5d.fts)")
    parser.add_argument('--homedir', default='/home/hoitim.cheung/glitch/',
                        help="Base directory path (default: /home/hoitim.cheung/glitch/)")
    parser.add_argument('--n', type=int, default=32,
                        help="Number of jobs (default: 32)")
    args = parser.parse_args()
    
    label = args.label
    setup_file = args.setup_file
    n = args.n

    # Configuration
    homedir = args.homedir.rstrip('/')
    m = 0.2
    tcoh = 86400 * 5
    factor = 4
    dx = grid_size(m, tcoh, factor)

    # Load CSV file into DataFrame
    try:
        df = pd.read_csv(os.path.join(homedir, f'data/{label}/signal_glitch_params.csv'))
        logger.info("DataFrame loaded successfully")
        logger.info(f"DataFrame head:\n{df}")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {os.path.join(homedir, f'data/{label}/signal_glitch_params.csv')}")
        return

    # Create output directory
    os.makedirs(os.path.join(homedir, 'results'), exist_ok=True)

    # Prepare arguments for multiprocessing
    command_args = [(i, homedir, label, setup_file, df, dx) for i in range(n)]

    # Run commands in parallel
    try:
        with Pool(processes=args.cpus) as pool:
            pool.map(run_command, command_args)
        logger.info("All commands completed")
    except Exception as e:
        logger.error(f"Multiprocessing failed: {str(e)}")

if __name__ == "__main__":
    main()
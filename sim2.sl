#!/bin/bash 
#SBATCH --job-name=sim_glitch
#SBATCH --mail-user=damoncht@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --account=kriles1
#SBATCH --partition=standard
#SBATCH --output=/scratch/kriles_root/kriles0/damoncht/cwglitch/out/sim_dnu_gauss.out
#SBATCH --error=/scratch/kriles_root/kriles0/damoncht/cwglitch/out/sim_dnu_gauss.err

source /home/damoncht/miniconda3/etc/profile.d/conda.sh

conda activate lalsuite-dev

python simulate.py --n_cpu 8 --label "wg_dnu_nu_1e-6_gaussian_gap" --freq 100.0 --h0 1e-24 --IFOS H1 --sqrtSX 1e-22

echo "Simulation job finished"

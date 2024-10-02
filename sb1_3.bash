#! /bin/bash

#SBATCH --job-name=ML_psnch3


#SBATCH --partition=XRV-Visu  # Partition Specification
#SBATCH  --nodelist=wombele16              # Node Requested
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                  # Run a single task	
#SBATCH --cpus-per-task=10        # Number of CPU cores per task
#SBATCH	--mem=2G	


module load anaconda/anaconda2
source activate /SCRATCH/psnchimie/chimie_env/
conda activate my_rdkit

cd Desktop/BBB_data/
omp_threads=$SLURM_CPUS_PER_TASK

srun python data_setup.py
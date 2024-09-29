#! /bin/bash

#SBATCH --job-name=ML_psnch3


#SBATCH --partition=All  # Partition Specification
#SBATCH  --nodelist=wombele26              # Node Requested
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                  # Run a single task	
#SBATCH --cpus-per-task=50        # Number of CPU cores per task
#SBATCH	--mem=30G	


module load anaconda/anaconda2
source activate /SCRATCH/psnchimie/chimie_env/

cd /SCRATCH/psnchimie/chimie_env/bin/
omp_threads=$SLURM_CPUS_PER_TASK

srun jupyter nbconvert --to notebook --execute /SCRATCH/psnchimie/chimie_env/bin/results/jobb6/job6_12.ipynb 
#!/bin/bash
#SBATCH --job-name=JoeyLLM               # Job name
#SBATCH --output=BabyJoey/logs/build_%j.log     # Standard output and error log
#SBATCH --ntasks=1                       # Run on one task
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=34G                        # Job memory request
#SBATCH --time=01:00:00                  # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                     # Request one GPU
# SBATCH --gres=gpu:a30:1                # Specific GPU request (uncomment if needed)
# SBATCH --gres=gpu:a6000:1              # Specific GPU request (uncomment if needed)
#SBATCH --partition=gpu                  # Partition name
#SBATCH --nodelist=gpusrv-5              # Specify the node (use only if necessary)

# Log the start time and job info
echo "Started Build on $(date)"
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "_________________________________________________"

# Install dependencies from requirements.txt and then run the main script
singularity exec --nv --bind /home/projects/BabyJoey:/home/u1138167 .singularity/ubuntu.sif \
    bash -c "pip install -r /home/u1138167/requirements.txt && python main.py" > BabyJoey/logs/output_%j.txt 2>&1

echo "_________________________________________________"
# Log the finish time
echo "Server stopped on $(date)"

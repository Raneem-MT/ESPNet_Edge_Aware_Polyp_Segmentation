#!/bin/bash
#SBATCH --job-name=test-edge-array   # Job name
#SBATCH --partition=gpu             # Partition (queue) name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --gres=gpu:1                # Number of GPUs per task
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --time=01:00:00             # Time limit (HH:MM:SS)
#SBATCH --mem=8G                    # Memory per node
#SBATCH --array=0-1                # Task array indices (0 to 11 for 10 tasks)
#SBATCH --mail-type=BEGIN,END       # Send email at the start and end of the job
#SBATCH --mail-user=scrmat@leeds.ac.uk  # Your email address

# Load the Conda environment
module load miniforge
source activate FSPNet

# Define the arrays of checkpoints and result directories

CKPTS=(#add your ckpt path here)

RESULTS=(#add your results path here)

# Task ID provided by Slurm (0-based indexing)
TASK_ID=$SLURM_ARRAY_TASK_ID

# Select the checkpoint and result path for this task
CKPT_PATH=${CKPTS[$TASK_ID]}
RESULT_PATH=${RESULTS[$TASK_ID]}

# Output for debugging
echo "SLURM_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Checkpoint Path: $CKPT_PATH"
echo "Result Path: $RESULT_PATH"

# Run the Python script with the appropriate arguments
srun python test.py --ckpt_path "$CKPT_PATH" --result_save_root "$RESULT_PATH"

#!/bin/bash
#SBATCH --job-name=thick_skip         
#SBATCH --partition=gpu             
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=4          
#SBATCH --gres=gpu:4                
#SBATCH --cpus-per-task=3            
#SBATCH --time=23:00:00              
#SBATCH --mem=64G                   
#SBATCH --mail-type=BEGIN,END      
#SBATCH --mail-user=scrmat@leeds.ac.uk  

# Tell OpenMP how much resource it has been given
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=12341
export WORLD_SIZE=4

# Get the first node name as master address
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Load miniforge for the conda environment
module load miniforge
source activate FSPNet

# The command to run the training script
srun python train.py --path "#add train data path here" --pretrain "your_path_to_vit_ckpt/base_patch16_384.pth"


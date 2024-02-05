#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
##SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --account=OD-221016
#set -e

#source /flush5/tay400/envs/mirorrnet/bin/activate
source /scratch1/tay400/envs/mirorrnet/bin/activate
#module load pytorch
module load pytorch/1.12.1-py39-cuda112-mpi
module load tensorflow/2.9.1-py39-cuda112
module load pytorch-lightning/1.8.6

# PYTHONPATH=src/mirorrnet/src/ \

export PYTHONUNBUFFERED=1

train_registration \
  --dataset "OASIS1" \
  --csv_file '/scratch1/tay400/src/data/OASIS-1/participants_modified.csv' \
  --ckpt-path '/scratch1/tay400/runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
  --registration-module-class NonrigidModule \
  --gpus 2 --num_workers ${SLURM_CPUS_PER_TASK}  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 5 --log_every_n_steps 5 --max_epochs 100 \
  --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 \
  --lambda-param-grad 1e-2 --lambda-depth-supervision 0.1

deactivate


  #--csv_file '/flush5/tay400/src/data/OASIS-1/participants_modified.csv' \

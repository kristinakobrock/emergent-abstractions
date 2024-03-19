#!/bin/bash
#SBATCH -J lin_3_4_standard
#SBATCH --time=02:30:00
#SBATCH --mem=400gb
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpu
#SBATCH --gpu-bind=single:1
source $HOME/.bashrc
spack load miniconda3@4.10.3
conda activate egg

srun python $HOME/emergent-abstractions/train.py --batch_size 32 --n_epochs 400 --dimensions 4 4 4 --learning_rate 0.0005 --game_size 10 --hidden_size 256 --temp_update 0.99 --temperature 2 --save True --num_of_runs 5 --path "$HOME/emergent-abstractions/" --mu_and_goodman True --speaker_hidden_size 128 --listener_hidden_size 128 --embedding_size 64 --zero_shot True --zero_shot_test 'generic' --load_dataset 'dim(4,8)_generic.ds'
#just make sure system got time to save all the models and stuff...
srun sleep 10

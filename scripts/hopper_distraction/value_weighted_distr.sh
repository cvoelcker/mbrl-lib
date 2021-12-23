#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 'rtx6000,t4v1,t4v2,p100'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=72:00:00      # time limit
#SBATCH --mem=16GB         # minimum amount of real memory
#SBATCH --job-name=vaml

source ~/.bashrc
conda activate ClaasICLR

export PYTHONPATH=/h/$USER/mbrl-lib-iclr
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/voelcker/.mujoco/mujoco200/bin:/usr/local/nvidia/lib64


cd /h/$USER/mbrl-lib-iclr

python3 -m mbrl.examples.main \
	seed=$1 \
	algorithm=mbpo \
	dynamics_model=value_weighted \
	overrides=mbpo_hopper_distraction \
	overrides.num_steps=500000 \
	overrides.distraction_dimensions=$2 \
	hydra.run.dir="$HOME/Claas/$SLURM_JOB_ID"

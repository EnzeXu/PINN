#!/bin/bash -l

#SBATCH --job-name="PP_1000000"
#SBATCH --partition=medium
#SBATCH --nodes=6
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL


#SBATCH --output="job/SimpleNetworkPP_1000000-%jo"
#SBATCH --error="job/SimpleNetworkPP_1000000-%je"

source /deac/csc/chenGrp/software/tensorflow/bin/activate

python /home/xue20/workspace/PINN/PINN_SimpleNetworkPP.py --epoch 1000000 --epoch_step 10000 --save_step 100000


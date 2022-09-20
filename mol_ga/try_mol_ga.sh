#!/bin/sh

#$ -cwd
#$ -l f_node=1
#$ -l h_rt=20:00:00

. /etc/profile.d/modules.sh

module load intel
module load cuda/10.1.105
module load cudnn/7.6

bash ./Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p ./miniconda
export PATH="./miniconda/bin:$PATH"
conda env create -f environment.yml python=3.8
source activate molgeneration
pip install git+https://github.com/DEAP/deap@master

start=`date +%s`

# train prediction model
python prediction_model.py --model_path "./model/fruity_stat.pkl"
# create fragment lib
python generate_frag_lib.py
# generate new molecules by ga
python mol_generator_ga.py

stop=`date +%s`
echo "All Done"
echo "Total Time = $[ stop - start ]s"
rm -rf miniconda

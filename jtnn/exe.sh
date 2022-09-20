d
#$ -l f_node=1
#$ -l h_rt=12:00:00

. /etc/profile.d/modules.sh

module load intel
module load cuda/8.0.61
module load cudnn/5.1

bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda
export PATH="./miniconda/bin:$PATH"
conda env create -f environment.yml python=3.5
source activate molcyclegan
conda update -n base -c defaults conda
pip install --upgrade numpy
conda install pytorch=0.4.1 -c pytorch

cd jtnn
python mol_tree.py < ../data/smiles_all.txt

cd ../molvae
python pretrain.py --train ../data/smiles_all.txt --vocab ../data/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --save_dir ../data/pre_model/
python vaetrain.py --train ../data/smiles_all.txt --vocab ../data/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0007 --beta 0.005 --model ../data/pre_model/model.iter-4 --save_dir ../data/vae_model/

cd ..
python gen_latent.py --data ./210726_data/smiles_all.txt --vocab ./210726_data/vocab.txt --hidden 450 --depth 3 --latent 56 --model ./210726_data/vae_model/model.iter-6

rm -rf miniconda

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Recap

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mol_path", default="./data/goodscents.csv")
parser.add_argument("--main_frag_smi_path", default="./data/lib/fruity_lib_main.smi")
parser.add_argument("--sub_frag_smi_path", default="./data/lib/fruity_lib_sub.smi")
parser.add_argument("--freq", default=1, type=int)

args = parser.parse_args()
mol_path = args.mol_path
main_frag_smi_path = args.main_frag_smi_path
sub_frag_smi_path = args.sub_frag_smi_path
freq = args.freq

data = pd.read_csv(mol_path)

mol = data['smiles'].map(lambda x: Chem.MolFromSmiles(x))
frag_r_mol = mol.map(lambda x: Recap.RecapDecompose(x) if x else 0)
# frag_c = frag_r_mol.map(lambda x: list(x.children.keys()) if x != 0 else 0)
frag_c = frag_r_mol.map(lambda x: list(x.GetLeaves().keys()) if x != 0 else 0)
frag_c = pd.DataFrame(frag_c)
frag_c.columns = ['frag']

frag_counter = {}

for _, row in frag_c.iterrows():
    tmp_r = row['frag']
    if tmp_r != 0:
        for i in tmp_r:
            if i in frag_counter:
                frag_counter[i] += 1
            else:
                frag_counter[i] = 1

# print('frag_counter =', len(frag_counter))


def count_occurrence(frag, freq=2):
    # print('total =', len(fgm))
    res = {}
    for key, value in frag.items():
        if value >= freq:
            res[key] = value
    return res


frag_twice = count_occurrence(frag_counter, freq)

frag = pd.DataFrame(list(frag_twice.keys()))
frag.columns = ['smiles']
# frag.to_csv(frag_csv_path, index=None)

# save fragments as smi file
frag.to_csv(main_frag_smi_path, index=None, header=None)

frag['count'] = frag['smiles'].map(lambda x: x.count('*'))
sub_frag = frag['smiles'].loc[frag['count'] == 1]

sub_frag.to_csv(sub_frag_smi_path, index=None, header=None)

print('success!')

import os

import argparse

import pandas as pd
import numpy as np
from sympy import content
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs

from models.net import Net



# predict results

# print(smiles)
# smiles = (smiles.loc[:49999, ['mols']])
# smiles.columns = ['smiles']
# print(smiles.shape)
def get_fp(smiles):
    mol = smiles.applymap(lambda x: Chem.MolFromSmiles(x))
    mol_clean = mol.applymap(lambda x: x if x else 0)
    mol_clean = mol_clean[~mol_clean['smiles'].isin([0])]
    mol_clean.index = [i for i in range(len(mol_clean))]
    mol_clean_smiles = mol_clean.applymap(lambda x: Chem.MolToSmiles(x))
    fp = mol_clean.applymap(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    # print(fp.shape)
    fp_bit_arr = []
    for i in fp.index:
        arr = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp.at[i, 'smiles'], arr)
        fp_bit_arr.append(arr)
    fp_bit_array = np.array(fp_bit_arr)

    X_fp_bit = pd.DataFrame(fp_bit_array)
    return mol_clean_smiles, X_fp_bit

def predict_res(X_fp_bit):
# pridict filled fragment
    mols_input = torch.FloatTensor(X_fp_bit.values)
    mols_input = mols_input.cuda()
    pred_mols = model(mols_input)
    pred_mols.sigmoid_()
    pred_mols = pred_mols.cpu()


    pred_mols_arr = pred_mols.detach().numpy()
    pred_mols_df = pd.DataFrame(pred_mols_arr)
    pred_mols_df = pred_mols_df.applymap(lambda x: np.around(x, 4))

    return pred_mols_df

def output_csv(smiles, pred_mols_df, proba_res):
    smiles.index = [i for i in range(len(smiles))]
    res_df = pd.concat([smiles, pred_mols_df], axis=1)
    #print(smiles)
    #print(pred_mols_df)
    #print(res_df.shape)
    res_df.columns = ['smiles', col]
    # ref_df = res_df
    ref_df = res_df.sort_values(by=col, ascending=False)
    ref_df.index = [i for i in range(len(smiles))]

    ref_df.to_csv(proba_res, index=None)
    print('ok!')

'''prior, later = 0, 50000
for i in range(4):
    smiles_p = smiles[prior + i * 50000: later + i * 50000]
    mol_clean_smiles, X_fp_bit = get_fp(smiles_p)
    pred_mols_df = predict_res(X_fp_bit)
    proba_res = 'zinc_p%d.csv' % i
    output_csv(smiles_p, pred_mols_df, proba_res)
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--col_name", default="proba")
    parser.add_argument("--model_path", default="pred_model/fruity.stat.pkl")
    parser.add_argument("--file_path", default="./data/fruity")

    args = parser.parse_args()

    col = args.col_name
    model_path = args.model_path
    file_path = args.file_path

    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.to('cuda:0')

    files = os.listdir(file_path)
    for file in files:
        file = os.path.join(file_path, file)
        if not os.path.isdir(file):
            smiles_file = file
            print(smiles_file)
            res_file = file + col + '.csv'
            # print(smiles_file)
            # print(res_file)
        
            # if the file format is .smi
            #content = Chem.SmilesMolSupplier(file)
            #smiles = [Chem.MolToSmiles(mol) for mol in content]
            #dic = {'smiles': smiles}
            #smiles = pd.DataFrame(dic).drop_duplicates().dropna()
            
            # if the file format is .csv
            smiles = pd.read_csv(smiles_file, usecols=['smiles']).drop_duplicates().dropna()

            mol_clean_smiles, X_fp_bit = get_fp(smiles)
            pred_mols_df = predict_res(X_fp_bit)
            output_csv(smiles, pred_mols_df, res_file)

    print('success!')

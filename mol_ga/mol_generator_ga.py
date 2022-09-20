import random
import time

import argparse

import numpy as np
import pandas as pd
import torch
from deap import base, creator, tools
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from method import generator
from method.net import Net


parser = argparse.ArgumentParser()
parser.add_argument("--main_frag_path", default="./data/lib/fruity_lib_main.smi")
parser.add_argument("--sub_frag_path", default="./data/lib/fruity_lib_sub.smi")
parser.add_argument("--res_path", default="./data/res/res_fruity.csv")
parser.add_argument("--eval_model_path", default="./model/fruity_stat.pkl")
parser.add_argument("--output_threshold", default=0.7, type=float)
parser.add_argument("--static_threshold", default=False, type=bool)
parser.add_argument("--penl1_model_path", default="./model/sweet_stat.pkl")
parser.add_argument("--penl2_model_path", default="./model/fatty_stat.pkl")
parser.add_argument("--penl3_model_path", default="./model/spicy_stat.pkl")

parser.add_argument("--number_of_iteration_of_ga", default=50, type=int)
parser.add_argument("--number_of_population", default=100, type=int)
parser.add_argument("--number_of_generation", default=100, type=int)
parser.add_argument("--probability_of_crossover", default=0.1, type=float)
parser.add_argument("--probability_of_mutation", default=0.6, type=float)

args = parser.parse_args()
main_frag_path = args.main_frag_path
sub_frag_path = args.sub_frag_path
res_path = args.res_path
eval_model_path = args.eval_model_path

output_threshold = args.output_threshold
static_threshold = args.static_threshold

# if the death penalty for other lables will be used
# go to fuction "feasible()" and cancel the comments
penl1_model_path = args.penl1_model_path
penl2_model_path = args.penl2_model_path
penl3_model_path = args.penl3_model_path

number_of_iteration_of_ga = args.number_of_iteration_of_ga
number_of_population = args.number_of_population
number_of_generation = args.number_of_generation
probability_of_crossover = args.probability_of_crossover
probability_of_mutation = args.probability_of_mutation

def model_fitting(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.to('cuda:0')
    model.eval()
    return model

eval_model = model_fitting(eval_model_path)

# if you need penalty on other odor labels
# cancel the following comments

# penl1_model = model_fitting(penl1_model_path)
# penl2_model = model_fitting(penl2_model_path)
# penl3_model = model_fitting(penl3_model_path)

start = time.time()

# generate molecules
main_molecules = [molecule for molecule in Chem.SmilesMolSupplier(main_frag_path,
                                                                  delimiter='\t', titleLine=False)
                  if molecule is not None]
fragment_molecules = [molecule for molecule in Chem.SmilesMolSupplier(sub_frag_path,
                                                                      delimiter='\t', titleLine=False)
                      if molecule is not None]

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
min_boundary = np.zeros(len(fragment_molecules) + 1)
max_boundary = np.ones(len(fragment_molecules) + 1) * 1.0


def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min, max in zip(min_boundary, max_boundary):
        index.append(random.uniform(min, max))
    return index


toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
toolbox.register('individual', tools.initIterate,
                 creator.Individual, toolbox.create_ind)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


# evaluate
def evalOneMax(individual):
    individual_array = np.array(individual)
    generated_smiles = generator.structure_generator_based_on_r_group(main_molecules, fragment_molecules,
                                                                      individual_array)
    generated_molecule = Chem.MolFromSmiles(generated_smiles)
    if generated_molecule is not None:
        gene_mole_fp = AllChem.GetMorganFingerprintAsBitVect(
            generated_molecule, 2)
        X = torch.FloatTensor(gene_mole_fp).cuda()
        X = X.unsqueeze(0)
        pred = eval_model(X).sigmoid_().cpu().detach().numpy()[0][0]
        value = pred

        # add static penalty (according to the MW)
        # if you want to use the death penalty, turn to the function feasible()
        if static_threshold:
            molwt = Descriptors.MolWt(generated_molecule)
            value = pred - (molwt - 300) / static_threshold
        
    else:
        value = 0.0

    return value,

def feasible(individual):
    individual_array = np.array(individual)
    generated_smiles = generator.structure_generator_based_on_r_group(main_molecules, fragment_molecules,
                                                                      individual_array)
    generated_molecule = Chem.MolFromSmiles(generated_smiles)

    if generated_molecule is not None:

        # add death penalty (according to the predicted value)
        # if the death penalty will be used, cancel these following comments

        '''
        gene_mole_fp = AllChem.GetMorganFingerprintAsBitVect(
            generated_molecule, 2)
        X = torch.FloatTensor(gene_mole_fp).cuda()
        X = X.unsqueeze(0)
        pred1 = penl1_model(X).sigmoid_().cpu().detach().numpy()[0][0]
        pred2 = penl2_model(X).sigmoid_().cpu().detach().numpy()[0][0]
        pred3 = penl3_model(X).sigmoid_().cpu().detach().numpy()[0][0]
        if pred1 >= 0.2 or pred2 >= 0.2 or pred3 >= 0.2:
            return False
        if pred1 >= 0.3:
            return False
        return True
        
        # these following codes are the death penalty of MW

        molwt = Descriptors.MolWt(generated_molecule)
        if molwt >= 300:
            return False
        return True
        '''
    return False


toolbox.register('evaluate', evalOneMax)

# if you would like to use the death penalty, cancel the following comment
# toolbox.decorate('evaluate', tools.DeltaPenalty(feasible, 0))

toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

generated_smiles_all = []
estimated_y_all = []
for iteration_number in range(number_of_iteration_of_ga):
    print(iteration_number + 1, '/', number_of_iteration_of_ga)
    # random.seed(random_seed + number_of_iteration_of_ga)
    random.seed()
    pop = toolbox.population(n=number_of_population)

    print('Start of evolution')

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print('  Evaluated %i individuals' % len(pop))

    for generation in range(number_of_generation):
        print('-- Generation {0} --'.format(generation + 1))

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_of_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < probability_of_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print('  Evaluated %i individuals' % len(invalid_ind))

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print('  Min %s' % min(fits))
        print('  Max %s' % max(fits))
        print('  Avg %s' % mean)
        print('  Std %s' % std)
        
    print('-- End of (successful) evolution --')

    for each_pop in pop:
        if each_pop.fitness.values[0] >= output_threshold:
            estimated_y_all.append(each_pop.fitness.values[0])
            each_pop_array = np.array(each_pop)
            # print(each_pop_array)
            smiles = generator.structure_generator_based_on_r_group(main_molecules, fragment_molecules,
                                                                    each_pop_array)
            generated_smiles_all.append(smiles)

end = time.time()

mols_df = pd.DataFrame(generated_smiles_all)
prob_df = pd.DataFrame(estimated_y_all)
res_df = pd.concat([mols_df, prob_df], axis=1)
res_df.columns = ['smiles', 'proba']
# print(len(res_df))
res_df = res_df.drop_duplicates(['smiles'])
# print(len(res_df))
if static_threshold:
    res_df['mol'] = res_df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    res_df['molwt'] = res_df['mol'].map(lambda x: Descriptors.MolWt(x))
    res_df['proba'] = res_df['proba'] + (res_df['molwt'] - 300) / static_threshold


ref_df = res_df.sort_values(by='proba', ascending=False)
ref_df.index = [i for i in range(len(res_df))]
id_df = pd.DataFrame([i for i in range(1, len(res_df) + 1)])
res_fin = pd.concat([id_df, ref_df['smiles'], ref_df['proba']], axis=1)
res_fin.columns = ['id', 'smiles', 'proba']
res_fin.drop(['id'], axis=1)
res_fin.to_csv(res_path, index=None)

print('running time =', end - start)

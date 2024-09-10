import glob
import itertools
import os
import random
from typing import Any
import numpy as np
import pandas as pd
from typing import Set
from sklearn.model_selection import KFold
import argparse

def load_data(mode="bimodal") -> pd.DataFrame:
    if mode == "code":
        path = '.code_only/processed_data/'
    else:
        path = './processed_data/grouped_by_code_snippets'

    all_files = glob.glob(os.path.join(path, '*.pkl'))

    df = pd.DataFrame()
    for filepath in all_files:
        print(filepath)
        df = pd.concat([df, pd.read_pickle(filepath)])

    df = df.reset_index(drop=True)
    df = df.fillna(0)

    return df
def load_data_stim(mode,simulation) -> pd.DataFrame:
    if mode == "code":
        path = '../processed_data/stim_code_imp'
    else:
        if simulation=='-7':
            path = '../processed_data/stim_data_-7'
        elif simulation=='random':
            path = f'../processed_data/stim_data_random11'
        elif simulation=='augment':
            path = f'../processed_data/stim_data_aug'
        elif simulation=='REA':
            path = f'../processed_data/stim_data_REA'
        elif simulation == 'zero':
            path = f'../processed_data/stim_data11_zero'
        elif simulation == 'randomall':
            path = f'../processed_data/stim_data_random_overall'
        elif simulation == '-9':
            path = f'../processed_data/stim_data_-9'
        elif simulation == 'real':
            path = '../processed_data/stim_real_data'
#        else:
#            path=f'../processed_data/real_data_plus{simulation[4:]}'
        elif simulation in [str(i) for i in range(0,11)]:
            path='../processed_data/stim_data_augnum'
        elif simulation in ['REA'+str(i) for i in range(0,11)]:
            path='../processed_data/stim_data_REA_num'
        elif int(simulation)>20:
            path=f'../processed_data/stim_data_newcov{simulation[:-1]}'
    all_files = glob.glob(os.path.join(path, '*.pkl'))
    print(all_files)
    if path=='../processed_data/stim_data_augnum':
        all_files=[file for file in all_files if file.split('_')[-1]==simulation+".pkl"]
    elif path=='../processed_data/stim_data_REA_num':
        all_files=[file for file in all_files if file.split('_')[-1]==simulation[3:]+".pkl"]
    elif path==f'../processed_data/stim_data_newcov{simulation[0]}':
        all_files=[file for file in all_files if file.split('_')[-1]==simulation+'.pkl']
    print(all_files)
    df = pd.DataFrame()
    for filepath in all_files:
        print(filepath)
        df = pd.concat([df, pd.read_pickle(filepath)])

    df = df.reset_index(drop=True)
    df = df.fillna(0)
    return df

# load data split by code snippet stratified by label
def train_test_split_by_code_snippets(
        code_snippet_ids: set,
        X: pd.DataFrame,
        problem_setting: int,
        fold_4: bool = True,
) -> list:
    if os.path.exists("train_test_splits_by_code_snippet.npy") and not fold_4:
        print('Loading existing split "train_test_splits_by_code_snippet.npy"'.center(79, '~'))
        train_test_splits = np.load('train_test_splits_by_code_snippet.npy', allow_pickle=True).tolist()
        return train_test_splits

    print('Splitting data by code snippet'.center(79, '~'))
    train_test_splits = []
    X['subjective_difficulty'] = (
            X['subjective_difficulty'] > X['subjective_difficulty'].median()
    ).astype(int)
    if fold_4:
        print('4 fold cv')
        cs_id_folds = [
            ['10-117-V1', '10-117-V2', '10-181-V2', '10-258-V1', '10-258-V2', '10-258-V3'],
            ['10-928-V1', '10-928-V2', '142-929-V1', '142-929-V2', '189-1871-V1', '189-1871-V2'],
            ['366-143-V1', '49-510-V1', '49-510-V2', '49-76-V3', 'A1117-2696', 'A1117-384'],
            ['A189-895', 'A34-6', 'A49-7086', 'A49-600', 'A84-600'],
        ]
        all_idxs = set(range(0, len(X)))
        for cs_id_list in cs_id_folds:
            cs_idxs = set(X[np.isin(X.iloc[:, 0], cs_id_list)].index.tolist())
            all_other_cs_idxs = all_idxs - cs_idxs
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        print(train_test_splits)
    else:
        cs_id_folds = [
            ['10-117-V1', '10-117-V2', '10-181-V2'],
            ['10-258-V1', '10-258-V2', '10-258-V3'],
            ['10-928-V1', '10-928-V2'],
            ['142-929-V1', '142-929-V2', '189-1871-V1', '189-1871-V2'],
            ['366-143-V1', '49-510-V1', '49-510-V2'],
            ['49-76-V3', 'A1117-2696', 'A1117-384'],
            ['A189-895', 'A34-6', 'A49-7086'],
            ['A49-600', 'A84-600'],
        ]
        all_idxs = set(range(0, len(X)))
        for cs_id_list in cs_id_folds:
            cs_idxs = set(X[np.isin(X.iloc[:, 0], cs_id_list)].index.tolist())
            all_other_cs_idxs = all_idxs - cs_idxs
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        np.save('train_test_splits_by_code_snippet.npy', train_test_splits)
        print(train_test_splits)

    return train_test_splits

def train_test_split_by_stim(
        test_stim,
        code_snippet_ids: set,
        X: pd.DataFrame,
        problem_setting: int,
        fold_6: bool = True,
) -> list:
    if os.path.exists("train_test_splits_by_code_stim.npy"):
        print('Loading existing split "train_test_splits_stim.npy"'.center(79, '~'))
        train_test_splits = np.load('train_test_splits_by_stim.npy', allow_pickle=True).tolist()
#        cs_id_folds = [
#            ['stim1','stim15'],
#            ['stim2','stim1'],
#            ['stim19','stim22'],
#        ]
        return train_test_splits

    print('Splitting data by code snippet'.center(79, '~'))
    train_test_splits = []

#    X['difficulty'] = (
#            X['difficulty'] >=3
#    ).astype(int)
    if fold_6:
        print('3 fold cv')

        cs_id_folds=test_stim
        all_idxs = set(range(0, len(X)))
        for cs_id_list in cs_id_folds:
            cs_idxs = set(X[np.isin(X.iloc[:, 0], cs_id_list)].index.tolist())
            print(cs_id_list)
            all_other_cs_idxs = all_idxs - cs_idxs
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        print(train_test_splits)
    else:
        cs_id_folds = [
            ['10-117-V1', '10-117-V2', '10-181-V2'],
            ['10-258-V1', '10-258-V2', '10-258-V3'],
            ['10-928-V1', '10-928-V2'],
            ['142-929-V1', '142-929-V2', '189-1871-V1', '189-1871-V2'],
            ['366-143-V1', '49-510-V1', '49-510-V2'],
            ['49-76-V3', 'A1117-2696', 'A1117-384'],
            ['A189-895', 'A34-6', 'A49-7086'],
            ['A49-600', 'A84-600'],
        ]
        cs_id_folds = [
            ['stim1'],
            ['stim2'],
            ['stim15'],
            ['stim16'],
            ['stim19'],
            ['stim22'],
        ]
        all_idxs = set(range(0, len(X)))
        for cs_id_list in cs_id_folds:
            cs_idxs = set(X[np.isin(X.iloc[:, 0], cs_id_list)].index.tolist())
            all_other_cs_idxs = all_idxs - cs_idxs
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        np.save('train_test_splits_by_code_snippet.npy', train_test_splits)
        print(train_test_splits)

    return train_test_splits


# load data split by participant stratified by label
def train_test_split_by_participants(
        participant_ids: set,
        X: pd.DataFrame,
        problem_setting: str,
        fold_4: bool = True,
) -> list:
    if problem_setting == "subjective_difficulty_score":
        problem_setting = "subjective_difficulty"

    if os.path.exists("train_test_splits_by_participant.npy") and not fold_4:
        print('Loading existing split "train_test_splits_by_participant.npy"'.center(79, '~'))
        train_test_splits = np.load('train_test_splits_by_participant.npy', allow_pickle=True).tolist()
        print(train_test_splits)
        return train_test_splits

    print('Splitting data'.center(79, '~'))
    run = True
    seen: set = set()
    train_test_splits = []
    X['subjective_difficulty'] = (
            X['subjective_difficulty'] > X['subjective_difficulty'].mean()
    ).astype(int)
    participant_ids = list(participant_ids)
    if fold_4:
        kfold = KFold(n_splits=4, shuffle=True, random_state=42)
        for train_ids, test_ids in kfold.split(participant_ids):

            all_other_cs_idxs = set(X[np.isin(X.iloc[:, 1], np.array(participant_ids)[train_ids])].index.tolist())
            cs_idxs = set(X[np.isin(X.iloc[:, 1], np.array(participant_ids)[test_ids])].index.tolist())
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        print(train_test_splits)
    else:
        while run:
            cs_id_list = []
            add_ids = False
            add_ids = False
            for cs_id in participant_ids:
                if len(seen) == len(participant_ids):
                    run = False
                    break

                if cs_id in seen:
                    continue
                all_idxs = set(range(0, len(X)))
                cs_id_list.append(cs_id)

                if len(X[np.isin(X.iloc[:, 1], cs_id_list)][problem_setting].unique()) == 2:
                    cs_idxs = set(X[np.isin(X.iloc[:, 1], cs_id_list)].index.tolist())
                    all_other_cs_idxs = all_idxs - cs_idxs
                    seen.update(cs_id_list)
                    add_ids = True
                    break
                elif len(X[np.isin(X.iloc[:, 1], cs_id_list)][problem_setting].unique()) == 1:
                    continue
                else:
                    raise ValueError('got no labels')
            if add_ids:
                train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        np.save('train_test_splits_by_participant.npy', train_test_splits)
        print(train_test_splits)

    return train_test_splits

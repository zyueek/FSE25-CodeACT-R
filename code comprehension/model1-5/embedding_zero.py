import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from unixcoder import UniXcoder
import pandas as pd
import random
import ast
from tqdm import *
import pickle
import warnings
warnings.filterwarnings("ignore")
def encode_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # You might want to take the mean of the last hidden states
    return outputs.last_hidden_state.mean(dim=1).cpu()
def get_line_from_file(lines, line_number):
    try:
        if 0 < line_number <= len(lines):
            return lines[line_number - 1].strip()
        else:
            return f"Error: Line number {line_number} is out of range."
    except FileNotFoundError:
        print("Error: File not found.")
#!pip install tokenizers==0.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine").to(device)
def compute_input_matrix_row(stim,difficulty):
    df_fixation=pd.read_csv('../cross_valid_result_number/cross_valid_result_20/'+stim+'_'+stim[4:]+"_aug0.csv")
    code_txt='../code/'+stim+".txt"
    EMBED_VECTOR_SIZE = 768
    max_sequence_length = 128
    max_sequence_with_embed_len = (EMBED_VECTOR_SIZE + 1) * max_sequence_length
#    IA_embedding = np.zeros(EMBED_VECTOR_SIZE)
    num_columns = max_sequence_with_embed_len+3
    first_three_columns = ['stim', 'participant', 'difficulty']
    additional_columns = [i for i in range(num_columns - len(first_three_columns))]
    print(stim)
    all_columns = first_three_columns + additional_columns
#    print(len(all_columns))
    df = pd.DataFrame(columns=all_columns)
    with open(code_txt, 'r') as file:
        lines = file.readlines()
#    input_matrix_row = []
    for i, row in df_fixation.iterrows():
        input_matrix_row = []
        scanpath=row['minsimu']
        scanpath = ast.literal_eval(scanpath)
        scanpath = scanpath[0][1]
        code_snippet_id = stim
        subjective_difficulty = difficulty
        for fixation in tqdm(scanpath):
            code=get_line_from_file(lines,fixation)
#            print(code)
            code_embedding = encode_code(code).detach().numpy()
#            print(type(code_embedding))
            participant_id = i
#            print(code_embedding.tolist()[0])
#            input_matrix_row += ([float(fixation)] + code_embedding.tolist()[0])
#            padded_input_matrix_row = np.pad(input_matrix_row, (0, max_sequence_with_embed_len - len(input_matrix_row)), mode='constant').tolist()
            new_elements = [float(fixation)] + code_embedding.tolist()[0]
            if len(input_matrix_row) + len(new_elements) <= max_sequence_with_embed_len:
                input_matrix_row += new_elements
            else:
                remaining_space = max_sequence_with_embed_len - len(input_matrix_row)
                input_matrix_row += new_elements[:remaining_space]
            if len(input_matrix_row) < max_sequence_with_embed_len:
                padded_input_matrix_row = np.pad(input_matrix_row, (0, max_sequence_with_embed_len - len(input_matrix_row)), mode='constant').tolist()
            else:
                padded_input_matrix_row = input_matrix_row[:max_sequence_with_embed_len]
        input_matrix_row=[code_snippet_id, participant_id, subjective_difficulty] + padded_input_matrix_row
        df.loc[len(df)] = input_matrix_row
    with open('../processed_data/stim_data11_zero/'+stim+'.pkl', 'wb') as file:
        pickle.dump(df, file)
#compute_input_matrix_row('stim1',5)
stim_list={1:4,2:5,13:4,17:2,21:1,24:5,15:3,16:1,19:2,22:2,5:5}
#stim_list={13:4,17:2,21:1,24:5,5:5}
for stim,dif in stim_list.items():
    print(stim,dif)
    compute_input_matrix_row('stim'+str(stim),dif)
#        print(input_matrix_row)

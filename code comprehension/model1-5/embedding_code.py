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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine").to(device)
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
def compute_input_matrix_row(stim,difficulty,df):
    code_txt='../code/'+stim+".txt"
    max_sequence_length = 128
    EMBED_VECTOR_SIZE=1
    max_sequence_with_embed_len = 128
#    IA_embedding = np.zeros(EMBED_VECTOR_SIZE)
    with open(code_txt, 'r') as file:
    # Read the content of the file and save it to the variable 'code'
        all_code = file.readlines()
    i=0
#    input_matrix_row = []
    for code in all_code:
        input_matrix_row = []
        code_snippet_id = stim
        subjective_difficulty = difficulty
        participant_id = i
        i+=1
#        code_embedding = [i for i in range(len(code_token))]
        code_embedding=encode_code(code).detach().numpy()
        input_matrix_row = code_embedding.tolist()[0]
        if(len(input_matrix_row)>max_sequence_with_embed_len):
            input_matrix_row=input_matrix_row[:max_sequence_with_embed_len]
        padded_input_matrix_row = np.pad(input_matrix_row, (0, max_sequence_with_embed_len - len(input_matrix_row)), mode='constant').tolist()
#        print(padded_input_matrix_row).
        input_matrix_row=[code_snippet_id, participant_id, subjective_difficulty] + padded_input_matrix_row

        df.loc[len(df)] = input_matrix_row
num_columns = 131
first_three_columns = ['stim', 'participant', 'difficulty']
additional_columns = [i for i in range(128)]
all_columns = first_three_columns + additional_columns
df = pd.DataFrame(columns=all_columns)
stim_list={1:4,2:5,13:4,17:2,21:1,24:5,15:3,16:1,19:2,22:2,5:5}
for stim,dif in stim_list.items():
    compute_input_matrix_row('stim'+str(stim),dif,df)
#rint(df)
with open('../processed_data/stim_code_imp/'+'code'+'.pkl', 'wb') as file:
    pickle.dump(df, file)

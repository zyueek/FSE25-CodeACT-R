import argparse
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from unixcoder import UniXcoder
import pandas as pd
import random
import ast
from tqdm import tqdm
import pickle
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
parser = argparse.ArgumentParser()
parser.add_argument('--number')
parser.add_argument('--aug')
parse = parser.parse_args()
# Set the start method to 'spawn' at the beginning of your script
set_start_method('spawn', force=True)
def random_erasing(data, s=(0.02, 0.1), r=(0.3, 3.3), value=0):
    """
    Apply Random Erasing to a single row of data.
    :param data: numpy array of a single row
    :param s: range of proportion of erased area over input area
    :param r: aspect ratio of erased area
    :param value: erasing value
    :return: erased numpy array
    """
    area = data.shape[0]
    target_area = np.random.uniform(s[0], s[1]) * area
    aspect_ratio = np.random.uniform(r[0], r[1])

    h = int(round(np.sqrt(target_area * aspect_ratio)))
    w = int(round(np.sqrt(target_area / aspect_ratio)))

    if w < area:
        x1 = np.random.randint(0, area - w)
        data[x1:x1+w] = value

    return data

# Function to encode a batch of code snippets
def encode_code_batch(code_snippets, model, tokenizer, device):
    inputs = tokenizer(code_snippets, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# Function to get a specific line from a file
def get_line_from_file(lines, line_number):
    try:
        if 0 < line_number <= len(lines):
            return lines[line_number - 1].strip()
        else:
            return f"Error: Line number {line_number} is out of range."
    except FileNotFoundError:
        print("Error: File not found.")


# Main function to process a single row
def process_row(args):
    row, lines, stim, difficulty, model_name, max_sequence_with_embed_len = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_matrix_row = []
    scanpath = ast.literal_eval(row['scanpath'])
    code_snippet_id = stim
    subjective_difficulty = difficulty

    # Collect all code snippets for this scanpath
    code_snippets = [get_line_from_file(lines, fixation) for fixation in scanpath]

    # Encode all code snippets in a batch
    code_embeddings = encode_code_batch(code_snippets, model, tokenizer, device)

    for j, fixation in enumerate(scanpath):
        code_embedding = code_embeddings[j]
        new_elements = [float(fixation)] + code_embedding.tolist()
        if len(input_matrix_row) + len(new_elements) <= max_sequence_with_embed_len:
            input_matrix_row += new_elements
        else:
            remaining_space = max_sequence_with_embed_len - len(input_matrix_row)
            input_matrix_row += new_elements[:remaining_space]

    if len(input_matrix_row) < max_sequence_with_embed_len:
        padded_input_matrix_row = np.pad(input_matrix_row, (0, max_sequence_with_embed_len - len(input_matrix_row)),
                                         mode='constant').tolist()
    else:
        padded_input_matrix_row = input_matrix_row[:max_sequence_with_embed_len]

    return [code_snippet_id, row.name, subjective_difficulty] + padded_input_matrix_row


# Function to compute the input matrix row for a given stimulus and difficulty
def get_original_embedding(stim, difficulty):
    # Read the fixation data
    df_fixation=pd.read_csv('../real_scanpath/'+str(stim[4:])+'.csv')
    code_txt = f'../code/{stim}.txt'
    EMBED_VECTOR_SIZE = 768
    max_sequence_length = 128
    with open(code_txt, 'r') as file:
        lines = file.readlines()
    print(len(df_fixation),parse.aug)
    aug_number=int(len(df_fixation)*int(parse.aug)//10)
    print(aug_number)
    random_rows = df_fixation.sample(n=aug_number)
#    print(random_rows)
    df_original = pd.DataFrame(random_rows)

    max_sequence_with_embed_len = (EMBED_VECTOR_SIZE + 1) * max_sequence_length
    num_columns = max_sequence_with_embed_len + 3
    first_three_columns = ['stim', 'participant', 'difficulty']
    additional_columns = [i for i in range(num_columns - len(first_three_columns))]
    all_columns = first_three_columns + additional_columns

    model_name = "microsoft/unixcoder-base-nine"

    args = [(row, lines, stim, difficulty, model_name, max_sequence_with_embed_len) for _, row in
            df_original.iterrows()]

    with Pool(processes=min(cpu_count(), len(args))) as pool:
        results = list(tqdm(pool.imap(process_row, args), total=len(args)))

    df = pd.DataFrame(results, columns=all_columns)
    return df,aug_number



if __name__ == '__main__':
    # Set up the device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of stimuli with their corresponding difficulty levels
    stim_list = {1: 4, 2: 5, 13: 4, 17: 2, 21: 1, 24: 5, 15: 3, 16: 1, 19: 2, 22: 2, 5: 5}
    for stim, dif in stim_list.items():
        df,aug_number=get_original_embedding('stim'+str(stim),dif)
        print(aug_number)
        augmented_data = []
        total_number=int(parse.number)
        
        simu_num=total_number//(aug_number)
        remain=total_number-simu_num*aug_number
        for _ in range(simu_num):
            for index, row in df.iterrows():
                augmented_row = random_erasing(row.values, value=np.nan)
                augmented_data.append(augmented_row)
        if remain>0:
            for index, row in df.head(remain).iterrows():
                augmented_row = random_erasing(row.values, value=np.nan)
                augmented_data.append(augmented_row)

        augmented_df = pd.DataFrame(augmented_data, columns=df.columns)
    # Process each stimulus
        with open('../processed_data/stim_data_REA_num/'+'stim'+str(stim)+"_"+parse.aug+'.pkl', 'wb') as file:
            pickle.dump(augmented_df, file)


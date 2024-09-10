import argparse
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from unixcoder import UniXcoder
import pandas as pd
import random
import ast
from tqdm import *
import argparse
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--number')
parser.add_argument('--aug')
parse = parser.parse_args()
from multiprocessing import Pool, cpu_count, set_start_method

# Set the start method to 'spawn' at the beginning of your script
set_start_method('spawn', force=True)


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
    scanpath = ast.literal_eval(row['minsimu'])[0][1]
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
def compute_input_matrix_row(stim, difficulty):
    # Read the fixation data
    df_fixation = pd.read_csv(f'../cross_valid_result_number/cross_valid_result_300/{stim}_{stim[4:]}_aug{parse.aug}.csv',nrows=int(parse.number))
    code_txt = f'../code/{stim}.txt'
    EMBED_VECTOR_SIZE = 768
    max_sequence_length = 128

    with open(code_txt, 'r') as file:
        lines = file.readlines()

    max_sequence_with_embed_len = (EMBED_VECTOR_SIZE + 1) * max_sequence_length
    num_columns = max_sequence_with_embed_len + 3
    first_three_columns = ['stim', 'participant', 'difficulty']
    additional_columns = [i for i in range(num_columns - len(first_three_columns))]
    all_columns = first_three_columns + additional_columns

    model_name = "microsoft/unixcoder-base-nine"

    args = [(row, lines, stim, difficulty, model_name, max_sequence_with_embed_len) for _, row in
            df_fixation.iterrows()]

    with Pool(processes=min(cpu_count(), len(args))) as pool:
        results = list(tqdm(pool.imap(process_row, args), total=len(args)))

    df = pd.DataFrame(results, columns=all_columns)

    with open(f'../processed_data/stim_data_augnum/{stim}_{parse.aug}.pkl', 'wb') as file:
        pickle.dump(df, file)


if __name__ == '__main__':
    # Set up the device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of stimuli with their corresponding difficulty levels
    stim_list = {1: 4, 2: 5, 13: 4, 17: 2, 21: 1, 24: 5, 15: 3, 16: 1, 19: 2, 22: 2, 5: 5}

    # Process each stimulus
    for stim, dif in stim_list.items():
        print(stim, dif)
        compute_input_matrix_row(f'stim{stim}', dif)


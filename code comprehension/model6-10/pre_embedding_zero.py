import os
import torch
import numpy
import numpy as np
import pandas as pd
import ast
import pickle
from transformers import AutoModel, AutoTokenizer
from unixcoder import UniXcoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine").to(device)

def encode_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # You might want to take the mean of the last hidden states
    return outputs.last_hidden_state.mean(dim=1).cpu()
import numpy as np
import pandas as pd
import ast
import pickle
from tqdm import *
def get_line_from_file(lines, line_number):
    try:
            # Line numbers start from 1, so we need to subtract 1 to get the correct index
        if 0 < line_number <= len(lines):
            return lines[line_number - 1].strip()
        else:
            return f"Error: Line number {line_number} is out of range."
    except FileNotFoundError:
        return f"Error: File  not found."


def create_pre_embeding(stim, PRE_COMPUTED_EMBEDDINGS):
    df_fixation=pd.read_csv('../cross_valid_result_number/cross_valid_result_20/'+stim+'_'+stim[4:]+"_aug0.csv")
    code_txt = '../code/' + stim + ".txt"
    tmp_embed_single_snippet = []
    max_length = 64
    with open(code_txt, 'r') as file:
        lines = file.readlines()
    print(stim)
    for i, row in df_fixation.iterrows():
        scanpath = row['minsimu']
        scanpath = ast.literal_eval(scanpath)
        scanpath = scanpath[0][1]
        if len(scanpath) > max_length:
            scanpath = scanpath[:max_length]
        for fixation in tqdm(scanpath):
            code = get_line_from_file(lines, fixation)
            code_embedding = encode_code(code).squeeze().tolist()
            code_embedding = np.array(code_embedding, dtype=np.float16).tolist()
            #            print(type(code_embedding[0]))
            zeros_list_float16 = np.zeros(768, dtype=np.float16).tolist()

            tmp_embed_single_snippet.append((code_embedding, zeros_list_float16))
    #            tmp_embed_single_snippet.append((code_embedding,[0]*768))
    #    print(PRE_COMPUTED_EMBEDDINGS)
    PRE_COMPUTED_EMBEDDINGS.append(tmp_embed_single_snippet)
PRE_COMPUTED_EMBEDDINGS=[]
stim_list=[1,2,15,16,19,22,13,17,21,24,5]
for stim in stim_list:
    create_pre_embeding('stim'+str(stim),PRE_COMPUTED_EMBEDDINGS)
#    print(PRE_COMPUTED_EMBEDDINGS)
# Determine the maximum length of the sublists
max_length = max(len(sublist) for sublist in PRE_COMPUTED_EMBEDDINGS)

# Pad each sublist with zeros to make them of consistent length
for sublist in PRE_COMPUTED_EMBEDDINGS:
    zero_embed = [0] * 768
    sublist.extend([[zero_embed,zero_embed]] * (max_length - len(sublist)))
#print(PRE_COMPUTED_EMBEDDINGS)
np.array(PRE_COMPUTED_EMBEDDINGS).shape
PRE_COMPUTED_EMBEDDINGS = np.array(PRE_COMPUTED_EMBEDDINGS)
np.save("../processed_data/PRE_COMPUTED_EMBEDDINGS_STIM_LINE_11_ZERO.npy", PRE_COMPUTED_EMBEDDINGS)

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
def create_pre_embeding(stim,PRE_COMPUTED_EMBEDDINGS):
    code_txt='../code/'+stim+".txt"
    tmp_embed_single_snippet = []
    print(stim)
    with open(code_txt, 'r') as file:
    # Read the content of the file and save it to the variable 'code'
        code_all = file.readlines()
    for code in code_all:
        code_embedding=encode_code(code)[0].detach().numpy()
        tmp_embed_single_snippet.append((code_embedding,[0]*768))
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
np.save("../processed_data/PRE_COMPUTED_EMBEDDINGS_CODE.npy", PRE_COMPUTED_EMBEDDINGS)

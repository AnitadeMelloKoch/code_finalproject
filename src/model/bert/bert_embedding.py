import os 
import glob
import torch
import numpy as np
from tqdm import tqdm

from transformers import BertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

files = glob.glob('data/flowers/captions/*')

embeddings = np.zeros(shape=(len(files), 10, 768))


for i, file_name in enumerate(tqdm(files)):
    file = open(file_name, 'r')
    lines = file.readlines()
    for j, line in enumerate(lines):
        input_sentence = torch.tensor(tokenizer.encode(line.strip())).unsqueeze(0)
        out = model(input_sentence)
        embeddings_of_last_layer = out[0]
        cls_embeddings = embeddings_of_last_layer[0].clone().detach().requires_grad_(False)
        embeddings[i, j] = np.mean(np.array(cls_embeddings),axis=0)
    file.close()

np.save('data/flowers/caption_embeds.npy', embeddings)
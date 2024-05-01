from datasets import load_dataset
from torch import nn
from tqdm import tqdm
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim import AdamW, lr_scheduler
from transformers import RobertaTokenizer, RobertaForMaskedLM
from util import config
def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
import os
import argparse
from transformers import RobertaTokenizer, RobertaForMaskedLM



def load(inp,lab,task,bs=25):
    if task == 'operational':
        batch_number = 400
    else:
        batch_number = 280
    batch = []
    for i in range(batch_number):
      my_dict = {}
      my_dict['inputs'] = {key: value[i*25:(i+1)*25] for key, value in inp.items()}
      my_dict['labels'] = lab[i*25:(i+1)*25]
      batch.append(my_dict)
    return batch


# Instantiate the parser
parser = argparse.ArgumentParser()
# Required positional argument
parser.add_argument('--task', type=str,
                    help='can either be relational or operational')
args = parser.parse_args()

# task = 'relational' # < > =
task = args.task # + -
layers, _, size_dataset, hidden_size = config(task)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    # Load the tensor from the file
    dataset = torch.load(os.path.join(f"./{task}", "tokenized.pt"))
    input_ids = dataset['inputs'].to(device)
    labels = dataset['labels'].to(device)
    print("Tensor loaded, continue on to the activation step")
except FileNotFoundError:
    print("The file was not found, please get the tokenized dataset first.")
    exit 

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
layer_activations = {
    l: torch.zeros(size_dataset, hidden_size, # 1024
                  dtype=torch.float16)
    for l in layers
}
print(timestamp(),'start getting activations')
data = load(input_ids,labels,args.task)
offset = 0
for idx, batch in enumerate(tqdm(data, disable = False)):
    inputs_ = batch['inputs']
    labels_ = batch['labels']
    with torch.no_grad():
        output = model(**inputs_, output_hidden_states=True) #output_hidden_states 25 x 512 x 768
    masked_indices = torch.where(inputs_['input_ids'] == tokenizer.mask_token_id)
    for lix, activation in enumerate(output.hidden_states):
        processed_activations = activation[masked_indices]
        # print(processed_activations.shape)
        layer_activations[lix][offset:offset + 25] = processed_activations
    offset += 25

save_path = f'./{task}/activations'
for layer_ix, activations in layer_activations.items():
    save_name = f'{layer_ix}.pt'
    # print(activations.shape)
    os.makedirs(save_path, exist_ok=True)
    torch.save(activations, os.path.join(save_path, save_name))

print(timestamp(),'finish getting activations')
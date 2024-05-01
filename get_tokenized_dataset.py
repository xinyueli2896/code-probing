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

# Instantiate the parser
parser = argparse.ArgumentParser()
# Required positional argument
parser.add_argument('--task', type=str,
                    help='can either be relational or operational')
args = parser.parse_args()


# task = 'relational' # < > =
task = args.task # + -
layers , _, size_dataset, hidden_size = config(task)



tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
print(timestamp(),'start loading dataset')
dataset = load_dataset("code_search_net", "python")

##load test dataset(manageable size) and filter 
dataset_test = dataset['test']
CODE = []
labels = []
if task == 'relational':
    dataset_test = [entry for entry in dataset_test if '=' in entry['func_code_string'] or '>' in entry['func_code_string'] or '<' in entry['func_code_string']]
    num0 = 0
    num1 = 0
    num2 = 0
    for entry in dataset_test:
        code = entry['func_code_string']
        if '>' in code and '<' in code:
            entry['func_code_string'] = entry['func_code_string'].replace("<", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(0)
            num0 += 1
        elif '>' in code:
            entry['func_code_string'] = entry['func_code_string'].replace(">", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(1)
            num1 += 1
        elif '<' in code:
            entry['func_code_string'] = entry['func_code_string'].replace("<", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(0)
            num0 += 1
        elif '=' in code:
            if num2 > 2500:
                continue
            entry['func_code_string'] = entry['func_code_string'].replace("=", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(2)
            num2 += 1
elif task == 'operational':
    dataset_test = dataset['test']
    dataset_test = [entry for entry in dataset_test if '+' in entry['func_code_string'] or '-' in entry['func_code_string']]
    num_plus = 0
    num_minus = 0

    for entry in dataset_test:
        code = entry['func_code_string']
        if ('+' in code and '-' in code):
            entry['func_code_string'] = entry['func_code_string'].replace("+", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(1)
            num_plus += 1
        elif '+' in code:
            entry['func_code_string'] = entry['func_code_string'].replace("+", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(1)
            num_plus += 1
        elif '-' in code:
            entry['func_code_string'] = entry['func_code_string'].replace("-", "<mask>", 1)
            CODE.append(entry['func_code_string'])
            labels.append(0)
            num_minus += 1
print(timestamp(),'finish loading dataset')
print(timestamp(),'start tokenizing dataset')
input_ids = tokenizer(CODE, max_length=512,truncation=True,padding=True,return_tensors="pt")

# making sure we filter out the one with the masked token truncated in the previous step 
labels_new = []
for i in range(len(input_ids['input_ids']) - 1, -1, -1):
  # print(input_id.shape)
  if tokenizer.mask_token_id not in input_ids['input_ids'][i]:
    input_ids['input_ids'] = torch.cat((input_ids['input_ids'][:i], input_ids['input_ids'][i+1:]), dim=0)
    input_ids['attention_mask'] = torch.cat((input_ids['attention_mask'][:i], input_ids['attention_mask'][i+1:]), dim=0)
  else:
    labels_new.append(labels[i])
labels_new = labels_new[::-1]
input_ids['input_ids'] = input_ids['input_ids'][:size_dataset]
input_ids['attention_mask'] = input_ids['attention_mask'][:size_dataset]
labels_new = labels_new[:size_dataset]
labels = labels_new
labels = torch.tensor(labels).type(torch.LongTensor)

# sanity check 
print('tokenized dataset information: ')
print('input_ids shape', input_ids['input_ids'].shape)
print('labels shape', labels.shape)

# save the dataset
dataset = {'inputs':input_ids, 'labels':labels}
tokenized_save_path = f"./{task}/"
os.makedirs(tokenized_save_path, exist_ok=True)
torch.save(dataset, os.path.join(tokenized_save_path, "tokenized.pt"))
print('successfully save the tokenized dataset')
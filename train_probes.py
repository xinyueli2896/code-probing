# from datasets import load_dataset
from torch import nn,optim
from tqdm import tqdm
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import datetime
from torch.utils.tensorboard import SummaryWriter
# from torch.optim import AdamW, lr_scheduler
# from transformers import RobertaTokenizer, RobertaForMaskedLM
from util import *
import os
import argparse
# from transformers import RobertaTokenizer, RobertaForMaskedLM
from model.probe import LinearProber, NonLinearProber


    
def train(args):
# task = 'relational' # < > =
    task = args.task # + -
    _,n_layers, _, _ = config(task)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        # Load the tensor from the file
        dataset = torch.load(os.path.join(f"./{task}", "tokenized.pt"))
        labels = dataset['labels'].to(device)
        print("Tensor loaded, continue on to the training step")
    except FileNotFoundError:
        print("The file was not found, please get the tokenized dataset first. Terminate now.")
        return 

    try:
        # Load the tensor from the file
        dataset = torch.load(os.path.join(f"./{task}/activations", f'{0}.pt'))
        print("Tensor loaded, continue on to the training step")
    except FileNotFoundError:
        print("The file was not found, please get the activations first. Terminate now.")
        return 

    bs = 50
    probe_type = args.probe_type
    num_epochs = args.num_epochs
    cn = 2 if task == 'operational' else 3
    # for agg in activation_aggs:
    print(timestamp(),
        f'running {probe_type} probe on CodeBERT for {task} task')


    for layer in tqdm(range(n_layers)):
        # init probe
        if probe_type == 'Linear':
            probe =  LinearProber(class_num = cn).to(device)
        elif probe_type == 'NonLinear':
            probe = NonLinearProber(class_num = cn).to(device)

        # load data
        activations = load_activation_probing_dataset(layer, args.task).dequantize()
        print(activations.shape)
        dataset = TensorDataset(activations, labels)
        l = len(dataset)
        split = (5500, 750, 750) if task == 'relational' else (8000, 1000, 1000)
        train_dataset, val_dataset, test_dataset = random_split(dataset, split)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


        optimizer = optim.Adam(probe.parameters(), lr=args.lr)

        writer = SummaryWriter(f'./{args.tag}_ckpts/{task}/{probe_type}/lr_0.001_epoch_{num_epochs}/{layer}/runs/my_experiment')

        best_dev_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
        # train
            probe.train()
            running_loss = 0
            train_cnt = 0
            train_acc = 0

            for feature,label in train_loader:

                feature = feature.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                pred = probe(feature)
                loss = nn.CrossEntropyLoss()(pred, label)

                n_hits = torch.sum(pred.argmax(dim=1) == label).item()
                train_acc += n_hits

                loss.backward()
                running_loss += loss.item()
                train_cnt += label.shape[0]
                optimizer.step()


            running_loss = running_loss / train_cnt
            train_acc = train_acc / train_cnt
            if epoch % 10 == 0:
                writer.add_scalar('Loss/train', running_loss, epoch)
                writer.add_scalar('Acc/train', train_acc, epoch)

            # scheduler.step()

        # dev
            probe.eval()

            dev_loss = 0
            dev_acc = 0
            dev_cnt = 0
            with torch.no_grad():
                for feature, label in val_loader:
                    feature = feature.to(device)
                    label = label.to(device)
                    pred = probe(feature)

                    loss = nn.CrossEntropyLoss()(pred, label)
                    dev_loss += loss.item()

                    n_hits = torch.sum(pred.argmax(dim=1) == label).item()
                    dev_acc += n_hits
                    dev_cnt += label.shape[0]

                dev_acc = dev_acc / dev_cnt
                dev_loss = dev_loss / dev_cnt

                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_epoch = epoch
                    best_model = probe.state_dict()
                    best_optimizer = optimizer.state_dict()
                    best_loss = running_loss

                if epoch % 10 == 0:
                    writer.add_scalar('Loss/dev', dev_loss, epoch)
                    writer.add_scalar('Acc/dev', dev_acc, epoch)
                    print(f'Epoch [{epoch+1}/{num_epochs}], Dev Loss: {dev_loss}, training acc: {train_acc}')
        print(f"Best Validation Accuracy for layer {layer} was achieved in Epoch {best_epoch+1}: {best_dev_acc}")
        os.makedirs(f'./{args.tag}_ckpts/{task}/{probe_type}/lr_0.001_epoch_{num_epochs}/', exist_ok=True)
        torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': best_model,
                        'optimizer_state_dict': best_optimizer,
                        'loss': best_loss,
                        'Acc/dev': best_dev_acc,
                    }, f'./{args.tag}_ckpts/{task}/{probe_type}/lr_0.001_epoch_{num_epochs}/{layer}.pth')
        writer.close()
    print(timestamp(),'done')

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_arguments()
    train(args)





from torch import nn
import torch

class LinearProber(torch.nn.Module):
    def __init__(self, class_num=2, model_dim=768):
        super().__init__()
        self.linear = nn.Linear(model_dim, class_num)

    def forward(self, x):
        return self.linear(x)
    
class NonLinearProber(torch.nn.Module):
    def __init__(self, class_num=2, hidden_size=512, model_dim=768, dropout_prob = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, hidden_size)
        self.relu = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout()
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, class_num)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
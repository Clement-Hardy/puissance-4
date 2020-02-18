import torch.nn as nn
import torch.nn.functional as F


class ModelDuelings(nn.Module):
    
    def __init__(self, input_shape, output_shape):
        super(ModelDuelings, self).__init__()
        
        self.dim_space = input_shape
        self.nb_actions = output_shape
        
        self.fc1 = nn.Linear(self.dim_space, 64)
        
        self.advantage1 = nn.Linear(64, 64)
        self.advantage2 = nn.Linear(64, self.nb_actions)
        
        self.value1 = nn.Linear(64, 64)
        self.value2 = nn.Linear(64, 1)
        
    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        
        ad = self.advantage1(x)
        ad = F.relu(ad)
        ad = self.advantage2(ad)
        
        va = self.value1(x)
        va = F.relu(va)
        va = self.value2(va)
        
        return va + ad - ad.mean()
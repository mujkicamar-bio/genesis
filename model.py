import torch 
from torch import nn


class DNA_CNN_Deep(nn.Module):
    def __init__(self, seq_len, num_filters=[64, 128], kernel_sizes=[6, 5], dropout=0.3):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(4, num_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Second convolutional block broader
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveMaxPool1d(1)  # collapse sequence dimension

        # Fully connected layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters[1], 1)  # outputs binary logit
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # have to permute to have 4 channels for ACTG, similar to 3 RGB in image classification
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  
        
        x = x.squeeze(-1)  
        x = self.dropout(x)
        x = self.fc(x)      
        return x


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class SeqOHE(Dataset):

    def __init__(self, df, seq_col="oriv_sequence", target_col="type_seq"):

        self.seqs = list(df[seq_col].values)
        self.seq_len = len(self.seqs[0])

        
        nuc_d = {"A":[1.0,0.0,0.0,0.0],
                 "C":[0.0,1.0,0.0,0.0], 
                 "G":[0.0,0.0,1.0,0.0],
                 "T":[0.0,0.0,0.0,1.0],
                 "N":[0.0,0.0,0.0,0.0]}
        encoded = [torch.tensor(np.array([nuc_d[n] for n in seq]), dtype=torch.long) for seq in self.seqs]
        self.ohe_seqs = pad_sequence(encoded, batch_first=True)
        self.labels = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):

        return len(self.seqs)

    def __getitem__(self, idx):

        seq = self.ohe_seqs[idx]
        labels = self.labels[idx]
        return seq, labels, idx

def build_dataloaders(train_df, test_df, val_df, seq_col="oriv_sequence", target_col="type_seq", batch_size=32):



    #create datasetss
    
    train_ds = SeqOHE(train_df)
    test_ds = SeqOHE(test_df)
    val_ds = SeqOHE(val_df)
    


    # Dataloaders 

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    
    return train_dl, test_dl, val_dl



        



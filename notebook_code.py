
import numpy as np
import torch 
import h5py # pour gérer les formats de données utilisés ici 
import matplotlib.pyplot as plt

import gudhi as gd

from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import numpy as np

import scipy

import os

data_path = r'C:\Users\octav\Documents\Cours\MVA\Apprentissage pour signaux\TP3\Database\samples.hdf5'

data = h5py.File(data_path , 'r')

signals = np.array(data['signaux'])
snr =  np.array(data['snr'])
labels_id = np.array(data['labels'])

data.close()

def get_labels(open_h5_file): 
    return {
        open_h5_file['label_name'].attrs[k] : k
        for k in open_h5_file['label_name'].attrs.keys()
    }

data = h5py.File(data_path , 'r')

labels = get_labels(data)

data.close()


class DumbDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path_to_data,
                 ):
        super().__init__()

        
        data = h5py.File(path_to_data , 'r')

        self.signals = np.array(data['signaux'])
        self.snr =  np.array(data['snr'])
        self.labels_id = np.array(data['labels'])
        self.labels_lookup_table = get_labels(data)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        signal = torch.tensor(self.signals[i])
        signal = signal.permute(1, 0)       # Make it compatible with conv1d
        snr = torch.tensor(self.snr[i])
        label_id = torch.tensor(self.labels_id[i])
        return [signal, snr, label_id]
    
dataset = DumbDataset(path_to_data= data_path)


dataloader = torch.utils.data.DataLoader(dataset, 
                        batch_size=10, 
                        shuffle=True,
                        #collate_fn=collate_fn
                       )

class DumbModel(torch.nn.Module):
    def __init__(self,
                 signal_shape = (2048, 2)):
        super().__init__()

        self.operations = torch.nn.ModuleList([

            torch.nn.Conv1d(in_channels     =signal_shape[1],
                            out_channels    =5,
                            kernel_size     =10,
                            stride          = 8),

            torch.nn.ReLU(),

            torch.nn.Conv1d(in_channels     =5,
                            out_channels    =10,
                            kernel_size     =10,
                            stride          = 8),

            torch.nn.ReLU(),

            torch.nn.Conv1d(in_channels     =10,
                            out_channels    =15,
                            kernel_size     =10,
                            stride          = 32),

            # torch.nn.ReLU(),

            # torch.nn.Conv1d(in_channels     =15,
            #                 out_channels    =25,
            #                 kernel_size     =10,
            #                 stride          = 4),

            torch.nn.Softmax()
        ])

    def forward(self, signal):

        z = signal
        for index, operation in enumerate(self.operations):
            z = operation(z)
        
        return z
    
baseline = DumbModel()

with torch.no_grad():
    for batch in dataloader:
        pred = baseline(batch[0])
        print(pred.size())


print("done")
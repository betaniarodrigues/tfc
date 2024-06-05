import numpy as np
import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import pandas as pd
from sliding_window import sliding_window


def opp_sliding_window(data_x, data_y, ws, ss):
    
    # data_x são os dados, data_y são os rótulos, ws é o tamanho da janela = 50, ss é o overlap = 25
    
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)


# Defining the data loader for the implementation
class HARDataset(Dataset):
    def __init__(self, args, phase):
        self.filename = os.path.join(args.root_dir, args.data_file)

        self.data_raw = self.load_dataset()
        # Verifica se a dimensão 1 dos dados (canais) é igual ao input_size (6)
        assert args.input_size == self.data_raw[phase]['data'].shape[1]
        print("Data Raw", self.data_raw[phase]['data'].shape)
        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase]['data'],
                               self.data_raw[phase]['labels'],
                               args.window, args.overlap)
        print("Data", self.data.shape)
        print("Labels", self.labels.shape)

    # Load .csv file
    
    def load_dataset (self):

        data_path = Path(self.filename)

        datasets = {}

        for phase in ['train', 'val', 'test']:

            phase_path = data_path / phase

            datas_x = []

            data_y = []

            for f in phase_path.glob('*.csv'):

                data = pd.read_csv(f)

                x = data[['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']].values

                # Expend dimension

                #x = np.swapaxes(x, 1, 0)
                
                datas_x.append(x)

               # print("X", len(datas_x)

                y = data['activity code'].values

                data_y.append(y)

                #print("Y", data_y)
            
            datasets[phase] = {'data': np.concatenate(datas_x), 'labels': np.concatenate(data_y)}
            datasets[phase]['data'] = datasets[phase]['data'].astype(np.float32)
            # print(datasets[phase]['data'].shape)
            # print("Phase", phase)
            datasets[phase]['labels'] = datasets[phase]['labels'].astype(np.uint8)
            # print(datasets[phase]['labels'].shape)
            #exit(0)
            
        return datasets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        data = torch.from_numpy(data).double()

        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


def load_dataset(args, classifier=False):
    datasets = {x: HARDataset(args=args, phase=x) for x in
                ['train', 'val', 'test']}

    def get_batch_size():
        if classifier:
            batch_size = args.classifier_batch_size
        else:
            batch_size = args.batch_size

        return batch_size
    
    # Get a subset of the dataset

    def get_subset(dataset, percentage):
        subset_size = int(len(dataset) * (percentage / 100.0))
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        return Subset(dataset, indices)

    data_loaders = {
    x: DataLoader(
        get_subset(datasets[x], args.data_percentage) if x == 'train' else datasets[x],
        batch_size=get_batch_size(),
        shuffle=True if x == 'train' else False,
        num_workers=2, pin_memory=True
    )

    for x in ['train', 'val', 'test']
}

    # Printing the batch sizes
    for phase in ['train', 'val', 'test']:
        print('The batch size for {} phase is: {}'
              .format(phase, data_loaders[phase].batch_size))

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
   # print(dataset_sizes)
    
    print("Dataset sizes:", dataset_sizes)
    for phase in ['train', 'val', 'test']:
        print(f"Dataset size for {phase}: {dataset_sizes[phase]}")

    return data_loaders, dataset_sizes




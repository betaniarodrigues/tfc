import numpy as np
import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd

from sliding_window import sliding_window


def opp_sliding_window(data_x, data_y, ws, ss):
    print("DATA X", data_x.shape)
    # Para cada usuário
    for i in range(0, data_x.shape[0]):
        # Para cada sensor
        for y in range(0, data_x.shape[2]):
            data_x = sliding_window(data_x[0], (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)


# Defining the data loader for the implementation
class HARDataset(Dataset):
    def __init__(self, args, phase):
        self.filename = os.path.join(args.root_dir, args.data_file)

        # # If the prepared dataset doesn't exist, give a message and exit
        # if not os.path.isfile(self.filename):
        #     print('The data is not available. '
        #           'Ensure that the data is present in the directory.')
        #     exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset()
        assert args.input_size == self.data_raw[phase][0].shape[1]

        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase][0],
                               self.data_raw[phase][1],
                               args.window, args.overlap)

    def load_dataset(self):
        data_path = Path(self.filename)
        data_raw = {}

        max_length = 0  # Inicializa a maior dimensão encontrada como zero

        # Percorre as fases 'train', 'val' e 'test'
        for phase in ['train', 'val', 'test']:
            phase_path = data_path / phase
            datas_x = []
            data_y = []

            # Percorre os arquivos CSV na fase atual
            for f in phase_path.glob('*.csv'):
                data = pd.read_csv(f)
                x = data[['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']].values
                x = np.swapaxes(x, 1, 0)
               # print("X", x.shape)

                # Atualiza a maior dimensão encontrada
                max_length = max(max_length, x.shape[1])

               # print("MAX LENGTH", max_length)

                datas_x.append(x)
                
                y = data['standard activity code'].values
                print("Y", y.shape)
                data_y.append(y)

            datas_x_padded = []  # Lista para armazenar arrays preenchidos com zero
            datas_y_padded = []  # Lista para armazenar arrays preenchidos com zero

            # Percorre os arrays x na fase atual
            for x in datas_x:
                # Calcula o número de zeros a serem adicionados
                padding_length = max_length - x.shape[1]
                #print("PADDING LENGTH", padding_length)
                #print("X SHAPE", x.shape)
                # Preenche o array com zeros à direita para igualar o tamanho máximo

                padded_x = np.pad(x, ((0, 0), (0, padding_length)), mode='constant', constant_values=0)

                print("PADDED X", padded_x.shape)
                datas_x_padded.append(padded_x)

            for y in data_y:
                padding_length = max_length - len(y)
                print("y", y.shape)
                padded_y = np.pad(y, (0, padding_length), mode='constant', constant_values=0)
                print("PADDED Y", padded_y.shape)
                datas_y_padded.append(padded_y)

            datas_x_padded = np.array(datas_x_padded)
            datas_y_padded = np.array(datas_y_padded)
            print("DATAS X PADDED", datas_x_padded.shape) # (21 - usuários, 6, max_length)
            print("DATA Y", datas_y_padded.shape) # (21 - usuários, max_length
            data_raw[phase] = (datas_x_padded, datas_y_padded)

          

            print("DATA RAW", data_raw[phase][0].shape, data_raw[phase][1].shape)

        return data_raw


    # def load_dataset(self):
    #     data_path = Path(self.filename)
    #     data_raw = {}

    #     for phase in ['train', 'val', 'test']:
    #         phase_path = data_path / phase
    #         datas_x = []
    #         data_y = []

    #         for f in phase_path.glob('*.csv'):
    #             data = pd.read_csv(f)
    #             x = data[['accel-x', 'accel-y', 'accel-z', 'gyro-x', 'gyro-y', 'gyro-z']].values
    #             x = np.swapaxes(x, 1, 0)
    #             datas_x.append(x)
                
    #             y = data['standard activity code'].values
    #             data_y.append(y)

    #         datas_x = np.array(datas_x)
    #         data_y = np.array(data_y)
    #         data_raw[phase] = (datas_x, data_y)

    #     return data_raw


    # def load_dataset(self, filename):
    #     data = loadmat(filename)
    #     data_raw = {'train': {'data': data['X_train'],
    #                           'labels': np.transpose(data['y_train'])},
    #                 'val': {'data': data['X_valid'],
    #                         'labels': np.transpose(data['y_valid'])},
    #                 'test': {'data': data['X_test'],
    #                          'labels': np.transpose(data['y_test'])}}

    #     for set in ['train', 'val', 'test']:
    #         data_raw[set]['data'] = data_raw[set]['data'].astype(np.float32)
    #         data_raw[set]['labels'] = data_raw[set]['labels'].astype(np.uint8)

    #     return data_raw

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

    data_loaders = {x: DataLoader(datasets[x],
                                  batch_size=get_batch_size(),
                                  shuffle=True if x == 'train' else False,
                                  num_workers=2, pin_memory=True)
                    for x in ['train', 'val', 'test']}

    # Printing the batch sizes
    for phase in ['train', 'val', 'test']:
        print('The batch size for {} phase is: {}'
              .format(phase, data_loaders[phase].batch_size))

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)

    return data_loaders, dataset_sizes



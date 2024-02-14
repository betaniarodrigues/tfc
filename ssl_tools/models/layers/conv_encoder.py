import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, num_channels = 6, encoding_size = 10, kernel_size=3, dropout_rate=0.2, device: str = "cpu", flatten=True):
        super(CNNEncoder, self).__init__()
        self.device = device

        self.flatten = flatten

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, encoding_size, kernel_size, padding=1), # 10 é o número de canais de saída -> encoding_size
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = x.to(self.device)  # Mova os dados para a GPU, se disponível
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)  # Permute para [batch_size, seq_len, num_channels]

        if self.flatten:
            #Aplana a duas primeiras dimensões (treinamento da tarefa pretexto)
            x = x.reshape(-1, x.size(-1))     
        else:
            # Mantém a dimensão do batch
            x = x.reshape(x.size(0), -1)   

        return x

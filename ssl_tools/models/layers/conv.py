import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels = 6, kernel_size=3, dropout_rate=0.2, device= 'cuda'):
        super(CNNEncoder, self).__init__()
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size, padding=1, padding_mode= 'reflect'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size, padding=1, padding_mode='reflect'), # 128 é fixo
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):

        x = x.to(self.device)  # Mova os dados para a GPU, se disponível
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.permute(0, 2, 1)  # Permute para [batch_size, seq_len, num_channels_saída]
     
        #Aplana a duas primeiras dimensões (treinamento da tarefa pretexto)
      #  x = x.reshape(-1, x.size(-1))      

      #  x = x.squeeze(0)

        return x
import torch
import lightning as L
import numpy as np

from ssl_tools.utils.configurable import Configurable
from ssl_tools.models.layers.gru import GRUEncoder
from ssl_tools.models.layers.conv import CNNEncoder as CNN


class CPC(L.LightningModule, Configurable):
    def __init__(
        self, 
        g_enc: torch.nn.Module, 
        density_estimator: torch.nn.Module,
        g_ar: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        window: int = 4,
        overlap: int = 2,
        num_steps_prediction: int = 2,
        batch_size: int = 64,
    ):
        super().__init__()

        self.g_enc = g_enc
        self.density_estimator = density_estimator
        self.g_ar = g_ar
        self.learning_rate = lr 
        self.weight_decay = weight_decay
        self.window = window
        self.overlap = overlap
        self.num_steps_prediction = num_steps_prediction
        self.batch_size = batch_size
        
        self.Wk = torch.nn.ModuleList([self.density_estimator
                                 for _ in range(num_steps_prediction)])

        self.softmax = torch.nn.Softmax(dim=1)
        self.lsoftmax = torch.nn.LogSoftmax(dim=1)

        self.batch_size = batch_size
        self.seq_len = window
        self.num_steps_prediction = num_steps_prediction
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, inputs):
        z = self.g_enc(inputs)
        start = torch.randint(int(inputs.shape[1] - self.num_steps_prediction),
                              size=(1,)).long()
        print("start: ", start)
        rnn_input = z[:, :start + 1, :]
        r_out, _ = self.g_ar(rnn_input, None)

        accuracy, nce, correct_steps = self.compute_cpc_loss(z, r_out, start)
        return accuracy, nce, correct_steps

    def compute_cpc_loss(self, z, c, t):
        batch_size = z.shape[0]
        c_t = c[:, t, :].squeeze(1)
        pred = torch.stack([self.Wk[k](c_t) for k in range(self.num_steps_prediction)])

        z_samples = z[:, t + 1: t + 1 + self.num_steps_prediction, :].permute(1, 0, 2)

        print("z_samples: ", z_samples.shape)

        nce = 0
        correct = 0
        correct_steps = []

        for k in range(self.num_steps_prediction):
            log_density_ratio = torch.mm(z_samples[k], pred[k].transpose(0, 1))
            print("log_density_ratio: ", log_density_ratio.shape)
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio), dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)
            correct = (correct + torch.sum(torch.eq(positive_batch_pred, positive_batch_actual)).item())
            correct_steps.append(torch.sum(torch.eq(positive_batch_pred, positive_batch_actual)).item())
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))
            print("nce: ", nce)

        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        accuracy = correct / (1.0 * batch_size * self.num_steps_prediction)
        correct_steps = torch.tensor(correct_steps)
        return accuracy, nce, correct_steps

    def training_step(self, batch, batch_idx):
        inputs = batch
        print("inputs: ", inputs.shape)
        accuracy, nce, _ = self.forward(inputs)
        #print("nce: ", nce)
        self.log('train_loss',
                nce,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
        )
        return nce

    def validation_step(self, batch, batch_idx):
        inputs = batch
        accuracy, nce, _ = self.forward(inputs)
        self.log('val_loss',
                nce,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,)
        return nce

    def test_step(self, batch, batch_idx):
        inputs = batch
        accuracy, nce, _ = self.forward(inputs)
        self.log('test_loss',
                nce,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,)
        return nce

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def get_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "window": self.window,
            "overlap": self.overlap,
        }

# Define the GRU encoder

def build_cpc(
    encoding_size: int = 150,
    in_channels: int = 6,
    gru_hidden_size: int = 100,
    gru_num_layers: int = 1,
    gru_bidirectional: bool = True,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    window: int = 4,
) -> CPC:
    """Builds a default CPC model. This function aid in the creation of a CPC
    model, by setting the default values of the parameters.

    Parameters
    ----------
    encoding_size : int, optional
        Size of the encoded representation (the output of the linear layer).
    in_channel : int, optional
        Number of input features (e.g. 6 for HAR data in MotionSense Dataset)
    gru_hidden_size : int, optional
        The number of features in the hidden state of the GRU.
    gru_num_layers : int, optional
        Number of recurrent layers in the GRU. E.g., setting ``num_layers=2``
        would mean stacking two GRUs together to form a `stacked GRU`,
        with the second GRU taking in outputs of the first GRU and
        computing the final results.
    gru_bidirectional : bool, optional
        If ``True``, becomes a bidirectional GRU.
    dropout : float, optional
        The dropout probability.
    learning_rate : float, optional
        The learning rate of the optimizer.
    weight_decay : float, optional
        The weight decay of the optimizer.
    window_size : int, optional
        Size of the input windows (X_t) to be fed to the encoder
    n_size : int, optional
        Number of negative samples to be used in the contrastive loss
        (steps to predict)

    Returns
    -------
    CPC
        The CPC model
    """
    g_enc = GRUEncoder(
        hidden_size=gru_hidden_size,
        in_channels=in_channels,
        encoding_size=encoding_size,
        num_layers=gru_num_layers,
        dropout=dropout,
        bidirectional=gru_bidirectional,
    )

    density_estimator = torch.nn.Linear(encoding_size, encoding_size)

    g_ar = torch.nn.GRU(
        input_size=encoding_size,
        hidden_size=encoding_size,
        batch_first=True,
    )

    model = CPC(
        g_enc=g_enc,
        density_estimator=density_estimator,
        g_ar=g_ar,
        lr=learning_rate,
        weight_decay=weight_decay,
        window=window,
    )

    return model

def build_cpc_conv(
    encoding_size: int = 150,
    in_channels: int = 6,
    kernel_size: int = 3,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    window: int = 4,
    overlap: int = 50,
) -> CPC:
    """Builds a default CPC model. This function aid in the creation of a CPC
    model, by setting the default values of the parameters.

    Parameters
    ----------
    encoding_size : int, optional
        Size of the encoded representation (the output of the linear layer).
    in_channel : int, optional
        Number of input features (e.g. 6 for HAR data in MotionSense Dataset)
    kernel_size : int, optional
        The size of the convolutional kernel.
    learning_rate : float, optional
        The learning rate of the optimizer.
    weight_decay : float, optional
        The weight decay of the optimizer.
    window_size : int, optional
        Size of the input windows (X_t) to be fed to the encoder
    n_size : int, optional
        Number of negative samples to be used in the contrastive loss
        (steps to predict)

    Returns
    -------
    CPC
        The CPC model
    """
    g_enc = CNN(
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        kernel_size=kernel_size,
    )

    density_estimator = torch.nn.Linear(encoding_size, 128)

    g_ar = torch.nn.GRU(
        input_size=128,
        hidden_size=encoding_size,
        num_layers=2,
        bidirectional=False,
        batch_first=True,
        dropout=0.2,
    )

    model = CPC(
        g_enc=g_enc,
        density_estimator=density_estimator,
        g_ar=g_ar,
        lr=learning_rate,
        weight_decay=weight_decay,
        window=window,
        overlap=overlap,
    )

    return model



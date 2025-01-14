#!/usr/bin/env python3

import lightning as L

from ssl_tools.experiments import LightningTrain, auto_main
from ssl_tools.data.data_modules import (
    MultiModalHARSeriesDataModule,
)
from ssl_tools.models.nets import MLPClassifier
from ssl_tools.experiments.har_classification._classification_base import (
    EvaluatorBase,
)
from ssl_tools.experiments.har_classification.utils import (
    FFT,
    Flatten,
    Spectrogram,
)


class MLPClassifierTrain(LightningTrain):
    _MODEL_NAME = "MLP"

    def __init__(
        self,
        data: str,
        input_size: int = 360,
        hidden_size: int = 64,
        num_hidden_layers: int = 1,
        num_classes: int = 6,
        transforms: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes

        self.transforms = None
        assert transforms in ["identity", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True), Flatten()]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram(), Flatten()]
        else:
            self.transforms = [Flatten()]

    def get_model(self) -> L.LightningModule:
        model = MLPClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            output_size=self.num_classes,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=self.transforms
        )

        return data_module


class MLPClassifierTest(EvaluatorBase):
    _MODEL_NAME = "MLP"

    def __init__(
        self,
        data: str,
        input_size: int = 360,
        hidden_size: int = 64,
        num_hidden_layers: int = 1,
        num_classes: int = 6,
        transforms: str = "identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = data
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        
        self.transforms = None
        assert transforms in ["identity", "fft", "spectrogram"]
        if transforms == "fft":
            self.transforms = [FFT(absolute=True, centered=True), Flatten()]
        elif transforms == "spectrogram":
            self.transforms = [Spectrogram(), Flatten()]
        else:
            self.transforms = [Flatten()]

    def get_model(self) -> L.LightningModule:
        model = MLPClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            output_size=self.num_classes,
        )
        return model

    def get_data_module(self) -> L.LightningDataModule:
        data_module = MultiModalHARSeriesDataModule(
            data_path=self.data,
            batch_size=self.batch_size,
            label="standard activity code",
            features_as_channels=True,
            num_workers=self.num_workers,
            transforms=self.transforms
        )

        return data_module


if __name__ == "__main__":
    options = {
        "fit": MLPClassifierTrain,
        "test": MLPClassifierTest,
    }
    auto_main(options)

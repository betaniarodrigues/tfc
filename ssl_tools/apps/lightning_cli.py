from typing import Union
import logging

class LightningTrainCLI:
    _APP_NAME = "LightningTrainCLI"
    _LOG_FORMAT = "[%(name)s] %(asctime)s - %(levelname)s - %(message)s"
    
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
        log_dir: str = "logs",
        name: str = None,
        load: str = None,
        resume: str = None,
        version: Union[str, int] = None,
        checkpoint_metric: str = None,
        checkpoint_metric_mode: str = "min",
        accelerator: str = "cpu",
        devices: int = 1,
        strategy: str = "auto",
        limit_train_batches: Union[float, int] = 1.0,
        limit_val_batches: Union[float, int] = 1.0,
        num_nodes: int = 1,
        num_workers: int = None,
        seed: int = None,
        verbose: int = 1
    ):
        """Defines a Main CLI for pre-training Pytorch Lightning models

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to pre-train the model
        batch_size : int, optional
            The batch size
        learning_rate : float, optional
            The learning rate of the optimizer
        log_dir : str, optional
            Path to the location where logs will be stored
        name: str, optional
            The name of the experiment (will be used as a prefix for the logs 
            and checkpoints). If not provided, the name of the model will be 
            used
        version: Union[int, str], optional
            The version of the experiment. If not is provided the current date 
            and time will be used as the version
        load: str, optional
            The path to a checkpoint to load
        resume: str, optional
            The path to a checkpoint to resume training
        checkpoint_metric: str, optional
            The metric to monitor for checkpointing. If not provided, the last 
            model will be saved
        checkpoint_metric_mode: str, optional
            The mode of the metric to monitor (min, max or mean). Defaults to 
            "min"
        accelerator: str, optional
            The accelerator to use. Defaults to "cpu"
        devices: int, optional
            The number of devices to use. Defaults to 1
        strategy: str, optional
            The strategy to use. Defaults to "auto"
        limit_train_batches: Union[float, int], optional
            The number of batches to use for training. Defaults to 1.0 (use 
            all batches)
        limit_val_batches: Union[float, int], optional
            The number of batches to use for validation. Defaults to 1.0 (use 
            all batches)
        num_nodes: int, optional
            The number of nodes to use. Defaults to 1
        num_workers: int, optional
            The number of workers to use for the dataloader. 
        seed: int, optional
            The seed to use.
        verbose: int, optional
            The verbosity level. Defaults to 1
            0: CRITICAL; ERROR
            1: CRITICAL; ERROR; INFO
            2. CRITICAL; ERROR; INFO; DEBUG
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_dir = log_dir
        self.experiment_name = name
        self.experiment_version = version
        self.load = load
        self.resume = resume
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_metric_mode = checkpoint_metric_mode
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.seed = seed
        self.verbose = verbose
        self._logger = self._setup_log()

    def _convert_log_level(self, verbose):
        if verbose == 0:
            return "ERROR"
        elif verbose == 1:
            return "INFO"
        elif verbose == 2:
            return "DEBUG"
        else:
            raise ValueError(f"Invalid verbose level: {verbose}")
        
    def _setup_log(self):
        level = self._convert_log_level(self.verbose)

        logger = logging.getLogger(self._APP_NAME)
        logger.setLevel(level)
        
        ch = logging.StreamHandler()
        ch.setLevel(level)
        
        formatter = logging.Formatter(self._LOG_FORMAT)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger 

    @property
    def log(self):
        return self._logger.info
    
    @property
    def logger(self):
        return self._logger

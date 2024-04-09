import yaml
from pathlib import Path
from typing import Any, Dict, Union
from abc import ABC, abstractmethod
from datetime import datetime
from jsonargparse import ActionConfigFile, ArgumentParser

EXPERIMENT_VERSION_FORMAT = "%Y-%m-%d_%H-%M-%S"


class Experiment(ABC):
    def __init__(
        self,
        name: str = "experiment",
        run_id: Union[str, int] = None,
        log_dir: str = "logs",
        seed: int = None,
    ):
        self.name = name
        self.run_id = run_id or datetime.now().strftime(
            EXPERIMENT_VERSION_FORMAT
        )
        self.log_dir = log_dir
        self.seed = seed

    @property
    def experiment_dir(self) -> Path:
        return Path(self.log_dir) / self.name / str(self.run_id)

    def setup(self):
        pass

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError

    def teardown(self):
        pass

    def execute(self):
        print(f"Setting up experiment: {self.name}...")
        self.setup()
        print(f"Running experiment: {self.name}...")
        result = self.run()
        print(f"Teardown experiment: {self.name}...")
        self.teardown()
        return result

    def __call__(self):
        return self.execute()

    def __str__(self):
        return f"Experiment(name={self.name}, run_id={self.run_id}, cwd={self.experiment_dir})"

    def __repr__(self) -> str:
        return str(self)


def get_parser(commands: Dict[str, Experiment]):
    parser = ArgumentParser()
    subcommands = parser.add_subcommands()

    for name, command in commands.items():
        subparser = ArgumentParser()
        subparser.add_class_arguments(command)
        subparser.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file, in YAML format",
        )
        subcommands.add_subcommand(name, subparser)

    return parser


# Quick and dirty implementation of auto_main
def auto_main(commands: Dict[str, Experiment], print_args: bool = False) -> Any:
    parser = get_parser(commands)

    args = parser.parse_args()
    config_file = args[args.subcommand].pop("config", None)

    if config_file:
        config_file = config_file[0].absolute
        with open(config_file, "r") as f:
            config_from_file = yaml.safe_load(f)

        config = config_from_file
        config.update(args[args.subcommand])
    else:
        config = dict(args[args.subcommand])
        
    if print_args:
        print(config)

    experiment: Experiment = commands[args.subcommand](**config)
    return experiment.execute()

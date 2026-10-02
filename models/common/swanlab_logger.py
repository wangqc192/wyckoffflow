"""SwanLab adapter for the project's pytorch_lightning package."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class SwanLabLogger(Logger):
    def __init__(
        self,
        save_dir,
        project="wyckoffflow-pl",
        workspace=None,
        experiment_name=None,
        mode="online",
        id=None,
        hyperparameters=None,
    ):
        super().__init__()
        self._save_dir = str(save_dir)
        self._experiment = None
        self._id = id
        self._kwargs = dict(
            project=project,
            workspace=workspace,
            name=experiment_name or Path(save_dir).name,
            mode=mode,
            config=hyperparameters,
            log_dir=str(Path(save_dir) / "swanlab"),
        )

    @property
    def name(self):
        return self._kwargs["project"]

    @property
    def version(self):
        return self._id

    @property
    def save_dir(self):
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            import swanlab

            self._experiment = swanlab.init(
                **self._kwargs,
                id=self._id,
                resume="allow",
                settings=swanlab.Settings(interactive=False),
            )
            self._id = self._experiment.id
            url = self._experiment.url if self._kwargs["mode"] == "online" else None
            directory = Path(self._kwargs["log_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "run.json").write_text(
                json.dumps({"id": self._id, "url": url}, indent=2) + "\n"
            )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        if OmegaConf.is_config(params):
            params = OmegaConf.to_container(params, resolve=True)
        self.experiment.config.update(dict(params))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self.experiment.log(dict(metrics), step=step)

    @rank_zero_only
    def finalize(self, status):
        if self._experiment is not None:
            self._experiment.finish(
                state="success" if status == "success" else "crashed"
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state

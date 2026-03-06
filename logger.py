import os
import logging
import warnings

import numpy as np
import torch
import re
from dotenv import load_dotenv
load_dotenv()


class Logger:
    """Wrapper for logging servers

    Parameters
    ----------
    experiment_name : str
        name of logging epxeriment
    port : int, optional
        port number, by default 27027
    api : str, optional
        name of logging server module, by default "mlflow"
    enabled : bool, optional
        enable logging, by default True

    Attributes
    ----------
    metrics : dict
        metrics temporarily saved in the logger
    """

    def __init__(
        self,
        experiment_name: str,
        port: int = 27027,
        api: str = "mlflow",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.api = api.lower().strip()
        self.__implemented_apis = ["mlflow", "wandb"]
        assert (
            self.api in self.__implemented_apis
        ), "Choose implemented tracking API from {self.__implemented_apis}. Found {self.api}"
        self.__choose_module()

        if self.enabled:
            self.experiment_name = experiment_name
            self.default_port = port

            self.metrics: dict = {}
            logging.getLogger(self.api).setLevel(logging.DEBUG)

    def setup_tracking(self, online: bool = False, port: int | None = None, **kwargs) -> None:
        """Set up remote tracking with logging server

        Parameters
        ----------
        online : bool, optional
            connect with logging server online instead of locally, by default False
        port : int | None, optional
            port number, by default None
        """
        if not self.enabled:
            return
        if port is None:
            port = self.default_port
        if self.api == "mlflow":
            self.__setup_mlflow_tracking(self.experiment_name, port, online, **kwargs)
        if self.api == "wandb":
            self.__setup_wandb_tracking(self.experiment_name, **kwargs)

    def __call__(self, **run_kwargs) -> "Logger":
        self.run_kwargs = run_kwargs
        return self

    def __enter__(self):
        self.start_run(**self.run_kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()

    def start_run(self, **kwargs) -> None:
        """Start Logging run"""
        if not self.enabled:
            return
        if self.api == "mlflow":
            self.__start_mlflow_run(**kwargs)
        if self.api == "wandb":
            self.__start_wandb_run(**kwargs)

    def end_run(self):
        """End logging run"""
        if not self.enabled:
            return
        if self.api == "mlflow":
            self.__end_mlflow_run()
        if self.api == "wandb":
            self.__end_wandb_run()

    def log_parameter(self, key: str, value: str) -> None:
        """Log a parameter

        Parameters
        ----------
        key : str
            name of parameter
        value : str
            value of parameter
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            mlflow.log_param(key, value)
        if self.api == "wandb":
            wandb.config.update({key: value})

    def log_artifact(self, path: str|list[str], name: str = "") -> None:
        """Log an artifact

        Parameters
        ----------
        path : str
            file path of artifact
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            if isinstance(path, list):
                for _path in path:
                    mlflow.log_artifact(_path)
            else:
                mlflow.log_artifact(path)
        if self.api == "wandb":
            name = f"{name}_{wandb.run.id}"
            artifact = wandb.Artifact(name=name, type="report")
            if isinstance(path, list):
                for _path in path:
                    artifact.add_file(_path)
            else:
                artifact.add_file(path)
            wandb.log_artifact(artifact)

    def load_artifact(self, artifact_path:str|list[str]) -> list[str]:
        if isinstance(artifact_path, str):
            artifact_path = [artifact_path]
        local_paths = []
        if self.api == "wandb":
            api = wandb.Api()
            for path in artifact_path:
                artifact = api.artifact(path)
                local_path = artifact.download()
                local_paths.append(local_path)
        return local_paths

    def log_metrics(self, metrics: dict, step: int, step_name: str = None) -> None:
        """Log multiple metrics

        Parameters
        ----------
        metrics : dict
            metrics dictionary
        step : int
            index in time of metrics
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        if self.api == "wandb":
            if step_name:
                metrics[step_name] = step
            wandb.log(metrics)

    def log_metric(self, key: str, value, step: int, step_name: str = None) -> None:
        """Log single metric

        Parameters
        ----------
        key : str
            name of metric
        value : _type_
            value of metric
        step : int
            index in time of metric
        """
        if not self.enabled:
            return
        if self.api == "mlflow":
            key = re.sub("/", ".", key)
            mlflow.log_metric(key, value, step)
        if self.api == "wandb":
            metrics = {key: value}
            if step_name:
                metrics[step_name] = step
            wandb.log(metrics)

    def log_metric_with_stats(self, name: str, value: torch.Tensor, step: int, step_name:str = None) -> None:
        """Log several matrix metrics on mlflow
        If the tensor is one-dimentional simply log its value

        Parameters
        ----------
        name : str
            name of the matrix
        value : torch.Tensor
            value of matrix
        step : int
            index in time of metric
        """
        if not self.enabled:
            return
        if not torch.is_tensor(value):
            self.log_metric(name, value, step=step)
            return
        elif value.dim() < 1:
            self.log_metric(name, value, step=step)
            return

        self.log_metric(f"{name}/mean", torch.mean(value), step=step, step_name=step_name)
        self.log_metric(f"{name}/median", torch.median(value), step=step, step_name=step_name)
        self.log_metric(f"{name}/std", torch.std(value), step=step, step_name=step_name)
        self.log_metric(f"{name}/max", torch.max(value), step=step, step_name=step_name)
        self.log_metric(f"{name}/fr-norm", torch.linalg.norm(value), step=step, step_name=step_name)
        self.log_metric(
            f"{name}/norm",
            torch.linalg.norm(value) / np.sqrt(torch.numel(value)),
            step=step,
            step_name=step_name,
        )

    def log_histogram(self, name: str, value: torch.Tensor, step: int , step_name: str = None) -> None:
        if not self.enabled:
            return
        if self.api == "wandb":
            metrics = {
                name: wandb.Histogram(
                    value.detach().cpu().numpy()
                )
            }
            if step_name:
                metrics[step_name] = step
            wandb.log(metrics)

    def watch_pytorch_model(self, model) -> None:
        if not self.enabled:
            return
        if self.api == "wandb":
            wandb.watch(model, log="all")

    def log_pytorch_model(self, model, name, x, path="", run_id: bool = True) -> None:
        if not self.enabled:
            return
        if self.api == "mlflow":
            signature = mlflow.models.infer_signature(
                x.to("cpu").numpy(), model(x).detach().cpu().numpy()
            )
            mlflow.pytorch.log_model(model, f"{name}", signature=signature)
        if self.api == "wandb":
            if run_id:
                name = f"{name}_{wandb.run.id}"
            file_path = os.path.join(path, f"{name}_model.pt")
            torch.save(model.state_dict(), file_path)
            artifact = wandb.Artifact(name=name, type="model")
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
    
    def load_pytorch_model(self, artifact_path: str) -> str:
        if self.api == "wandb":
            api = wandb.Api()
            artifact = api.artifact(artifact_path, type="model")
            local_path = artifact.download()
            return local_path+"/model.pt"
        return ""

    def save_metrics(self, metrics: dict) -> None:
        """Save multiple growth statistics for later logging

        Parameters
        ----------
        metrics : dict
            metrics dictionary
        """
        if not self.enabled:
            return
        for key, value in metrics.items():
            self.save_metric(key, value)

    def save_metric(self, name: str, value: torch.Tensor | int | float) -> None:
        """Save growth statistics for later logging

        Parameters
        ----------
        name : str
            name of tensor
        value : torch.Tensor | int | float
            value of tensor
        """
        if not self.enabled:
            return
        self.metrics[name] = value

    def log_all_metrics(self, step: int, with_stats: bool = True, step_name: str = None) -> None:
        """Log all saved metrics

        Parameters
        ----------
        step : int
            index in time of metrics
        with_stats : bool, optional
            if there are matrices log their statistics, by default True
        """
        if not self.enabled:
            return
        for key, value in self.metrics.items():
            if with_stats:
                self.log_metric_with_stats(key, value, step, step_name=step_name)
            else:
                self.log_metric(key, value, step, step_name=step_name)

    def clear(self) -> None:
        """Clear all saved metrics"""
        if not self.enabled:
            return
        self.metrics.clear()

    def __choose_module(self) -> None:
        if not self.enabled:
            return
        try:
            if self.api == "mlflow":
                global mlflow
                import mlflow
            if self.api == "wandb":
                global wandb
                import wandb
        except ImportError as err:
            warnings.warn(f"{err}. Logging will be skipped.", ImportWarning)
            self.enabled = False

    def __setup_mlflow_tracking(
        self, experiment_name: str, port: int = 27027, online: bool = False, file_path: str = None
    ) -> None:
        """Set up mlflow online tracking

        Parameters
        ----------
        experiment_name : str
            name for experiment bucket on mlflow server
        port : int, optional
            port number, by default 27027
        online : bool, optional
            connect with mlflow server online instead of locally, by default False

        Returns
        -------
        str
            tracking uri
        """
        uri = f"http://127.0.0.1:{port}"
        if online:
            mlflow.set_tracking_uri(uri=uri)
        elif file_path is not None:
            mlflow.set_tracking_uri(uri=file_path)
        mlflow.set_experiment(experiment_name)
        print(f"Tracking at {mlflow.get_tracking_uri()}")
    
    def __start_mlflow_run(self, tags: dict | None = None, **kwargs) -> None:
        mlflow.start_run(log_system_metrics=True, tags=tags, **kwargs)
    
    def __end_mlflow_run(self) -> None:
        mlflow.end_run()

    def __setup_wandb_tracking(
            self, experiment_name: str, file_path: str = None, **kwargs
    ):
        """Set up wandb online tracking

        Parameters
        ----------
        experiment_name : str
            name for experiment bucket on mlflow server
        """
        api_key = os.environ.get("WANDB_KEY")
        wandb.login(key=api_key)
        self.file_path = file_path

    def __start_wandb_run(self, tags: dict = {}, **kwargs) -> None:
        """Start Weights & Biases run"""
        _tags = [f"{key}:{value}" for key, value in tags.items() if value is not None]
        wandb.init(
            project=self.experiment_name,
            tags=_tags,
            save_code=False,
            dir=self.file_path,
            **kwargs
        )

    def __end_wandb_run(self) -> None:
        """End Weights & Biases run"""
        wandb.finish()
        
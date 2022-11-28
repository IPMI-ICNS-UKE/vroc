from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from aim.sdk.repo import Repo, RepoStatus, Run
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vroc.common_types import Number, PathLike
from vroc.decorators import convert, timing
from vroc.helper import concat_dicts
from vroc.logger import LoggerMixin


class BaseTrainer(ABC, LoggerMixin):
    METRICS = {}

    @convert("run_folder", converter=Path)
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        run_folder: PathLike | None = None,
        experiment_name: str | None = None,
        device: str = "cuda",
    ):
        run_folder: Path

        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.run_folder = run_folder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_folder = run_folder
        self.run_folder.mkdir(parents=True, exist_ok=True)

        self.model = model.to(device=self.device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

        self._aim_folder = self.run_folder / "aim"
        repo = BaseTrainer._get_aim_repo(self._aim_folder)
        self.aim_run = Run(repo=repo, experiment=experiment_name)
        self._set_run_params()

        self._model_folder = self.run_folder / f"models_{self.aim_run.hash}"
        self._model_folder.mkdir(parents=True, exist_ok=True)

        # metric tracking and model saving
        self.val_model_saver = BestModelSaver(
            tracked_metrics=self.METRICS,
            model=self.model,
            optimizer=self.optimizer,
            output_folder=self._model_folder / "validation",
            top_k=3,
        )
        self.train_model_saver = BestModelSaver(
            tracked_metrics={"step": MetricType.LARGER_IS_BETTER},
            model=self.model,
            optimizer=self.optimizer,
            output_folder=self._model_folder / "training",
            top_k=10,
        )

        # dict for metric history (for, e.g., calculating running means)
        self._metric_history = {}

        # training step and epoch tracking
        self.i_step = 0
        self.i_epoch = 0

    def _set_run_params(self):
        params = {
            "model": self.model.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
        }
        self.aim_run["params"] = params

    @staticmethod
    def _get_aim_repo(path: PathLike) -> Repo:
        path = str(path)
        repo_status = Repo.check_repo_status(path)
        if repo_status == RepoStatus.MISSING:
            repo = Repo.from_path(path, init=True)
        elif repo_status == RepoStatus.UPDATE_REQUIRED:
            raise RuntimeError("Please upgrade repo")
        else:
            repo = Repo.from_path(path)

        return repo

    def _prefixed_log(self, message, context: str, level: int):
        context = context.upper()
        prefix = f"[Step {self.i_step:<6} | {context:>6}]"
        self.logger.log(level=level, msg=f"{prefix} {message}", stacklevel=3)

    def log_info(self, message, context: str):
        self._prefixed_log(message, context=context, level=logging.INFO)

    def log_debug(self, message, context: str):
        self._prefixed_log(message, context=context, level=logging.DEBUG)

    @timing()
    def _track_metrics(self, metrics: dict, context: dict | None = None):
        for metric_name, metic_value in metrics.items():
            # metric value may be a list containing a value for each sample
            if (
                isinstance(metic_value, list)
                and len(metic_value) > 0
                and isinstance(metic_value[0], (int, float))
            ):
                metric_name = f"mean_batch_{metric_name}"
                metic_value = np.mean(metic_value)

            if isinstance(metic_value, (int, float)):
                subset = context["subset"]
                self._metric_history.setdefault(subset, {})
                history = self._metric_history[subset].setdefault(
                    metric_name, deque(maxlen=100)
                )
                history.append(metic_value)
                self.log_info(
                    f"Running mean of {metric_name}: {np.mean(history):.6f}",
                    context=subset,
                )

            self.aim_run.track(
                metic_value,
                epoch=self.i_epoch,
                step=self.i_step,
                name=metric_name,
                context=context,
            )

    @timing()
    def train_one_epoch(self):
        self.model.train()
        self.log_info("started epoch", context="TRAIN")
        for data in self.train_loader:
            batch_metrics = self._train_on_batch(data)

            batch_metrics_formatted = self._format_metric_dict(batch_metrics)
            self.log_debug(f"metrics: {batch_metrics_formatted}", context="TRAIN")

            self._track_metrics(batch_metrics, context={"subset": "train"})

            yield batch_metrics

            self.i_step += 1
        self.i_epoch += 1
        self.log_info("finished epoch", context="TRAIN")

    def _format_metric_dict(self, metrics: dict) -> str:
        metrics_formatted = {
            key: f"{value:.4f}" if isinstance(value, float) else str(value)
            for key, value in metrics.items()
        }
        formatted_str = " / ".join(
            f"{key}: {value}" for key, value in metrics_formatted.items()
        )

        return formatted_str

    @timing()
    def validate(self):
        if self.val_loader is None:
            raise RuntimeError("No validation loader given")

        self.log_info("started validation", context="VAL")
        # set to eval mode
        self.model.eval()

        metrics = []
        for data in self.val_loader:
            batch_metrics = self._validate_on_batch(data)
            batch_metrics_formatted = self._format_metric_dict(batch_metrics)

            self.log_debug(f"metrics: {batch_metrics_formatted}", context="VAL")
            metrics.append(batch_metrics)

        metrics = concat_dicts(metrics, extend_lists=True)

        self._track_metrics(metrics, context={"subset": "val"})
        self.val_model_saver.track(metrics, step=self.i_step)
        self.log_info("finished validation", context="VAL")

        # set to train mode again
        self.model.train()

    @timing()
    def run(
        self,
        steps: int = 10_000,
        validation_interval: int = 1000,
        save_interval: int = 1000,
    ):
        self.log_info("started run", context="RUN")
        self.i_step = 0
        self.i_epoch = 0
        training_finished = False
        while True:
            # train one epoch, i.e. one dataset iteration
            for batch_metrics in self.train_one_epoch():
                if (
                    self.i_step > 0
                    and self.val_loader is not None
                    and self.i_step % validation_interval == 0
                ):
                    # run validation at given intervals (if validation loader is given)
                    self.validate()
                elif save_interval and self.i_step % save_interval == 0:
                    # save model without validation
                    self.train_model_saver.track(
                        {"step": self.i_step}, step=self.i_step
                    )

                if self.i_step >= steps:
                    # stop training
                    training_finished = True
                    break

            if training_finished:
                self.log_info("finished training", context="TRAIN")
                break

    @timing()
    def _train_on_batch(self, data: dict) -> dict:
        return self.train_on_batch(data=data)

    @timing()
    def _validate_on_batch(self, data: dict) -> dict:
        return self.validate_on_batch(data=data)

    @abstractmethod
    def train_on_batch(self, data: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def validate_on_batch(self, data: dict) -> dict:
        raise NotImplementedError


class MetricType(Enum):
    SMALLER_IS_BETTER = auto()
    LARGER_IS_BETTER = auto()


class BestModelSaver(LoggerMixin):
    @convert("output_folder", converter=Path)
    def __init__(
        self,
        tracked_metrics: dict[str, MetricType],
        model: nn.Module,
        output_folder: PathLike,
        top_k: int = 1,
        model_name: str | None = None,
        optimizer: Optimizer | None = None,
        move_to_cpu: bool = True,
    ):
        output_folder: Path

        self.model = model
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.model_name = model_name
        self.optimizer = optimizer
        self.move_to_cpu = move_to_cpu

        self._tracked_metrics = tracked_metrics
        self._best_metrics = {
            metric_name: deque(maxlen=self.top_k)
            for metric_name in tracked_metrics.keys()
        }
        self._best_models = {
            metric_name: deque(maxlen=self.top_k)
            for metric_name in tracked_metrics.keys()
        }

    def _save_model(self, output_filepath: Path):
        if output_filepath.is_file():
            # model already saved
            self.logger.info(f"Model {output_filepath} already saved")
            return
        model_state = self.model.state_dict()
        if self.move_to_cpu:
            model_state = {k: v.cpu() for k, v in model_state.items()}
        state = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
        }
        torch.save(state, output_filepath)
        with open(output_filepath.parent / "models.yaml", "wt") as f:
            yaml.dump(
                {
                    metric_name: {
                        "values": list(self._best_metrics[metric_name]),
                        "models": [
                            model_filepath.name
                            for model_filepath in self._best_models[metric_name]
                        ],
                    }
                    for metric_name in self._tracked_metrics
                },
                f,
            )

        self.logger.info(f"Saved model to {output_filepath}")

    def _model_is_referenced(self, model_filepath: Path) -> bool:
        is_referenced = False
        for model_filepaths in self._best_models.values():
            if model_filepath in model_filepaths:
                is_referenced = True
                break

        return is_referenced

    def track(self, metrics: dict[str, Number | Sequence[Number]], step: int):
        if self.model_name:
            output_filepath = (
                self.output_folder / f"{self.model_name.lower()}_step_{step:03d}.pth"
            )
        else:
            output_filepath = self.output_folder / f"step_{step:03d}.pth"

        for metric_name, metric_value in metrics.items():
            if metric_name not in self._tracked_metrics:
                # skip this metric (tracking disabled for this metric)
                continue

            if isinstance(metric_value, (np.ndarray, list, tuple)) and isinstance(
                metric_value[0], (int, float)
            ):
                # multiple metric valued passed, e.g. for each sample in batch,
                # take the mean
                metric_value = float(np.mean(metric_value))
            else:
                # convert to float if value is numpy 0-dim float
                metric_value = float(metric_value)

            metric_type = self._tracked_metrics[metric_name]
            best_so_far = self._best_metrics[metric_name]

            if not best_so_far:
                # no metric tracked so far (length of deque == 0),
                # thus current metric is best
                new_best_metric = True
            elif (
                metric_value < best_so_far[-1]
                and metric_type == MetricType.SMALLER_IS_BETTER
            ):
                new_best_metric = True
            elif (
                metric_value > best_so_far[-1]
                and metric_type == MetricType.LARGER_IS_BETTER
            ):
                new_best_metric = True
            else:
                new_best_metric = False

            if new_best_metric:
                self.logger.info(f"New best {metric_name}: {metric_value}")

                if len(
                    self._best_models[metric_name]
                ) == self.top_k and not self._model_is_referenced(
                    remove_model_filepath := self._best_models[metric_name][0]
                ):
                    # remove worst model from top k best models
                    logging.debug(f"Remove model {remove_model_filepath}")
                    os.remove(remove_model_filepath)

                self._best_metrics[metric_name].append(metric_value)
                self._best_models[metric_name].append(output_filepath)

                self._save_model(output_filepath)

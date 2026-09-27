# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local experiment metrics, configuration, and validation-generation logging."""
import dataclasses
import json
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class Tracking:
    supported_backend = ["console", "jsonl", "tensorboard"]

    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        backends = [default_backend] if isinstance(default_backend, str) else default_backend
        self.logger = {}
        for backend in backends:
            if backend not in self.supported_backend:
                raise ValueError(f"Use a local logging backend: {self.supported_backend}")
        self.directory = Path((config or {}).get("release_log_dir") or
                              os.environ.get("SIPO_LOG_DIR", "runs/logs"))
        self.directory.mkdir(parents=True, exist_ok=True)
        os.environ["SIPO_LOG_DIR"] = str(self.directory)
        if config is not None:
            (self.directory / "config.json").write_text(json.dumps(config, indent=2, default=_json_default))
        if "console" in backends:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.logger["console"] = LocalLogger(print_to_console=True)
        if "jsonl" in backends:
            self.logger["jsonl"] = _JsonlAdapter(self.directory / "metrics.jsonl")
        if "tensorboard" in backends:
            self.logger["tensorboard"] = _TensorboardAdapter(self.directory / "tensorboard")

    def log(self, data, step, backend=None):
        for name, logger in self.logger.items():
            if backend is None or name in backend:
                logger.log(data=data, step=step)

    def finish(self):
        for logger in self.logger.values():
            if hasattr(logger, "finish"):
                logger.finish()

    def __del__(self):
        if hasattr(self, "logger"):
            self.finish()


class _JsonlAdapter:
    def __init__(self, path):
        self.path = path

    def log(self, data, step):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"step": step, **data}, default=_json_default) + "\n")


class _TensorboardAdapter:
    def __init__(self, directory=None):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(directory or "runs/tensorboard"))

    def log(self, data, step):
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if not any(name in loggers for name in Tracking.supported_backend):
            return
        directory = Path(os.environ.get("SIPO_LOG_DIR", "runs/logs"))
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "validation_samples.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, "samples": samples}, default=_json_default) + "\n")


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")

def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x

def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans

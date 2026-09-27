"""Portable entry point for SIPO training, checkpoint resume, and evaluation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def local_path(value):
    return str(Path(value).expanduser().resolve())


def build_command(args):
    # Each override is a separate argv item; no shell evaluation is used.
    output = Path(local_path(args.output))
    values = {
        "actor_rollout_ref.model.path": args.model,
        "data.train_files": local_path(args.train_data),
        "data.val_files": local_path(args.val_data),
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.experiment_name": args.name,
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.nnodes": 1,
        "trainer.total_training_steps": args.steps,
        "trainer.test_freq": args.eval_every,
        "trainer.save_freq": args.save_every,
        "release_log_dir": str(output / "logs"),
        "hydra.run.dir": str(output / "hydra"),
        "custom_reward_function.path": str(ROOT / "verl/utils/reward_score/deep_research_em.py"),
        "actor_rollout_ref.rollout.tools.tool_instances.search.params.host": args.retriever_host,
        "actor_rollout_ref.rollout.tools.tool_instances.search.params.port": args.retriever_port,
    }
    if Path(args.model).expanduser().exists():
        values["actor_rollout_ref.model.path"] = local_path(args.model)
    if args.resume:
        values["trainer.resume_mode"] = "resume_path"
        values["trainer.resume_from_path"] = local_path(args.resume)
    if args.eval_only:
        values["trainer.val_only"] = True
        values["trainer.val_before_train"] = True
    if args.dump_rollouts:
        values["trainer.rollout_data_dir"] = str(output / "rollouts")
        values["trainer.validation_data_dir"] = str(output / "validation")
    command = [sys.executable, "-m", "verl.trainer.main_ppo",
               "--config-path", str(ROOT / "configs"), "--config-name", "sipo"]
    command.extend(f"{key}={json.dumps(value)}" for key, value in values.items())
    command.extend(args.override)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory or public model identifier")
    parser.add_argument("--train-data", required=True, help="Prepared training Parquet (also required for evaluation setup)")
    parser.add_argument("--val-data", required=True, help="Prepared validation Parquet")
    parser.add_argument("--output", default="runs/sipo")
    parser.add_argument("--name", default="sipo")
    parser.add_argument("--gpus", type=positive_int, default=8)
    parser.add_argument("--steps", type=positive_int, default=240)
    parser.add_argument("--eval-every", type=positive_int, default=20)
    parser.add_argument("--save-every", type=positive_int, default=40)
    parser.add_argument("--retriever-host", default="localhost")
    parser.add_argument("--retriever-port", type=positive_int, default=8000)
    parser.add_argument("--resume", help="Checkpoint directory named global_step_N")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dump-rollouts", action="store_true")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra key=value override; repeat as needed")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without loading models or starting training")
    args = parser.parse_args(argv)
    if args.retriever_port > 65535:
        parser.error("retriever port must be at most 65535")
    for value in args.override:
        if "=" not in value or value.startswith("--"):
            parser.error("each override must be a Hydra key=value expression")
    if not args.dry_run:
        for name in ("train_data", "val_data"):
            if not Path(getattr(args, name)).expanduser().is_file():
                parser.error(f"{name.replace('_', '-')} must be an existing file")
        if args.resume:
            checkpoint = Path(args.resume).expanduser()
            if not checkpoint.is_dir() or not checkpoint.name.startswith("global_step_") or not checkpoint.name[12:].isdigit():
                parser.error("resume must name an existing global_step_N directory")
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    output = Path(local_path(args.output))
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["SIPO_LOG_DIR"] = str(output / "logs")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("VLLM_USE_V1", "0")
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    # Stream to the console and a local run log, preserving the trainer exit code.
    with (output / "run.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return 130


if __name__ == "__main__":
    sys.exit(main())

"""CPU checks for release entry points, config, search, logging, and data prep."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_launcher_paths_and_evaluation_config(self):
        from hydra import initialize_config_dir, compose
        from omegaconf import OmegaConf
        train = load_file("sipo_train", "scripts/train.py")
        with tempfile.TemporaryDirectory(prefix="sipo test ") as temporary:
            args = SimpleNamespace(output=temporary, model="models/qwen", train_data="data/train.parquet",
                val_data="data/test.parquet", name="test", gpus=8, steps=240, eval_every=20,
                save_every=40, retriever_host="localhost", retriever_port=8000,
                resume=str(Path(temporary) / "global_step_40"), eval_only=True,
                dump_rollouts=False, override=[])
            command = train.build_command(args)
            with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
                cfg = compose(config_name="sipo", overrides=command[7:])
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            self.assertTrue(cfg.trainer.val_only)
            self.assertTrue(cfg.trainer.val_before_train)
            self.assertEqual(cfg.trainer.resume_from_path, args.resume)
            self.assertEqual(cfg.release_log_dir, str(Path(temporary) / "logs"))
            rollout = cfg.actor_rollout_ref.rollout
            self.assertEqual(rollout.n, rollout.initial_rollouts +
                             rollout.expansion_iterations * rollout.beam_size * rollout.num_branches)
            self.assertEqual(rollout.postsel_correction_weight, 1)
            self.assertFalse(rollout.val_kwargs.do_sample)
            # Resolve and instantiate exactly the tool requested by the config.
            path, name = rollout.tools.tool_instances.search.class_path.rsplit(".", 1)
            module = __import__(path, fromlist=[name])
            tool = getattr(module, name)(**rollout.tools.tool_instances.search.params)
            self.assertEqual(tool.trigger_tag, "search")

    def test_launcher_invalid_input(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"),
            "--model", "models/qwen", "--train-data", "missing", "--val-data", "missing"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("existing file", result.stderr)

    def test_search_request_and_formatting(self):
        from verl.workers.rollout.tools.search_tool import LocalSearchTool
        connection = MagicMock()
        connection.getresponse.return_value.status = 200
        connection.getresponse.return_value.read.return_value = json.dumps({"result": [[
            {"document": {"contents": " First passage "}},
            {"document": {"contents": "First passage"}},
            {"document": {"contents": "Second passage"}}]]}).encode()
        with patch("http.client.HTTPConnection", return_value=connection) as factory:
            result = LocalSearchTool(max_results=2, result_length=6).execute('"question"', timeout=9)
        self.assertEqual(result, "Page 1: First \nPage 2: Second")
        factory.assert_called_once_with("localhost", 8000, timeout=9)
        args, kwargs = connection.request.call_args
        self.assertEqual(args, ("POST", "/retrieve"))
        self.assertEqual(json.loads(kwargs["body"]), {"queries": ["question"], "topk": 3, "return_scores": True})
        connection.close.assert_called_once()

    def test_search_empty_failure_and_timeout(self):
        from verl.workers.rollout.tools.search_tool import LocalSearchTool
        tool = LocalSearchTool()
        self.assertEqual(tool._extract_and_format_results({"result": [[]]}), "No search results found.")
        for status, body, failure in [(503, b"", None), (200, b"not-json", None),
                                      (200, b"", TimeoutError())]:
            connection = MagicMock()
            connection.getresponse.return_value.status = status
            connection.getresponse.return_value.read.return_value = body
            connection.request.side_effect = failure
            with self.subTest(status=status, failure=failure), patch("http.client.HTTPConnection", return_value=connection):
                self.assertEqual(tool.execute("query"), "")
            connection.close.assert_called_once()

    def test_local_tracking_and_samples(self):
        from verl.utils.tracking import Tracking, ValidationGenerationsLogger
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            tracking = Tracking("sipo", "test", ["jsonl"], {"release_log_dir": temporary})
            tracking.log({"validation/acc": .5}, 20)
            ValidationGenerationsLogger().log(["jsonl"], [("question", "answer", 1)], 20)
            row = json.loads((Path(temporary) / "metrics.jsonl").read_text())
            self.assertEqual(row, {"step": 20, "validation/acc": .5})
            row = json.loads((Path(temporary) / "validation_samples.jsonl").read_text())
            self.assertEqual(row["samples"][0], ["question", "answer", 1])

    def test_retriever_response_protocol(self):
        server = load_file("sipo_retriever", "rag_server/retrieval_server.py")
        self.assertIsNone(server.retriever)  # Importing does not load a model or index.
        server.config = SimpleNamespace(retrieval_topk=3)
        server.retriever = MagicMock()
        doc = {"contents": "A synthetic passage"}
        server.retriever.batch_search.return_value = ([[doc]], [[.75]])
        response = server.retrieve_endpoint(server.QueryRequest(queries=["query"], return_scores=True))
        self.assertEqual(response, {"result": [[{"document": doc, "score": .75}]]})
        server.retriever.batch_search.return_value = [[doc]]
        response = server.retrieve_endpoint(server.QueryRequest(queries=["query"]))
        self.assertEqual(response, {"result": [[doc]]})

    def test_dataset_preparation(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            raw = temporary / "raw"
            for dataset in ("hotpotqa", "nq"):
                (raw / dataset).mkdir(parents=True)
                for split in ("train", "dev", "test"):
                    rows = [{"id": "synthetic", "question": "What is two plus two", "golden_answers": ["4"]}]
                    (raw / dataset / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            for script, source in [("hotpotqa_multihop_train.py", "hotpotqa"), ("nq_singlehop_train.py", "nq"),
                                   ("multihop_test_merge.py", "hotpotqa"), ("singlehop_test_merge.py", "nq")]:
                output = temporary / script.removesuffix(".py")
                command = [sys.executable, str(ROOT / "data_process" / script),
                           "--input-dir", str(raw), "--local_dir", str(output)]
                if "merge" in script:
                    command += ["--data_sources", source]
                env = dict(os.environ, HF_DATASETS_CACHE=str(temporary / "cache"), HF_DATASETS_OFFLINE="1")
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                frame = pd.read_parquet(output / "test.parquet")
                self.assertEqual(len(frame), 1)
                self.assertEqual(frame.iloc[0]["data_source"], source)
                self.assertEqual(list(frame.iloc[0]["reward_model"]["ground_truth"]["target"]), ["4"])
                prompt = frame.iloc[0]["prompt"][0]["content"]
                self.assertIn("What is two plus two?", prompt)
                self.assertIn("<search>", prompt)


if __name__ == "__main__":
    unittest.main()

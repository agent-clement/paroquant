from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


def _install_dependency_stubs():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class FakeTensor:
            def __init__(self, data):
                self.data = data

            def numel(self):
                if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
                    return sum(len(row) for row in self.data)
                if isinstance(self.data, list):
                    return len(self.data)
                return 1

            def squeeze(self, dim=0):
                if isinstance(self.data, list) and len(self.data) == 1 and isinstance(self.data[0], list):
                    return FakeTensor(self.data[0])
                return self

            @property
            def shape(self):
                if isinstance(self.data, list):
                    return (len(self.data),)
                return (1,)

            def __getitem__(self, item):
                return FakeTensor(self.data[item])

        def tensor(data):
            return FakeTensor(data)

        def cat(tensors, dim=0):
            if dim != 1:
                raise NotImplementedError(dim)
            combined = []
            for tensor in tensors:
                combined.extend(tensor.data[0])
            return FakeTensor([combined])

        def no_grad():
            def decorator(fn):
                return fn

            return decorator

        torch.Tensor = FakeTensor
        torch.tensor = tensor
        torch.cat = cat
        torch.no_grad = no_grad
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.device = lambda name: name
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)

        nn = types.ModuleType("torch.nn")
        nn.Module = type("Module", (), {})
        nn.ModuleList = list
        nn.Linear = type("Linear", (), {})
        torch.nn = nn

        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = nn

    if "datasets" not in sys.modules:
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: None
        sys.modules["datasets"] = datasets

    if "tqdm" not in sys.modules:
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = tqdm

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
        transformers.AutoTokenizer = type("AutoTokenizer", (), {})
        sys.modules["transformers"] = transformers


_install_dependency_stubs()

from paroquant.optim import util


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shuffle_calls = 0

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed: int):
        self.shuffle_calls += 1
        return self

    def __iter__(self):
        return iter(self.rows)


class CalibDatasetLoadingTest(unittest.TestCase):
    def test_get_calib_dataset_does_not_reshuffle_loaded_dataset(self):
        dataset = FakeDataset([{"text": "ab"}, {"text": "cd"}, {"text": "ef"}])
        spec = util.CalibDatasetSpec(kind="hf", dataset_id="org/dataset", split="train")

        def fake_load_dataset_from_spec(spec_arg, *, split: str, seed: int):
            self.assertEqual(spec_arg, spec)
            self.assertEqual(split, "train")
            self.assertEqual(seed, 7)
            return dataset

        with patch.object(util, "_load_dataset_from_spec", side_effect=fake_load_dataset_from_spec):
            samples = util.get_calib_dataset(
                data=spec,
                tokenizer=FakeTokenizer(),
                n_samples=2,
                block_size=2,
                seed=7,
                split="validation",
            )

        self.assertEqual(dataset.shuffle_calls, 0)
        self.assertEqual(len(samples), 2)

    def test_load_dataset_from_spec_shuffles_hf_datasets_once(self):
        dataset = FakeDataset([{"text": "ab"}])
        spec = util.CalibDatasetSpec(kind="hf", dataset_id="org/dataset", split="train")
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return dataset

        with patch.object(util, "load_dataset", side_effect=fake_load_dataset):
            loaded = util._load_dataset_from_spec(spec, split="train", seed=11)

        self.assertIs(loaded, dataset)
        self.assertEqual(dataset.shuffle_calls, 1)
        self.assertEqual(calls, [((spec.dataset_id,), {"split": "train", "trust_remote_code": False})])

    def test_load_dataset_from_spec_shuffles_redpajama_before_split(self):
        class FakeRedPajamaDataset(FakeDataset):
            def __init__(self):
                super().__init__([{"text": str(i)} for i in range(10)])
                self.selected_ranges = []

            def select(self, rng):
                values = list(rng)
                self.selected_ranges.append(values)
                selected_rows = [self.rows[i] for i in values]
                selected = FakeDataset(selected_rows)
                selected.source = self
                return selected

        dataset = FakeRedPajamaDataset()
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return dataset

        with patch.object(util, "load_dataset", side_effect=fake_load_dataset):
            spec = util.CalibDatasetSpec(kind="builtin", dataset_id="redpajama", split="validation")
            loaded = util._load_dataset_from_spec(spec, split="validation", seed=3)

        self.assertEqual(dataset.shuffle_calls, 1)
        self.assertEqual(dataset.selected_ranges, [[7]])
        self.assertIsInstance(loaded, FakeDataset)
        self.assertEqual(
            calls,
            [
                (
                    ("liang2kl/RedPajama-Data-1T-Sample-Backup",),
                    {"split": "train", "trust_remote_code": True},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()

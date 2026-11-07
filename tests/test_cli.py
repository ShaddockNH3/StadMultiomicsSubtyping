"""CLI 与包入口的单测。不 import numpy / pandas，所以没有科学计算依赖也能跑。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stad_subtyping  # noqa: E402
from stad_subtyping.cli import build_parser  # noqa: E402


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_pipeline_requires_dirs(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["pipeline"])

    def test_pipeline_defaults_seed(self):
        args = self.parser.parse_args(["pipeline", "--data-dir", "d", "--out-dir", "o"])
        self.assertEqual(args.seed, 2026)
        self.assertEqual(args.command, "pipeline")

    def test_pipeline_accepts_custom_seed(self):
        args = self.parser.parse_args(["pipeline", "--data-dir", "d", "--out-dir", "o", "--seed", "7"])
        self.assertEqual(args.seed, 7)

    def test_classifier_optional_lists(self):
        args = self.parser.parse_args(["classifier"])
        self.assertEqual(args.command, "classifier")
        self.assertIsNone(args.classifiers)
        self.assertIsNone(args.queues)

    def test_classifier_collects_names(self):
        args = self.parser.parse_args(["classifier", "--classifiers", "mlp", "lstm"])
        self.assertEqual(args.classifiers, ["mlp", "lstm"])

    def test_unknown_command_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["nope"])


class LazyExportTest(unittest.TestCase):
    def test_package_does_not_pull_numpy(self):
        self.assertNotIn("numpy", sys.modules)

    def test_public_name_is_listed(self):
        self.assertIn("run_full_pipeline", stad_subtyping.__all__)
        self.assertIn("run_full_pipeline", dir(stad_subtyping))

    def test_unknown_attribute_raises(self):
        with self.assertRaises(AttributeError):
            getattr(stad_subtyping, "not_a_real_name")


if __name__ == "__main__":
    unittest.main()

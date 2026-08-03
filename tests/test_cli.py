"""Command line argument parsing tests.

These exist because a broken CI invocation shipped once: ``--output`` was defined
on the top-level parser while the workflow passed it *after* the subcommand, so
``cartpole bench --output artifacts --check`` died with argparse exit code 2
while every unit test still passed. The commands documented in the README and
used by CI are now asserted here.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from cartpole.cli import (
    MAX_LONG_RUN_DRIFT_M,
    MIN_COSINE_SIMILARITY,
    _learning_regressions,
    build_parser,
    command_all,
    command_animate,
    command_bench,
    command_learn,
    command_robustness,
)

CI_INVOCATION = ["bench", "--output", "artifacts", "--check"]


class TestArgumentParsing(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_ci_invocation_parses(self):
        """The exact argument list used by .github/workflows/ci.yml."""
        args = self.parser.parse_args(CI_INVOCATION)

        self.assertEqual(args.output, "artifacts")
        self.assertTrue(args.check)
        self.assertIs(args.handler, command_bench)

    def test_every_subcommand_accepts_output_after_the_subcommand(self):
        for command, handler in [
            ("bench", command_bench),
            ("robustness", command_robustness),
            ("animate", command_animate),
            ("learn", command_learn),
            ("all", command_all),
        ]:
            with self.subTest(command=command):
                args = self.parser.parse_args([command, "--output", "somewhere"])
                self.assertEqual(args.output, "somewhere")
                self.assertIs(args.handler, handler)

    def test_readme_quick_start_commands_parse(self):
        for argv in (["bench"], ["robustness"], ["animate"], ["all"]):
            with self.subTest(argv=argv):
                self.assertTrue(self.parser.parse_args(argv).output)

    def test_optional_flags_parse(self):
        self.assertEqual(self.parser.parse_args(["robustness", "--grid", "5"]).grid, 5)
        self.assertEqual(self.parser.parse_args(["animate", "--fps", "12"]).fps, 12)
        self.assertTrue(self.parser.parse_args(["all", "--check"]).check)
        self.assertTrue(self.parser.parse_args(["learn", "--check"]).check)
        self.assertTrue(self.parser.parse_args(["learn", "--quick"]).quick)
        self.assertEqual(self.parser.parse_args(["learn", "--seed", "7"]).seed, 7)

    def test_output_defaults_to_the_results_directory(self):
        self.assertTrue(self.parser.parse_args(["bench"]).output.endswith("results"))

    def test_missing_subcommand_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_unknown_subcommand_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["nonsense"])


class TestLearningRegressionGate(unittest.TestCase):
    """`learn --check` is only worth running in CI if it can actually go red."""

    @staticmethod
    def _result(**overrides):
        defaults = dict(
            algorithm="ARS",
            comparison=SimpleNamespace(cosine_similarity=0.99, is_stabilising=True),
            metrics=[object()] * 4,
            scenarios_passed=4,
            long_run_drift_m=0.0,
        )
        return SimpleNamespace(**{**defaults, **overrides})

    def test_a_healthy_result_reports_nothing(self):
        self.assertEqual(_learning_regressions([self._result()]), [])

    def test_a_low_cosine_similarity_is_flagged(self):
        result = self._result(
            comparison=SimpleNamespace(cosine_similarity=MIN_COSINE_SIMILARITY - 0.01, is_stabilising=True)
        )
        self.assertIn("cosine", _learning_regressions([result])[0])

    def test_a_non_stabilising_gain_is_flagged(self):
        result = self._result(comparison=SimpleNamespace(cosine_similarity=0.99, is_stabilising=False))
        self.assertIn("not stabilising", _learning_regressions([result])[0])

    def test_a_failed_scenario_is_flagged(self):
        self.assertIn("3/4 scenarios", _learning_regressions([self._result(scenarios_passed=3)])[0])

    def test_long_run_drift_is_flagged(self):
        """The horizon study showed a policy can hold the pole and still walk away."""
        result = self._result(long_run_drift_m=MAX_LONG_RUN_DRIFT_M + 1.0)
        self.assertIn("drift", _learning_regressions([result])[0])

    def test_a_missing_comparison_is_flagged(self):
        self.assertIn("no gain comparison", _learning_regressions([self._result(comparison=None)])[0])

    def test_every_algorithm_is_reported_not_just_the_first(self):
        failures = _learning_regressions(
            [self._result(algorithm="ARS", scenarios_passed=0), self._result(algorithm="CEM", scenarios_passed=1)]
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(any(line.startswith("CEM") for line in failures))


class TestWorkflowStaysInSyncWithTheCli(unittest.TestCase):
    def test_workflow_commands_match_the_parser(self):
        """Parse the command out of the workflow file and check the CLI accepts it."""
        from pathlib import Path

        workflow = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
        lines = [line.strip() for line in workflow.read_text(encoding="utf-8").splitlines()]
        commands = [
            line.split("run:", 1)[1].strip()
            for line in lines
            if line.startswith("run:") and "cartpole.cli" in line
        ]

        self.assertTrue(commands, "no cartpole.cli invocation found in the workflow")
        for command in commands:
            with self.subTest(command=command):
                argv = command.split()[3:]  # drop: python -m cartpole.cli
                build_parser().parse_args(argv)


if __name__ == "__main__":
    unittest.main()

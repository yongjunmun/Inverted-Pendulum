"""Command line argument parsing tests.

These exist because a broken CI invocation shipped once: ``--output`` was defined
on the top-level parser while the workflow passed it *after* the subcommand, so
``cartpole bench --output artifacts --check`` died with argparse exit code 2
while every unit test still passed. The commands documented in the README and
used by CI are now asserted here.
"""

from __future__ import annotations

import unittest

from cartpole.cli import (
    build_parser,
    command_all,
    command_animate,
    command_bench,
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

    def test_output_defaults_to_the_results_directory(self):
        self.assertTrue(self.parser.parse_args(["bench"]).output.endswith("results"))

    def test_missing_subcommand_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_unknown_subcommand_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["nonsense"])


class TestWorkflowStaysInSyncWithTheCli(unittest.TestCase):
    def test_workflow_benchmark_command_matches_the_parser(self):
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

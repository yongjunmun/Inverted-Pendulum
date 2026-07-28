"""Tests for the learning-based controllers.

These check the *properties* that make the comparison against classical control
meaningful, not just that training runs without crashing.
"""

from __future__ import annotations

import unittest

import numpy as np

from cartpole.controllers.learned import LearnedController
from cartpole.controllers.lqr import LQRController
from cartpole.dynamics import CartPoleParams
from cartpole.learning.analysis import compare_to_lqr
from cartpole.learning.policies import LinearPolicy, MLPPolicy
from cartpole.learning.rollout import TrainingConfig, evaluate, rollout, sample_initial_state
from cartpole.learning.trainers import train_ars, train_cem
from cartpole.metrics import evaluate as score_run
from cartpole.simulate import SimConfig, simulate

PARAMS = CartPoleParams()
FAST = TrainingConfig(episode_seconds=3.0)


class TestPolicies(unittest.TestCase):
    def test_linear_policy_round_trips_parameters(self):
        policy = LinearPolicy()
        values = np.array([1.0, -2.0, 3.0, -4.0])
        policy.set_params(values)
        np.testing.assert_allclose(policy.get_params(), values)

    def test_output_respects_the_force_limit(self):
        policy = LinearPolicy(force_limit=10.0)
        policy.set_params(np.full(4, 1e6))
        self.assertLessEqual(abs(policy.act(np.ones(4))), 10.0 + 1e-9)

    def test_linear_policy_reproduces_the_lqr_law(self):
        """Loading ``-K`` must make the policy behave exactly like the LQR."""
        lqr = LQRController(PARAMS)
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        policy.set_params(-lqr.gain[0])

        for state in (np.array([0.1, 0.0, 0.05, 0.0]), np.array([-0.2, 0.3, -0.02, 0.1])):
            expected = np.clip(lqr.compute(state, 0.0, 0.0), -PARAMS.force_limit, PARAMS.force_limit)
            self.assertAlmostEqual(policy.act(state), float(expected), places=9)

    def test_mlp_parameter_count_matches_its_layers(self):
        mlp = MLPPolicy(n_observations=4, hidden_sizes=(16, 16))
        self.assertEqual(mlp.n_params, 4 * 16 + 16 + 16 * 16 + 16 + 16 * 1 + 1)

    def test_mlp_is_deterministic(self):
        mlp = MLPPolicy(hidden_sizes=(8,))
        mlp.set_params(np.random.default_rng(0).normal(size=mlp.n_params))
        observation = np.array([0.1, -0.2, 0.05, 0.3])
        self.assertEqual(mlp.act(observation), mlp.act(observation))


class TestReward(unittest.TestCase):
    def test_crashing_early_is_not_rewarded(self):
        """The failure penalty must make a crash worse than surviving.

        Every reward is negative, so without the penalty an agent that ends the
        episode early accumulates less negative reward and scores *better*.
        """
        good = LinearPolicy(force_limit=PARAMS.force_limit)
        good.set_params(-LQRController(PARAMS).gain[0])

        # Positive feedback on cart position: whichever way the cart starts, it
        # is driven further that way until it leaves the rail.
        crasher = LinearPolicy(force_limit=PARAMS.force_limit)
        crasher.set_params(np.array([1e6, 0.0, 0.0, 0.0]))

        rng = np.random.default_rng(0)
        start = np.array([0.1, 0.0, 0.0, 0.0])
        good_return, good_steps = rollout(good, PARAMS, FAST, rng, start)
        bad_return, bad_steps = rollout(crasher, PARAMS, FAST, rng, start)

        self.assertLess(bad_steps, good_steps, "the crasher should end its episode early")
        self.assertLess(bad_return, good_return, "an early crash must not score better")

    def test_reward_is_zero_at_the_equilibrium(self):
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        total, steps = rollout(policy, PARAMS, FAST, np.random.default_rng(0), np.zeros(4))
        self.assertEqual(steps, FAST.n_steps)
        self.assertAlmostEqual(total, 0.0, places=9)

    def test_initial_states_stay_inside_the_configured_ranges(self):
        config = TrainingConfig(initial_angle=0.2, initial_position=1.5)
        rng = np.random.default_rng(0)
        for _ in range(50):
            state = sample_initial_state(config, rng)
            self.assertLessEqual(abs(state[0]), 1.5)
            self.assertLessEqual(abs(state[2]), 0.2)


class TestTrainers(unittest.TestCase):
    def test_ars_recovers_a_stabilising_gain(self):
        policy, history = train_ars(
            LinearPolicy(force_limit=PARAMS.force_limit), PARAMS, FAST, iterations=40, seed=0
        )
        report = compare_to_lqr(policy, PARAMS)

        self.assertTrue(report.is_stabilising, "ARS should find a stabilising gain")
        self.assertGreater(report.cosine_similarity, 0.9, "learned gain should align with the LQR gain")
        self.assertGreater(history.total_env_steps, 0)

    def test_cem_improves_on_the_untrained_policy(self):
        rng = np.random.default_rng(3)
        untrained = evaluate(LinearPolicy(force_limit=PARAMS.force_limit), PARAMS, FAST, rng, episodes=5)
        policy, _ = train_cem(
            LinearPolicy(force_limit=PARAMS.force_limit), PARAMS, FAST, iterations=12, population=20, seed=0
        )
        self.assertGreater(evaluate(policy, PARAMS, FAST, rng, episodes=5), untrained)

    def test_training_is_reproducible_from_a_seed(self):
        first, _ = train_ars(LinearPolicy(force_limit=PARAMS.force_limit), PARAMS, FAST, iterations=8, seed=7)
        second, _ = train_ars(LinearPolicy(force_limit=PARAMS.force_limit), PARAMS, FAST, iterations=8, seed=7)
        np.testing.assert_allclose(first.get_params(), second.get_params())

    def test_history_records_monotonically_increasing_sample_cost(self):
        _, history = train_ars(LinearPolicy(force_limit=PARAMS.force_limit), PARAMS, FAST, iterations=6, seed=0)
        self.assertEqual(history.env_steps, sorted(history.env_steps))
        self.assertEqual(len(history.iterations), 6)


class TestGainComparison(unittest.TestCase):
    def test_the_lqr_gain_compares_to_itself_perfectly(self):
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        policy.set_params(-LQRController(PARAMS).gain[0])
        report = compare_to_lqr(policy, PARAMS)

        self.assertAlmostEqual(report.cosine_similarity, 1.0, places=9)
        self.assertAlmostEqual(report.magnitude_ratio, 1.0, places=9)
        self.assertTrue(report.is_stabilising)

    def test_a_scaled_gain_keeps_direction_but_changes_magnitude(self):
        """Cosine similarity must be scale-invariant - that is why it is used."""
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        policy.set_params(-2.5 * LQRController(PARAMS).gain[0])
        report = compare_to_lqr(policy, PARAMS)

        self.assertAlmostEqual(report.cosine_similarity, 1.0, places=9)
        self.assertAlmostEqual(report.magnitude_ratio, 2.5, places=6)

    def test_a_sign_flipped_gain_is_detected_as_unstable(self):
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        policy.set_params(LQRController(PARAMS).gain[0])
        report = compare_to_lqr(policy, PARAMS)

        self.assertLess(report.cosine_similarity, 0.0)
        self.assertFalse(report.is_stabilising)


class TestLearnedController(unittest.TestCase):
    def test_a_learned_policy_runs_in_the_standard_benchmark(self):
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        policy.set_params(-LQRController(PARAMS).gain[0])

        result = simulate(
            LearnedController(policy, PARAMS, name="test"),
            PARAMS,
            config=SimConfig(duration=5.0),
            initial_state=np.array([0.0, 0.0, 0.2, 0.0]),
        )
        self.assertTrue(score_run(result).success)

    def test_the_controller_honours_the_reference(self):
        policy = LinearPolicy(force_limit=PARAMS.force_limit)
        policy.set_params(-LQRController(PARAMS).gain[0])
        controller = LearnedController(policy, PARAMS)

        state = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertNotEqual(controller.compute(state, 0.0, 0.0), controller.compute(state, 0.0, 1.0))
        self.assertAlmostEqual(controller.compute(state, 0.0, 1.0), 0.0, places=9)


class TestGymEnvIsOptional(unittest.TestCase):
    def test_importing_the_package_does_not_require_gymnasium(self):
        """The core must stay dependency-free; gymnasium is opt-in only."""
        import cartpole.learning as learning

        self.assertTrue(hasattr(learning, "train_ars"))

        from cartpole.learning import gym_env

        self.assertIn(gym_env.GYMNASIUM_AVAILABLE, (True, False))
        if not gym_env.GYMNASIUM_AVAILABLE:
            with self.assertRaises(ImportError):
                gym_env.CartPoleContinuousEnv()


if __name__ == "__main__":
    unittest.main()

"""Small linear-algebra toolbox: matrix exponential, discretisation, Riccati.

SciPy is deliberately *not* a dependency. Everything here is built on NumPy so
that the numerical method behind each controller is explicit and testable.
"""

from __future__ import annotations

import numpy as np


def expm(matrix: np.ndarray, terms: int = 24) -> np.ndarray:
    """Matrix exponential via scaling-and-squaring with a truncated Taylor series.

    The matrix is first scaled down by ``2**s`` so the series converges quickly,
    then the result is squared back ``s`` times.
    """
    matrix = np.asarray(matrix, dtype=float)
    norm = np.abs(matrix).sum(axis=1).max()
    squarings = int(np.ceil(np.log2(norm))) + 1 if norm > 0.5 else 0
    squarings = max(squarings, 0)

    scaled = matrix / (2.0**squarings)
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for order in range(1, terms + 1):
        term = term @ scaled / order
        result = result + term

    for _ in range(squarings):
        result = result @ result
    return result


def discretize(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation of ``(A, B)`` over ``dt``.

    Uses the block-matrix identity ``expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]``,
    which stays valid even when ``A`` is singular (the cart-pole ``A`` is, because
    cart position is a pure integrator).
    """
    n_states = state_matrix.shape[0]
    n_inputs = input_matrix.shape[1]

    block = np.zeros((n_states + n_inputs, n_states + n_inputs))
    block[:n_states, :n_states] = state_matrix
    block[:n_states, n_states:] = input_matrix

    exponential = expm(block * dt)
    return exponential[:n_states, :n_states], exponential[:n_states, n_states:]


def solve_care(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_cost: np.ndarray,
    input_cost: np.ndarray,
) -> np.ndarray:
    """Solve the continuous-time algebraic Riccati equation.

    Finds ``P`` such that ``A'P + PA - PBR^-1B'P + Q = 0`` using the stable
    invariant subspace of the Hamiltonian matrix

    ``H = [[A, -B R^-1 B'], [-Q, -A']]``.

    The eigenvectors belonging to the ``n`` eigenvalues with negative real part
    span that subspace; splitting them as ``[U1; U2]`` gives ``P = U2 U1^-1``.
    """
    state_matrix = np.asarray(state_matrix, dtype=float)
    input_matrix = np.asarray(input_matrix, dtype=float)
    state_cost = np.asarray(state_cost, dtype=float)
    input_cost = np.asarray(input_cost, dtype=float)

    n_states = state_matrix.shape[0]
    input_cost_inv = np.linalg.inv(input_cost)

    hamiltonian = np.block(
        [
            [state_matrix, -input_matrix @ input_cost_inv @ input_matrix.T],
            [-state_cost, -state_matrix.T],
        ]
    )

    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    stable = np.argsort(eigenvalues.real)[:n_states]
    if np.any(eigenvalues.real[stable] >= 0.0):
        raise np.linalg.LinAlgError("CARE has no stabilising solution for this (A, B, Q, R)")

    basis = eigenvectors[:, stable]
    upper, lower = basis[:n_states, :], basis[n_states:, :]
    riccati = np.linalg.solve(upper.T, lower.T).T

    riccati = np.real(0.5 * (riccati + riccati.conj().T))
    return riccati


def solve_dare(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_cost: np.ndarray,
    input_cost: np.ndarray,
    max_iterations: int = 5000,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Solve the discrete-time algebraic Riccati equation by value iteration.

    Iterates ``P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q`` to convergence. Used for
    the MPC terminal cost, which is what makes a short horizon behave like an
    infinite-horizon controller.
    """
    state_matrix = np.asarray(state_matrix, dtype=float)
    input_matrix = np.asarray(input_matrix, dtype=float)
    state_cost = np.asarray(state_cost, dtype=float)
    input_cost = np.asarray(input_cost, dtype=float)

    riccati = state_cost.copy()
    for _ in range(max_iterations):
        weighted = input_cost + input_matrix.T @ riccati @ input_matrix
        gain = np.linalg.solve(weighted, input_matrix.T @ riccati @ state_matrix)
        updated = state_cost + state_matrix.T @ riccati @ (state_matrix - input_matrix @ gain)
        updated = 0.5 * (updated + updated.T)
        if np.max(np.abs(updated - riccati)) < tolerance:
            return updated
        riccati = updated

    raise RuntimeError("DARE iteration did not converge")


def lqr_gain(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_cost: np.ndarray,
    input_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time LQR gain ``K`` and Riccati solution ``P`` for ``u = -Kx``."""
    riccati = solve_care(state_matrix, input_matrix, state_cost, input_cost)
    gain = np.linalg.solve(np.asarray(input_cost, dtype=float), input_matrix.T @ riccati)
    return gain, riccati


def solve_box_qp(
    hessian: np.ndarray,
    gradient_offset: np.ndarray,
    lower: float,
    upper: float,
    initial: np.ndarray | None = None,
    max_iterations: int = 120,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Minimise ``0.5 z'Hz + g'z`` subject to ``lower <= z <= upper``.

    Solved with FISTA (accelerated projected gradient). For a box-constrained QP
    the projection is just a clip, so no external QP solver is needed and the
    constraint is satisfied exactly at every iteration.
    """
    hessian = np.asarray(hessian, dtype=float)
    gradient_offset = np.asarray(gradient_offset, dtype=float)

    lipschitz = float(np.linalg.eigvalsh(hessian).max())
    step = 1.0 / max(lipschitz, 1e-12)

    current = np.zeros_like(gradient_offset) if initial is None else np.asarray(initial, dtype=float).copy()
    current = np.clip(current, lower, upper)
    momentum = current.copy()
    weight = 1.0

    for _ in range(max_iterations):
        gradient = hessian @ momentum + gradient_offset
        candidate = np.clip(momentum - step * gradient, lower, upper)
        next_weight = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * weight * weight))
        momentum = candidate + ((weight - 1.0) / next_weight) * (candidate - current)
        if np.max(np.abs(candidate - current)) < tolerance:
            return candidate
        current, weight = candidate, next_weight

    return current

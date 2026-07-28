"""Nonlinear cart-pole dynamics, energy functions and linearisation.

State vector ``s = [x, x_dot, theta, theta_dot]``

* ``x``          cart position along the rail [m]
* ``x_dot``      cart velocity [m/s]
* ``theta``      pole angle from the *upright* vertical [rad], positive
                 counter-clockwise (``theta = pi`` is hanging straight down)
* ``theta_dot``  pole angular rate [rad/s]

The pole is a uniform rigid rod, so its inertia about its own centre of mass is
``m L^2 / 12``. Viscous friction acts on the rail (``b_c``) and on the hinge
(``b_p``). The single input ``u`` is a horizontal force on the cart [N].

Euler-Lagrange equations of motion, with no small-angle approximation::

    [ M + m        m l cos(th) ] [ x_ddot  ]   [ u - b_c x_dot + m l sin(th) th_dot^2 ]
    [ m l cos(th)  J           ] [ th_ddot ] = [ m g l sin(th) - b_p th_dot           ]

where ``l = L / 2`` is the pivot-to-centre-of-mass distance and
``J = m l^2 + m L^2 / 12`` is the pole inertia about the pivot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Index of each entry in the state vector, for readable indexing.
POSITION, VELOCITY, ANGLE, RATE = 0, 1, 2, 3

STATE_LABELS = ("cart position [m]", "cart velocity [m/s]", "pole angle [rad]", "pole rate [rad/s]")


@dataclass(frozen=True)
class CartPoleParams:
    """Physical parameters of the cart-pole plant."""

    cart_mass: float = 0.5
    """Cart mass ``M`` [kg]."""
    pole_mass: float = 0.2
    """Pole mass ``m`` [kg]."""
    pole_length: float = 0.6
    """Total pole length ``L`` [m]."""
    cart_damping: float = 0.1
    """Viscous rail friction ``b_c`` [N/(m/s)]."""
    pole_damping: float = 0.005
    """Viscous hinge friction ``b_p`` [N.m/(rad/s)]."""
    force_limit: float = 10.0
    """Actuator saturation, ``|u| <= force_limit`` [N]."""
    rail_limit: float = 2.5
    """Half-length of the usable rail [m]; used for plots and failure checks."""
    gravity: float = 9.81
    """Gravitational acceleration [m/s^2]."""

    @property
    def half_length(self) -> float:
        """Pivot to pole centre of mass, ``l = L / 2`` [m]."""
        return 0.5 * self.pole_length

    @property
    def com_inertia(self) -> float:
        """Pole inertia about its own centre of mass, ``m L^2 / 12`` [kg.m^2]."""
        return self.pole_mass * self.pole_length**2 / 12.0

    @property
    def pivot_inertia(self) -> float:
        """Pole inertia about the pivot, ``J = m l^2 + m L^2 / 12`` [kg.m^2]."""
        return self.com_inertia + self.pole_mass * self.half_length**2

    def perturbed(self, mass_scale: float = 1.0, length_scale: float = 1.0) -> "CartPoleParams":
        """Return a copy with the pole mass and length scaled.

        Used by the robustness study to build a *true* plant that differs from
        the *nominal* model the controller was designed against.
        """
        return CartPoleParams(
            cart_mass=self.cart_mass,
            pole_mass=self.pole_mass * mass_scale,
            pole_length=self.pole_length * length_scale,
            cart_damping=self.cart_damping,
            pole_damping=self.pole_damping,
            force_limit=self.force_limit,
            rail_limit=self.rail_limit,
            gravity=self.gravity,
        )


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) into the half-open interval ``[-pi, pi)``.

    The boundary case matters for swing-up: a pole hanging at ``theta = pi``
    wraps to ``-pi``, so switching logic must compare on ``abs(...)``.
    """
    return -(np.pi - (np.asarray(angle) + np.pi) % (2.0 * np.pi))


def derivative(state: np.ndarray, force: float, params: CartPoleParams) -> np.ndarray:
    """Return ``ds/dt`` for the full nonlinear plant."""
    _, velocity, angle, rate = state

    mass_pole = params.pole_mass
    half_length = params.half_length
    coupling = mass_pole * half_length
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)

    total_mass = params.cart_mass + mass_pole
    inertia = params.pivot_inertia

    # Right-hand side of the 2x2 mass-matrix system.
    rhs_cart = force - params.cart_damping * velocity + coupling * sin_angle * rate**2
    rhs_pole = mass_pole * params.gravity * half_length * sin_angle - params.pole_damping * rate

    determinant = total_mass * inertia - (coupling * cos_angle) ** 2
    cart_accel = (inertia * rhs_cart - coupling * cos_angle * rhs_pole) / determinant
    pole_accel = (total_mass * rhs_pole - coupling * cos_angle * rhs_cart) / determinant

    return np.array([velocity, cart_accel, rate, pole_accel])


def rk4_step(state: np.ndarray, force: float, dt: float, params: CartPoleParams) -> np.ndarray:
    """Advance the plant one step with classical fourth-order Runge-Kutta."""
    k1 = derivative(state, force, params)
    k2 = derivative(state + 0.5 * dt * k1, force, params)
    k3 = derivative(state + 0.5 * dt * k2, force, params)
    k4 = derivative(state + dt * k3, force, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def pole_energy(state: np.ndarray, params: CartPoleParams) -> float:
    """Rotational plus potential energy of the pole about the pivot [J]."""
    angle, rate = state[ANGLE], state[RATE]
    kinetic = 0.5 * params.pivot_inertia * rate**2
    potential = params.pole_mass * params.gravity * params.half_length * np.cos(angle)
    return float(kinetic + potential)


def target_energy(params: CartPoleParams) -> float:
    """Pole energy at the upright equilibrium, ``m g l`` [J]."""
    return float(params.pole_mass * params.gravity * params.half_length)


def total_energy(state: np.ndarray, params: CartPoleParams) -> float:
    """Total mechanical energy of cart plus pole [J].

    Conserved when there is no damping and no applied force, which is what the
    integrator accuracy test checks.
    """
    velocity, angle, rate = state[VELOCITY], state[ANGLE], state[RATE]
    half_length = params.half_length

    cart_kinetic = 0.5 * params.cart_mass * velocity**2
    com_velocity_x = velocity + half_length * np.cos(angle) * rate
    com_velocity_y = -half_length * np.sin(angle) * rate
    pole_kinetic = 0.5 * params.pole_mass * (com_velocity_x**2 + com_velocity_y**2)
    pole_rotational = 0.5 * params.com_inertia * rate**2
    potential = params.pole_mass * params.gravity * half_length * np.cos(angle)

    return float(cart_kinetic + pole_kinetic + pole_rotational + potential)


def linearize_upright(params: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """Analytic Jacobians ``(A, B)`` of the plant about ``theta = 0, u = 0``.

    Derived by hand from the equations of motion with ``sin(th) -> th``,
    ``cos(th) -> 1`` and ``th_dot^2 -> 0``. :func:`numeric_jacobian` provides an
    independent finite-difference check (see ``tests/test_dynamics.py``).
    """
    total_mass = params.cart_mass + params.pole_mass
    inertia = params.pivot_inertia
    coupling = params.pole_mass * params.half_length
    gravity_torque = params.pole_mass * params.gravity * params.half_length
    determinant = total_mass * inertia - coupling**2

    state_matrix = np.zeros((4, 4))
    state_matrix[POSITION, VELOCITY] = 1.0
    state_matrix[ANGLE, RATE] = 1.0

    state_matrix[VELOCITY, VELOCITY] = -inertia * params.cart_damping / determinant
    state_matrix[VELOCITY, ANGLE] = -coupling * gravity_torque / determinant
    state_matrix[VELOCITY, RATE] = coupling * params.pole_damping / determinant

    state_matrix[RATE, VELOCITY] = coupling * params.cart_damping / determinant
    state_matrix[RATE, ANGLE] = total_mass * gravity_torque / determinant
    state_matrix[RATE, RATE] = -total_mass * params.pole_damping / determinant

    input_matrix = np.zeros((4, 1))
    input_matrix[VELOCITY, 0] = inertia / determinant
    input_matrix[RATE, 0] = -coupling / determinant

    return state_matrix, input_matrix


def numeric_jacobian(
    state: np.ndarray,
    force: float,
    params: CartPoleParams,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians ``(A, B)`` of the nonlinear plant."""
    state = np.asarray(state, dtype=float)
    state_matrix = np.zeros((4, 4))

    for column in range(4):
        step = np.zeros(4)
        step[column] = epsilon
        forward = derivative(state + step, force, params)
        backward = derivative(state - step, force, params)
        state_matrix[:, column] = (forward - backward) / (2.0 * epsilon)

    forward = derivative(state, force + epsilon, params)
    backward = derivative(state, force - epsilon, params)
    input_matrix = ((forward - backward) / (2.0 * epsilon)).reshape(4, 1)

    return state_matrix, input_matrix

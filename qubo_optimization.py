from typing import Dict, Tuple, List, Optional

try:
    # Qiskit Optimization
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    # Qiskit Algorithms v2
    from qiskit_algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA

    # qBraid Provider
    from qbraid import QbraidProvider
    from qiskit.primitives import Sampler

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def build_qubo_problem(num_vars, linear_coeffs, quadratic_coeffs, constant):
    qubo = QuadraticProgram()
    for i in range(num_vars):
        qubo.binary_var(name=f"x{i}")
    qubo.minimize(linear=linear_coeffs, quadratic=quadratic_coeffs, constant=constant)
    return qubo


def solve_with_qaoa(problem: QuadraticProgram, reps: int = 1, maxiter: int = 100):
    """Solve a QUBO using QAOA on qBraid simulator (Qiskit v2 + qBraid Provider)."""
        # Conectarse al provider de qBraid
    service = QbraidProvider()
    backend = service.get_device("qbraid_qir_simulator")  # nombre correcto del backend

        # Crear sampler compatible con qBraid
    try:
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()
    except ImportError:
        sampler = Sampler()

    qaoa = QAOA(
        reps=reps,
        optimizer=COBYLA(maxiter=maxiter),
        sampler=sampler,
    )

    eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = eigen_optimizer.solve(problem)

    return result
"""
QUBO Template using Qiskit
==========================

This module provides a generic template for formulating and solving
Quadratic Unconstrained Binary Optimization (QUBO) problems using
IBM's Qiskit Optimization module.  It contains helper functions to
construct a QUBO from user‑supplied linear and quadratic coefficients,
convert the resulting problem into an Ising Hamiltonian suitable for
quantum algorithms, and solve the problem either with a classical
exact eigensolver or with the Quantum Approximate Optimization
Algorithm (QAOA).  Extensive comments are included to guide users on
how to adapt the template to their specific problem instances.

Because this file may be imported in environments where Qiskit is not
available, all imports are wrapped in a try/except block.  When
Qiskit is missing, the public functions will raise descriptive
exceptions rather than failing silently.  A module level flag
``HAS_QISKIT`` indicates whether the underlying libraries have been
successfully imported.

Functions
---------

* ``build_qubo_problem`` – create a ``QuadraticProgram`` from a set of
  linear and quadratic coefficients.
* ``problem_to_ising`` – convert a QUBO to an Ising Hamiltonian and
  constant offset.
* ``solve_classically`` – solve the QUBO using a classical exact
  eigensolver via Qiskit's ``NumPyMinimumEigensolver``.
* ``solve_with_qaoa`` – solve the QUBO using QAOA with user‑selectable
  circuit depth.
* ``print_solution`` – utility to print solution vectors in a
  human‑readable format.

Usage
-----

The typical workflow for using this template is:

1. Define the number of binary decision variables in your problem.
2. Specify a dictionary of linear coefficients where the keys are the
   zero‑based indices of variables and the values are the linear
   coefficients.  Positive coefficients penalise selecting the
   corresponding variable, while negative coefficients reward it.
3. Specify a dictionary of quadratic coefficients where the keys are
   two‑tuples ``(i, j)`` representing interactions between variables
   ``i`` and ``j`` (with ``i < j``) and the values are the coupling
   strengths.  Positive couplings penalise simultaneous activation of
   the two variables, while negative couplings reward it.
4. Optionally provide a constant term that will be added to the
   objective.  The constant does not affect the optimisation but may
   be useful for energy comparisons.
5. Call ``build_qubo_problem`` to obtain a ``QuadraticProgram``.
6. Use ``problem_to_ising`` to convert the QUBO into an Ising
   Hamiltonian if you intend to run a quantum algorithm directly.
7. Choose either ``solve_classically`` or ``solve_with_qaoa``
   depending on whether you want an exact solution or a variational
   quantum solution.  Each solver returns a result object whose
   ``x`` attribute contains the optimal assignment of binary
   variables.

Examples of how to use these functions can be found in the test
module ``test_qubo_template.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional

# Try importing Qiskit and its optimisation modules.  If these imports
# fail, set HAS_QISKIT to False so that callers can detect the
# availability of quantum functionality.
try:
    # Qiskit Optimization provides the QuadraticProgram class used
    # throughout this module.  Importing from qiskit_optimization as
    # opposed to qiskit directly isolates the optimisation toolkit.
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToIsing
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit.algorithms import NumPyMinimumEigensolver, QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit import Aer
    HAS_QISKIT = True
except ImportError:
    # If importing fails, set the flag to False.  The functions below
    # will raise an ImportError when called.
    HAS_QISKIT = False

def _ensure_qiskit() -> None:
    """Raise an ImportError if Qiskit is unavailable.

    This helper function simplifies error handling in the public API.
    When Qiskit is not installed in the environment, calling any
    solver or conversion function will cause an exception with a
    descriptive message.
    """
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit and qiskit_optimization are required to use this "
            "function. Please install qiskit and qiskit-optimization "
            "modules to enable quantum optimisation capabilities."
        )


def build_qubo_problem(
    num_vars: int,
    linear_coeffs: Dict[int, float],
    quadratic_coeffs: Dict[Tuple[int, int], float],
    constant: float = 0.0,
    minimize: bool = True,
) -> QuadraticProgram:
    """Create a Qiskit ``QuadraticProgram`` for a QUBO problem.

    Parameters
    ----------
    num_vars : int
        The total number of binary decision variables in the problem.
    linear_coeffs : Dict[int, float]
        A dictionary mapping variable indices to linear coefficients.
        Each variable index ``i`` (0 ≤ i < num_vars) should be an
        integer.  The corresponding value is the coefficient of ``x_i``
        in the objective function.  Coefficients with positive values
        penalise selection of the variable, while negative values
        reward it.
    quadratic_coeffs : Dict[Tuple[int, int], float]
        A dictionary mapping pairs of variable indices to quadratic
        coefficients.  Each key must be a tuple ``(i, j)`` with
        ``0 ≤ i < j < num_vars``.  The value is the coefficient of
        the term ``x_i * x_j`` in the objective.  Positive coefficients
        penalise simultaneous activation of both variables, whereas
        negative coefficients reward their joint selection.
    constant : float, optional
        A constant offset to add to the objective function.  This
        constant does not change the optimisation result but may be
        useful when comparing energies between different problems.
        Default is 0.0.
    minimize : bool, optional
        Whether to minimise (True) or maximise (False) the objective.
        Qiskit supports both minimisation and maximisation.  Default
        is True (minimisation).

    Returns
    -------
    QuadraticProgram
        A fully defined Qiskit ``QuadraticProgram`` representing the
        provided QUBO.

    Raises
    ------
    ImportError
        If Qiskit is not available in the environment.
    ValueError
        If any index in ``linear_coeffs`` or ``quadratic_coeffs``
        exceeds ``num_vars`` or if a quadratic coefficient has an
        invalid index order.

    Notes
    -----
    This function builds the quadratic objective in a way that is
    compatible with Qiskit 0.45+.  For earlier versions, you may need
    to adapt the API slightly (e.g., using ``problem.linear_coef`` or
    ``problem.quadratic_coefficients`` directly).
    """
    _ensure_qiskit()

    # Create a new QuadraticProgram instance.  Each variable will be
    # binary, representing a decision that can take value 0 or 1.
    problem = QuadraticProgram()

    # Add binary variables named x0, x1, ..., x{num_vars-1}.  The
    # ``binary_var`` method returns a Variable object, but we discard it
    # here since we access variables by index later.
    for idx in range(num_vars):
        problem.binary_var(name=f"x{idx}")

    # Verify that all linear indices are within range.
    for idx in linear_coeffs:
        if idx < 0 or idx >= num_vars:
            raise ValueError(
                f"linear_coeffs index {idx} is out of bounds for "
                f"num_vars={num_vars}"
            )

    # Verify that all quadratic indices are within range and ordered.
    for (i, j) in quadratic_coeffs:
        if not (0 <= i < j < num_vars):
            raise ValueError(
                f"quadratic_coeffs key {(i, j)} must satisfy 0 ≤ i < j < num_vars"
            )

    # Prepare the linear and quadratic coefficient vectors for Qiskit.  In
    # Qiskit, the ``minimize`` method accepts a list/array of linear
    # coefficients and a 2D array or dict for quadratic coefficients.
    linear_array = [0.0] * num_vars
    for idx, coeff in linear_coeffs.items():
        linear_array[idx] = coeff

    # Quadratic coefficients can be supplied as a dictionary of
    # {(i, j): coeff} pairs.  Missing entries are assumed to be zero.
    quadratic_dict = {(i, j): coeff for (i, j), coeff in quadratic_coeffs.items()}

    # Add the objective to the problem.  Qiskit allows either
    # ``problem.minimize`` or ``problem.maximize`` with a constant,
    # linear and quadratic part.  The constant is added via the
    # ``constant`` parameter.
    if minimize:
        problem.minimize(constant=constant, linear=linear_array, quadratic=quadratic_dict)
    else:
        problem.maximize(constant=constant, linear=linear_array, quadratic=quadratic_dict)

    return problem


def problem_to_ising(problem: QuadraticProgram):
    """Convert a QUBO ``QuadraticProgram`` to an Ising Hamiltonian.

    The resulting Hamiltonian is represented as a ``PauliSumOp`` (a
    weighted sum of Pauli operators) along with a constant offset.  This
    form can be used directly with Qiskit quantum algorithms such as
    ``QAOA`` or ``VQE``.

    Parameters
    ----------
    problem : QuadraticProgram
        The QUBO problem to convert.  It must not contain any
        additional constraints; only binary variables and the
        quadratic objective are supported.

    Returns
    -------
    Tuple[PauliSumOp, float]
        A tuple containing the Pauli operator representing the
        Hamiltonian and the constant energy offset.

    Raises
    ------
    ImportError
        If Qiskit is not available.
    """
    _ensure_qiskit()

    # Use Qiskit's converter to map the QuadraticProgram to an Ising
    # Hamiltonian.  The converter handles the mapping from binary
    # variables to Pauli operators.  The constant offset is returned
    # separately and can be added back to expectation values if
    # necessary.
    converter = QuadraticProgramToIsing()
    op, offset = converter.convert(problem)
    return op, offset


def solve_classically(problem: QuadraticProgram):
    """Solve a QUBO using a classical exact eigensolver.

    This function wraps Qiskit's ``NumPyMinimumEigensolver`` in a
    ``MinimumEigenOptimizer`` to find the optimal assignment of the
    binary decision variables.  It is suitable for small to moderate
    sized problems where an exact solution is desired.

    Parameters
    ----------
    problem : QuadraticProgram
        A QUBO problem constructed with ``build_qubo_problem``.

    Returns
    -------
    qiskit_optimization.algorithms.optimization_result.OptimizationResult
        An object containing the optimal bitstring in its ``x``
        attribute and the optimal objective value in ``fval``.  See
        Qiskit's documentation for details on the result structure.

    Raises
    ------
    ImportError
        If Qiskit is not available.
    """
    _ensure_qiskit()

    # Instantiate the classical eigensolver.  NumPyMinimumEigensolver
    # computes the exact smallest eigenvalue and corresponding
    # eigenstate of the Hamiltonian.  When passed to
    # MinimumEigenOptimizer, it solves the QUBO exactly.
    exact_eigensolver = NumPyMinimumEigensolver()

    # Wrap the eigensolver in the optimisation wrapper.  The wrapper
    # translates between optimisation problems and eigensolver results.
    optimizer = MinimumEigenOptimizer(exact_eigensolver)

    # Solve the problem.  The result contains the bitstring solution
    # accessible via ``result.x`` and the objective value via
    # ``result.fval``.
    result = optimizer.solve(problem)
    return result


def solve_with_qaoa(
    problem: QuadraticProgram,
    reps: int = 1,
    optimizer: Optional[object] = None,
    quantum_instance: Optional[object] = None,
):
    """Solve a QUBO using the Quantum Approximate Optimisation Algorithm.

    Parameters
    ----------
    problem : QuadraticProgram
        The optimisation problem to solve.
    reps : int, optional
        The number of QAOA layers (also called ``p``).  More layers
        generally improve approximation quality at the expense of
        deeper circuits.  Default is 1.
    optimizer : object, optional
        Classical optimizer to tune QAOA parameters.  If None, the
        default COBYLA optimizer from Qiskit is used.  The optimizer
        should implement the SciPy interface ``minimize(fun, x0)``.
    quantum_instance : object, optional
        A backend or ``QuantumInstance`` describing where to run the
        quantum circuits.  If None, the default is an Aer statevector
        simulator with shots disabled.  Using a shot‑based simulator
        (i.e., qasm simulator) may provide more realistic sampling
        behaviour.

    Returns
    -------
    qiskit_optimization.algorithms.optimization_result.OptimizationResult
        The optimisation result.  The returned ``x`` attribute is the
        bitstring representing the schedule found by QAOA.

    Raises
    ------
    ImportError
        If Qiskit is not available.
    """
    _ensure_qiskit()

    # Use the user‑provided classical optimiser or fall back to
    # COBYLA.  The COBYLA optimiser is derivative‑free and performs
    # reasonably well for small parameter spaces.
    classical_optimizer = optimizer if optimizer is not None else COBYLA()

    # Select the quantum backend.  If no backend is provided, use
    # Qiskit's Aer simulator in statevector mode to avoid sampling
    # noise.  Note: the statevector simulator does not support
    # measurement noise and returns exact expectation values.
    if quantum_instance is None:
        quantum_instance = Aer.get_backend("aer_simulator_statevector")

    # Build the QAOA algorithm object.  The ``reps`` argument sets the
    # depth (number of alternating operator applications).  The
    # ``quantum_instance`` will handle circuit execution.
    qaoa = QAOA(
        reps=reps,
        optimizer=classical_optimizer,
        quantum_instance=quantum_instance,
    )

    # Wrap QAOA in a MinimumEigenOptimizer so that it can be used as a
    # solver for QuadraticProgram instances.  The optimisation wrapper
    # translates the cost Hamiltonian into circuits, runs them, and
    # interprets the results as an optimisation solution.
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)

    # Solve the problem.  The output is the optimisation result.
    result = qaoa_optimizer.solve(problem)
    return result


def print_solution(result) -> None:
    """Pretty print an optimisation result.

    This convenience function prints the decision vector and its
    objective value in a clear, readable format.  It assumes that the
    provided result object has attributes ``x`` (a sequence of bits)
    and ``fval`` (the corresponding objective value).

    Parameters
    ----------
    result : qiskit_optimization.algorithms.optimization_result.OptimizationResult
        The result object returned by a solver.

    Returns
    -------
    None
    """
    if result is None:
        print("No result to display.")
        return
    bitstring = result.x
    objective = result.fval
    print("Solution bitstring:", bitstring)
    print("Objective value:", objective)
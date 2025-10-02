"""
Test suite for ``qubo_template``
===============================

This module exercises the functions provided in ``qubo_template``.  It
starts by compiling the template to ensure there are no syntax errors.
Then, if Qiskit is available, it constructs a simple QUBO instance,
solves it exactly with a classical solver, and optionally runs QAOA
for demonstration.  If Qiskit is not installed, the tests are skipped
with informative output.
"""

import sys
import py_compile

import qubo_template

def compile_template() -> None:
    """Compile the template to check for syntax errors.

    This test ensures that the ``qubo_template.py`` file is free of
    syntax errors.  Python's built‑in ``py_compile`` module raises
    ``py_compile.PyCompileError`` if any syntax errors are found.
    """
    try:
        py_compile.compile("qubo_template.py", doraise=True)
        print("Template compiled successfully.")
    except py_compile.PyCompileError as exc:
        print("Template failed to compile:", exc)
        sys.exit(1)


def test_small_qubo() -> None:
    """Solve a small QUBO if Qiskit is available.

    This function constructs a simple QUBO problem with two binary
    variables and solves it both classically and with QAOA.  The
    objective is chosen to demonstrate linear and quadratic terms.  The
    problem solved is:

        minimise  -x0 - x1 + 2 x0 x1

    which penalises both variables being 1 while rewarding each being 1
    individually.  The optimum is to pick either x0=1, x1=0 or x0=0,
    x1=1.  The classical solution and QAOA solution should agree.
    """
    if not qubo_template.HAS_QISKIT:
        print("Qiskit is not installed; skipping QUBO solving test.")
        return

    # Build a small QUBO with two variables.  Linear coefficients
    # reward turning on either variable (hence negative), but the
    # quadratic term penalises turning on both (positive).  The
    # constant term is zero.
    num_vars = 2
    linear = {0: -1.0, 1: -1.0}
    quadratic = {(0, 1): 2.0}
    problem = qubo_template.build_qubo_problem(
        num_vars=num_vars, linear_coeffs=linear, quadratic_coeffs=quadratic, constant=0.0
    )

    # Solve the QUBO exactly using the classical solver
    classical_result = qubo_template.solve_classically(problem)
    print("Classical solver result:")
    qubo_template.print_solution(classical_result)

    # Solve the QUBO with QAOA using one layer.  In practice you may
    # increase the number of repetitions for better approximation.  We
    # use the statevector simulator here for deterministic results.
    qaoa_result = qubo_template.solve_with_qaoa(problem, reps=1)
    print("QAOA solver result:")
    qubo_template.print_solution(qaoa_result)


if __name__ == "__main__":
    # Run the tests in sequence
    compile_template()
    test_small_qubo()
# ===========================
# CUÁNTICO (modificado)
# ===========================
import numpy as np

# --- Qiskit (Aer SamplerV2 si existe; si no, Sampler clásico) ---
try:
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    _SAMPLER_V2 = True
except Exception:
    from qiskit_aer.primitives import Sampler as AerSampler
    _SAMPLER_V2 = False

from qiskit import QuantumCircuit

# ===========================
# 1) Mix different QUBO Matrixies between all objective Function
# ===========================
def get_blending_QUBO_weights(weights):

    weights = np.array(weights)
    normalize_weights = weights / np.sum(weights)

    return normalize_weights  # sum=1

def build_qubo_matrices(number_of_productors, 
                        number_of_substations, 
                        substation_of_productors, 
                        energy_productions, 
                        is_green_energy, 
                        cost_of_productors, 
                        co2_emmisions_by_productors,
                        energy_demand):
    """
    Devuelve Qs, qs, consts para f1..f5.
    Cambios:
      - f1 SOLO balance global (sin capacidad por subcentral).
      - f5 mantiene anti-concentración por subcentral.
    """

    reserve_frac = 0.10
    energy_production_target = energy_demand * (1 + reserve_frac)

    QUBO_quadratic_matrixes = {k: np.zeros((number_of_productors, number_of_productors), dtype=np.float64) for k in ['f1','f2','f3','f4','f5']}
    QUBO_linear_vectors = {k: np.zeros(number_of_productors) for k in ['f1','f2','f3','f4','f5']}
    QUBO_constans = {k: 0.0 for k in ['f1','f2','f3','f4','f5']}
    
    QUBO_constans['f1'] += (energy_production_target**2)

    # f1: Balance global (D_target - Σ g_hat y)^2
    QUBO_quadratic_matrixes['f1'] += np.outer(energy_productions, energy_productions)
    QUBO_linear_vectors['f1'] += -2.0 * energy_production_target * energy_productions

    # f2: Renovables (max)
    QUBO_linear_vectors['f2'] += (is_green_energy * energy_productions)

    # f3: Costos (min) → lineal negativo
    QUBO_linear_vectors['f3'] += -(cost_of_productors * energy_productions)

    # f4: Emisiones (min) → lineal negativo
    QUBO_linear_vectors['f4'] += -(co2_emmisions_by_productors * energy_productions)

    # f5: Uniformidad → lineal positivo
    distribution_of_productors = np.zeros(substation_of_productors.shape, dtype=float)
    for s in range(number_of_substations):
        same_substation_mask = (substation_of_productors == s)
        different_substation_mask = (substation_of_productors != s)
        distribution_of_productors[same_substation_mask] = np.sum(different_substation_mask)

    QUBO_linear_vectors['f5'] += distribution_of_productors

    return QUBO_quadratic_matrixes, QUBO_linear_vectors, QUBO_constans

def combine_objectives_with_blending_weights(QUBO_quadratic_matrixes, QUBO_linear_vectors, QUBO_constans, blending_weights):
    """Combina f1..f5 con pesos c (sum(c)=1) -> Q, q, const."""
    features = ['f1','f2','f3','f4','f5']
    QUBO_blended_matrix = np.zeros_like(QUBO_quadratic_matrixes['f1'])
    QUBO_blended_vector = np.zeros_like(QUBO_linear_vectors['f1'])
    QUBO_blended_constant = 0.0
    
    for wi, k in zip(blending_weights, features):
        QUBO_blended_matrix += wi * QUBO_quadratic_matrixes[k]
        QUBO_blended_vector += wi * QUBO_linear_vectors[k]
        QUBO_blended_constant += wi * QUBO_constans[k]
    
    #Q = (Q + Q.T) / 2.0
    return QUBO_blended_matrix, QUBO_blended_vector, QUBO_blended_constant

# ===========================
# 2) Transform QUBO to Ising
# ===========================
def qubo_to_ising(QUBO_quadratic_matrix, QUBO_linear_vector, QUBO_constant):
    """Convierte QUBO (x∈{0,1}) a Ising (z∈{-1,1})."""
    
    # Matriz de interacción Ising J (solo términos cuadráticos)
    J_ising = QUBO_quadratic_matrix / 4
    
    # Campo local Ising h
    h_ising = QUBO_linear_vector / 2 + np.sum(QUBO_quadratic_matrix, axis=1) / 4
    
    # Constante Ising
    c_ising = QUBO_constant + np.sum(QUBO_linear_vector) / 2 + np.sum(QUBO_quadratic_matrix) / 4
    
    return J_ising, h_ising, c_ising
    
# ===========================
# 3) Solve by QAOA
# ===========================
def get_transferred_angles(p=3, seed=0):
    """Random Angles for the Cirquit."""
    rng = np.random.default_rng(seed)
    gammas = (rng.random(p) * 0.6 + 0.2).tolist()
    betas  = (rng.random(p) * 0.6 + 0.2).tolist()
    return gammas, betas

def build_qaoa_circuit(J_ising, h_ising, num_layers, gammas, betas):
    """
    Build standar qaoa cirquit to solve Ising
    """
    number_of_qubits = len(h_ising)
    qc = QuantumCircuit(number_of_qubits)
    for i in range(number_of_qubits):
        qc.h(i)
    for layer in range(num_layers):
        gamma, beta = gammas[layer], betas[layer]
        
        # Z
        for i in range(number_of_qubits):
            if abs(h_ising[i]) > 1e-12:
                qc.rz(2.0 * gamma * h_ising[i], i)

        # ZZ
        for i in range(number_of_qubits):
            for j in range(i+1, number_of_qubits):
                if abs(J_ising[i, j]) > 1e-12:
                    qc.cx(i, j)
                    qc.rz(4.0 * gamma * J_ising[i, j], j)
                    qc.cx(i, j)
        # Mixer
        for i in range(number_of_qubits):
            qc.rx(2.0 * beta, i)
    
    qc.measure_all()
    return qc

# ===========================
# 4) Evaluate sample solutions by each objective
# ===========================
def evaluate_objectives(solution_candidate,
                        energy_productions,
                        is_green_energy,
                        cost_of_productors,
                        co2_emmisions_by_productors,
                        substation_of_productors,
                        energy_production_target):
    """
    f1: fiabilidad (max) = -(10*deficit)   [SIN sobrecarga por subcentral]
    f2: renovables (max) = Σ ren*g_hat*y
    f3: -costo (max)     = -Σ cost*g_hat*y
    f4: -CO2 (max)       = -Σ co2*g_hat*y
    f5: -var(b_s) (max)  = -Var(# productores por subcentral)
    """
    solution_candidates_array = np.array(solution_candidate, dtype=int)

    f1 = get_f1_value(solution_candidates_array,
                      energy_productions,
                      is_green_energy,
                      cost_of_productors,
                      co2_emmisions_by_productors,
                      substation_of_productors,
                      energy_production_target)
    
    f2 = get_f2_value(solution_candidates_array,
                      energy_productions,
                      is_green_energy,
                      cost_of_productors,
                      co2_emmisions_by_productors,
                      substation_of_productors,
                      energy_production_target)
    
    f3 = get_f3_value(solution_candidates_array,
                      energy_productions,
                      is_green_energy,
                      cost_of_productors,
                      co2_emmisions_by_productors,
                      substation_of_productors,
                      energy_production_target)
    
    f4 = get_f4_value(solution_candidates_array,
                      energy_productions,
                      is_green_energy,
                      cost_of_productors,
                      co2_emmisions_by_productors,
                      substation_of_productors,
                      energy_production_target)
    
    f5 = get_f5_value(solution_candidates_array,
                      energy_productions,
                      is_green_energy,
                      cost_of_productors,
                      co2_emmisions_by_productors,
                      substation_of_productors,
                      energy_production_target)
    
    f6 = get_f1_bis_value(solution_candidates_array,
                          energy_productions,
                          is_green_energy,
                          cost_of_productors,
                          co2_emmisions_by_productors,
                          substation_of_productors,
                          energy_production_target)
    
    return np.array([f1, f2, f3, f4, f5, f6], dtype=float)

def get_f1_value(solution_candidates_array,
                 energy_productions,
                 is_green_energy,
                 cost_of_productors,
                 co2_emmisions_by_productors,
                 substation_of_productors,
                 energy_production_target):
    
    energy_production = float((energy_productions * solution_candidates_array).sum())
    deficit = max(0, energy_production_target - energy_production)

    return - (10.0 * deficit)

def get_f1_bis_value(solution_candidates_array,
                    energy_productions,
                    is_green_energy,
                    cost_of_productors,
                    co2_emmisions_by_productors,
                    substation_of_productors,
                    energy_production_target):
    
    energy_production = float((energy_productions * solution_candidates_array).sum())
    deficit = (energy_production_target - energy_production)**2

    return - (10.0 * deficit)

def get_f2_value(solution_candidates_array,
                 energy_productions,
                 is_green_energy,
                 cost_of_productors,
                 co2_emmisions_by_productors,
                 substation_of_productors,
                 energy_production_target):
    
    return float((is_green_energy * energy_productions * solution_candidates_array).sum())

def get_f3_value(solution_candidates_array,
                 energy_productions,
                 is_green_energy,
                 cost_of_productors,
                 co2_emmisions_by_productors,
                 substation_of_productors,
                 energy_production_target):
    
    return - float((cost_of_productors * energy_productions * solution_candidates_array).sum())

def get_f4_value(solution_candidates_array,
                 energy_productions,
                 is_green_energy,
                 cost_of_productors,
                 co2_emmisions_by_productors,
                 substation_of_productors,
                 energy_production_target):
    
    return - float((co2_emmisions_by_productors  * energy_productions * solution_candidates_array).sum())

def get_f5_value(solution_candidates_array,
                 energy_productions,
                 is_green_energy,
                 cost_of_productors,
                 co2_emmisions_by_productors,
                 substation_of_productors,
                 energy_production_target):
    
    ammount_of_substations = int(substation_of_productors.max()) + 1
    distribution_of_productors = np.zeros(solution_candidates_array.shape, dtype=float)
    
    for s in range(ammount_of_substations):
        same_substation_mask = (substation_of_productors == s)
        different_substation_mask = (substation_of_productors != s)
        distribution_of_productors[same_substation_mask] = np.sum(different_substation_mask)


    return float(np.sum(distribution_of_productors * solution_candidates_array))

def filter_candidates_by_pareto_domination(objective_values, candidate_choices, eps=1e-6):
    """Only use the results that are not dominated by another solution"""
    keep = []

    for i in range(objective_values.shape[0]):
        dominated = False
        for j in range(objective_values.shape[0]):
            if i == j:
                continue
            if np.all(objective_values[j] >= objective_values[i] - eps) and np.any(objective_values[j] > objective_values[i] + eps):
                dominated = True
                break
        if not dominated:
            keep.append(i)

    return objective_values[keep], candidate_choices[keep]

def remove_duplicate_solutions(counts, n):
    """Solo bitstrings únicos; no expandir multiplicidades."""
    Ys_unique = []
    for outcome in counts.keys():
        if isinstance(outcome, (int, np.integer)):
            bits = format(outcome, f"0{n}b")
        else:
            bits = outcome
        y = np.fromiter(bits[::-1], dtype=np.int8)  # little-endian
        Ys_unique.append(y)
    return Ys_unique

# ===========================
# 6) Sampling (Aer) (The Quantum Part)
# ===========================
def run_sampler_and_get_candidate_solutions(sampler, qc, shots):
    # We should change this to actually run it in a quantum computing
    res = sampler.run([qc], shots=shots).result()
    if _SAMPLER_V2:
        counts = res[0].data.meas.get_counts()
    else:
        counts = res.quasi_dists[0]
    return counts

def sample_optimal_choices(number_of_samples, number_of_choices, qc):
    
    sampler = AerSampler()
    counts = run_sampler_and_get_candidate_solutions(sampler, qc, number_of_samples)
    Ys = remove_duplicate_solutions(counts, n=number_of_choices)
    return np.array(Ys)

def run_qaoa_sampling(QUBO_quadratic_matrixes, QUBO_linear_vectors,
                      QUBO_constans, weights, energy_demand,
                    substation_of_productors, 
                        energy_productions, 
                        is_green_energy, 
                        cost_of_productors, 
                        co2_emmisions_by_productors,
                      number_of_solutions=2000, number_angles=3, seed_angles=0,
                      rng=np.random.default_rng(1),fiabilidad_minima=-1e-3):
    """
    f1 permite leve negativo controlado por fiab_min (p.ej. -0.5).
    """

    reserve_frac = 0.10
    energy_production_target = energy_demand * (1 + reserve_frac)

    gammas, betas = get_transferred_angles(p=number_angles, seed=seed_angles)
    
    Pareto_F, Pareto_Y = [], []
    
    # (a) pesos c -> QUBO
    blending_weights = get_blending_QUBO_weights(weights)
    Q, q, cst = combine_objectives_with_blending_weights(QUBO_quadratic_matrixes, 
                                                             QUBO_linear_vectors,
                                                             QUBO_constans, 
                                                             blending_weights)
    
    # (b) Ising
    J, h, _ = qubo_to_ising(Q, q, cst)

    # (c) QAOA + medir
    qc = build_qaoa_circuit(J, h, number_angles, gammas, betas)

    number_of_choices = len(h)

    optimal_choices = sample_optimal_choices(number_of_solutions, number_of_choices, qc)

    # (d) evaluar y filtrar por fiabilidad relajada
    valuation_by_choice = np.array([evaluate_objectives(choice, 
                                        energy_productions, 
                                        is_green_energy, 
                                        cost_of_productors, 
                                        co2_emmisions_by_productors,
                                        substation_of_productors, 
                                        energy_production_target) for choice in optimal_choices])
    
    valid_choices = valuation_by_choice[:, 0] >= fiabilidad_minima
    if not np.any(valid_choices):
        print("No Existe Solucion Posible")
        return None, None
    
    valuation_by_choice = valuation_by_choice[valid_choices]
    optimal_choices = optimal_choices[valid_choices]
    
    Pareto_F, Pareto_Y = filter_candidates_by_pareto_domination(valuation_by_choice, optimal_choices, eps=1e-6)
    s = Pareto_F.sum(axis=1)
    top = np.argsort(s)[-1]
    Pareto_F = Pareto_F[top,:]
    Pareto_Y = Pareto_Y[top,:]

    return Pareto_F, Pareto_Y

# ===========================
# 7) Main Function
# ===========================
def get_best_control_signal(
    number_of_productors, 
    number_of_substations, 
    substation_of_productors, 
    energy_productions, 
    is_green_energy, 
    cost_of_productors, 
    co2_emmisions_by_productors,
    energy_demand,
    weights):

    QUBO_quadratic_matrixes, QUBO_linear_vectors, QUBO_constans = build_qubo_matrices(
        number_of_productors, 
        number_of_substations, 
        substation_of_productors, 
        energy_productions, 
        is_green_energy, 
        cost_of_productors, 
        co2_emmisions_by_productors,
        energy_demand
    )

    Pareto_F, Pareto_Y = run_qaoa_sampling(
        QUBO_quadratic_matrixes, QUBO_linear_vectors, QUBO_constans,
        weights, energy_demand,
                    substation_of_productors, 
                        energy_productions, 
                        is_green_energy, 
                        cost_of_productors, 
                        co2_emmisions_by_productors,
        number_of_solutions=1,
        number_angles=3,
        seed_angles=0,
        fiabilidad_minima=-1e-3
    )

    return Pareto_F, Pareto_Y



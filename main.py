from data_preprocessing import load_stations_from_csv, load_features_from_csv
from qubo_translator import translate_qubo
from qubo_optimization import build_qubo_problem, solve_with_qaoa
from  moo_solver import get_best_control_signal
import time

print("Loading Data")
features_of_grid = load_features_from_csv("datasets/simplified_dataset.csv", -1)
print("Data Loaded")
print()

number_of_productors = features_of_grid["number_of_productors"]
number_of_substations = features_of_grid["number_of_substations"]
substation_of_productors = features_of_grid["substation_of_productors"]
energy_productions = features_of_grid["energy_productions"]
is_green_energy = features_of_grid["is_green_energy"]
cost_of_productors = features_of_grid["cost_of_productors"]
co2_emmisions_by_productors = features_of_grid["co2_emmisions_by_productors"]
energy_demand = features_of_grid["energy_demand"]

"""
W1: fiabilidad 
W2: renovables 
w3: costo 
w4: CO2 
w5: uniformidad

"""

WEIGHTS = [100, 0, 0, 0, 15]

start_time = time.perf_counter()
print("Getting Best Control Signal...")
value, signal = get_best_control_signal(number_of_productors,
                                        number_of_substations, 
                                        substation_of_productors,
                                        energy_productions, 
                                        is_green_energy, 
                                        cost_of_productors, 
                                        co2_emmisions_by_productors, 
                                        energy_demand,
                                        WEIGHTS)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
print(value, signal)

# station_1 = [1, -10, -3, -8, -5,27]
# station_2 = [0, -2, -3,26,28]

# stations = [station_1, station_2]

# constant, linear_coefficients, quadratic_coefficients = translate_qubo(stations, 3)

# number_of_variables = len(linear_coefficients)

# print(number_of_variables)
# print(constant)
# print(linear_coefficients)
# print(quadratic_coefficients)

# qubo_problem = build_qubo_problem(number_of_variables, linear_coefficients, quadratic_coefficients, constant)
# solution = solve_with_qaoa(qubo_problem, reps=10, maxiter=200)
# print(solution)
from data_preprocessing import load_features_from_csv
from moo_solver import get_best_control_signal
import time

print("Loading Electrical Grid Features")
features_of_grid = load_features_from_csv("datasets/simplified_dataset.csv", -1)
print("Features Loaded")
print("="*80)

number_of_productors = features_of_grid["number_of_productors"]
number_of_substations = features_of_grid["number_of_substations"]
substation_of_productors = features_of_grid["substation_of_productors"]
energy_productions = features_of_grid["energy_productions"]
is_green_energy = features_of_grid["is_green_energy"]
cost_of_productors = features_of_grid["cost_of_productors"]
co2_emmisions_by_productors = features_of_grid["co2_emmisions_by_productors"]
energy_demand = features_of_grid["energy_demand"]

print("Number Of Productors: ", number_of_productors)
print("Number Of SubStations: ", number_of_substations)
print("Current Energy Demand: ", energy_demand)

weights_desc = ["Disbalance", "Green Energy", "Energy Cost", "Co2 Emmision", "Uniformity"]
WEIGHTS = [100, 0, 0, 0, 15]

print("="*80)
print("WEIGHTS")
for i in range(len(weights_desc)):
    print(weights_desc[i] + ": " + str(WEIGHTS[i]))

start_time = time.perf_counter()

print()
print("Computing Best Control Signal using MOO...")
value_of_objectives, control_signal = get_best_control_signal(number_of_productors,
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
print()
print("Best Control Signal: \n", control_signal)
print()
print("Values Of Each Objective Function")
print("="*80)

for i in range(len(weights_desc)):
    print(weights_desc[i] + ": " + str(value_of_objectives[i]))

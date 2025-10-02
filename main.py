from data_preprocessing import load_stations_from_csv
from qubo_translator import translate_qubo
import time

print("Loading Data")
stations = load_stations_from_csv("datasets/prosumer_dataset.csv", -1)
print("Data Loaded")
print()
print("Amount of SubStations: ", len(stations))
print()

start_time = time.perf_counter()
print("Getting Coefficients...")
constant, linear_coefficients, quadratic_coefficients = translate_qubo(stations, 0)

print("Total Consuption: ", constant)
print("Linear Coefficients for Qubo: \n", linear_coefficients)
print("quadratic Coefficients for Qubo: \n", quadratic_coefficients)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
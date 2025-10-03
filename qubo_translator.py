from typing import Dict, Tuple, List, Optional
import numpy as np

def is_productor(microgenerator: float):
    return microgenerator > 0

def filter_productors(microgenerators:List[float]):
    return [microgenerator for microgenerator in microgenerators if is_productor(microgenerator)]

def filter_consumers(microgenerators:List[float]):
    return [microgenerator for microgenerator in microgenerators if not is_productor(microgenerator)]

def get_total_consumption(consumers_by_stations: List[List[float]]):
    return sum(sum(consumers) for consumers in consumers_by_stations)

def get_total_ammount_of_productors(productors_by_stations: List[List[float]]):
    
    ammount = 0
    for station in productors_by_stations:
         ammount += len(station)

    return ammount

def get_distribution_value_of_productor(productors_by_stations: List[List[float]], station: List[float], house:float):
    amount_of_productors = get_total_ammount_of_productors(productors_by_stations) 
    return amount_of_productors - len(station)

def get_productors_coefficients(productors_by_stations: List[List[float]]):
    productors_list = []
    productors_distribution = []
    
    for station in productors_by_stations:
        for productor in station:
            productors_list.append(productor)
            productors_distribution.append(get_distribution_value_of_productor(productors_by_stations, station, productor))
    
    return productors_list, productors_distribution

def get_linear_coefficients(total_consumption, productors_array, distribution_array, lambda_arg):

    rest = {}
    for i in range(len(productors_array)):
        rest[i] = 2 * total_consumption * productors_array[i] - lambda_arg * distribution_array[i]

    return rest

def get_quadratic_coefficients(total_consumption, productors_array, distribution_array, lambda_arg):
    return  {
                (i, j): 2 * productors_array[i] * productors_array[j]
                for i in range(len(productors_array))
                for j in range(i, len(productors_array))
            }

def translate_qubo(stations: List[List[float]], lambda_arg: float):
    
    productors_by_stations = list(map(filter_productors, stations))
    consumers_by_stations = list(map(filter_consumers, stations))

    total_consumption = get_total_consumption(consumers_by_stations)
    productors_list, productors_distribution = get_productors_coefficients(productors_by_stations)
    productors_array = np.array(productors_list)
    distribution_array = np.array(productors_distribution)

    linear_coefficients = get_linear_coefficients(total_consumption, productors_array, distribution_array, lambda_arg)
    quadratic_coefficients = get_quadratic_coefficients(total_consumption, productors_array, distribution_array, lambda_arg)

    return total_consumption, linear_coefficients, quadratic_coefficients



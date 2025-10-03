import pandas as pd
from typing import Dict, Tuple, List, Optional
import numpy as np

def get_energy_value_of_prosumer(prosumer):
    consumption = float(prosumer["unsatisfied_consumption_kWh"])
    generation = float(prosumer["excess_energy_kWh"])
    if  consumption > 0:
        return -consumption
    else:
        return generation

def load_stations_from_csv(csv_path = "data.csv", truncate_at=-1) -> List[List[float]]:

    df = pd.read_csv(csv_path, header=0, encoding="utf-8-sig", low_memory=False)

    stations = {}

    if truncate_at > 0:
        df = df.head(truncate_at)

    for _, row in df.iterrows():
        
        prosumer = row.to_dict()
        sub_station_id = prosumer['substation_id']
        
        if sub_station_id not in stations:
            stations[sub_station_id] = []
        
        station = stations[sub_station_id]

        station.append(get_energy_value_of_prosumer(prosumer))


    station_list = []
    for key in stations:
        station_list.append(stations[key])

    return station_list

def load_features_from_csv(csv_path = "data.csv", truncate_at=-1) -> List[List[float]]:

    df = pd.read_csv(csv_path, header=0, encoding="utf-8-sig", low_memory=False)

    number_of_productors = 0 
    number_of_substations = 0
    seen_substations = [] 
    substation_of_productors = [] 
    energy_productions = []
    is_green_energy = []
    cost_of_productors = [] 
    co2_emmisions_by_productors = []
    energy_demand = 0

    if truncate_at > 0:
        df = df.head(truncate_at)

    for _, row in df.iterrows():
        prosumer = row.to_dict()
        sub_station_id = prosumer['substation_id']
        
        if sub_station_id not in seen_substations:
            seen_substations.append(sub_station_id)
        

        energy_production = get_energy_value_of_prosumer(prosumer)
        
        if energy_production <=0:
            energy_demand += energy_production

        else:
            number_of_productors += 1
            substation_of_productors.append(sub_station_id)
            energy_productions.append(energy_production)
            is_green_energy.append(0)
            cost_of_productors.append(0)
            co2_emmisions_by_productors.append(0)
        
        # Para limitar el uso de mejoria y qubits en el simulador local
        if number_of_productors > 15:
            break

    number_of_substations = len(seen_substations)

    return {
            "number_of_productors" : number_of_productors,
            "number_of_substations" : number_of_substations,
            "substation_of_productors" : np.array(substation_of_productors),
            "energy_productions" : np.array(energy_productions),
            "is_green_energy" : np.array(is_green_energy),
            "cost_of_productors" : np.array(cost_of_productors),
            "co2_emmisions_by_productors" : np.array(co2_emmisions_by_productors),
            "energy_demand" : energy_demand
            }

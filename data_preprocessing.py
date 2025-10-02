import pandas as pd
from typing import Dict, Tuple, List, Optional

def get_energy_value_of_prosumer(prosumer):
    consumption = float(prosumer["consumption_kWh"])
    generation = float(prosumer["generation_kWh"])
    return generation - consumption 

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

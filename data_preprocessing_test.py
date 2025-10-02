from data_preprocessing import load_stations_from_csv

stations = load_stations_from_csv("datasets/prosumer_dataset.csv", -1)
print(len(stations))

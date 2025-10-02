from qubo_translator import translate_qubo

station_1 = [1, -2, -3, 4, -5, 8]
station_2 = [0, -2, 10]

stations = [station_1, station_2]

constant, linear_coefficients, quadratic_coefficients = translate_qubo(stations, 0)

print(constant)

print(linear_coefficients)

print(quadratic_coefficients)



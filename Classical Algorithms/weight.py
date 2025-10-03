def weight(node):
    if float(node.get('unsatisfied_consumption_kWh', "0")) > 0:
        return -float(node.get('unsatisfied_consumption_kWh', "0"))
    else:
        return float(node.get('excess_energy_kWh', "0"))

def weight_node(node):
    if float(node[1].get('unsatisfied_consumption_kWh', "0")) > 0:
        return -float(node[1].get('unsatisfied_consumption_kWh', "0"))
    else:
        return float(node[1].get('excess_energy_kWh', "0"))
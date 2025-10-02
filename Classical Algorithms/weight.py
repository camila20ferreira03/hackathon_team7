def weight(node):
    return node[1].get('generation_kWh', 0) - node[1].get('consumption_kWh', 0)
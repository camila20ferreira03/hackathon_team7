def weight(node):
    #print(node)
    return float(node.get('generation_kWh', "0")) - float(node.get('consumption_kWh', "0"))

def weight_node(node):
    return float(node[1].get('generation_kWh', "0")) - float(node[1].get('consumption_kWh', "0"))
from flask import Flask, render_template, request, jsonify
import networkx as nx
import json
#from simulation import Simulation, state_transition_SIR, state_transition_SIRV, initial_state

app = Flask(__name__)

# selected_alg_type = 'sir'
# G = nx.complete_graph(10)

# if selected_alg_type == 'sir':
#     _state_transition_function = state_transition_SIR
# elif selected_alg_type == 'sirv':
#     _state_transition_function = state_transition_SIRV
# else:
#     raise ValueError(f"Invalid algorithm type: {selected_alg_type}")

# sim1 = Simulation(G, initial_state, _state_transition_function)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    data = request.get_json()
    num_nodes = int(data['numNodes'])
    graph_type = data['graphType']

    # Aggiungi la logica per generare il grafo in base al tipo scelto (esempio: Erdos Renyi)
    if graph_type == 'erdos_renyi':
        G = nx.erdos_renyi_graph(num_nodes, 0.2)  # Modifica la probabilità a seconda delle tue esigenze
    elif graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(num_nodes, 3)  # Modifica il numero di nodi da collegare a ciascun nuovo nodo
    elif graph_type == 'gnp_random':
        G = nx.gnp_random_graph(num_nodes, 0.2)  # Modifica la probabilità a seconda delle tue esigenze
    elif graph_type == 'fully_connected':
        G = nx.complete_graph(num_nodes)
    else:
        return jsonify({'error': 'Tipo di grafo non supportato'})

    # Puoi fare ulteriori azioni qui con il grafo G se necessario

    graph_data = json_graph.node_link_data(G)

    return jsonify({'success': True, 'graph': graph_data})

# @app.route('/run_simulation/<int:steps>')
# def run_simulation(steps):
#     simulation_result = sim1.run(steps)
#     return simulation_result.plot()

if __name__ == '__main__':
    app.run(debug=True)

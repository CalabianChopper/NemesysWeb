import random
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly as plt
import plotly.graph_objects as go
import pandas as pd
import io
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib as mpl


#Algoritmo SIRV

def initial_state(G):
    state = {node: 'S' for node in G.nodes}
    patient_zero_1 = random.choice(list(G.nodes))
    state[patient_zero_1] = 'I'
    return state

def state_transition_SIRV(G, current_state, alpha, beta, gamma, pvacc, feb):
    ALPHA = float(alpha)
    BETA = float(beta)
    GAMMA = float(gamma)
    PVACC = float(pvacc)
    FEB = float(feb)

    next_state = {}
    
    for node in G.nodes:
        if current_state[node] == 'I': # Se è infetto, può passare a ricoverato
            if random.random() < BETA:
                next_state[node] = 'R'
        elif current_state[node] == 'R': # Da ricoverato può tornare ad essere suscettibile
            if random.random() < GAMMA:
                next_state[node] = 'S'
        else: # Da suscettibile può passare a vaccinato
            if current_state[node] == 'S':
                if random.random() < PVACC:
                    next_state[node] = 'V'
                else: # Oppure in base allo stato dei vicini può infettarsi
                    for neighbor in G.neighbors(node):
                        if current_state[neighbor] == 'I' :
                            if random.random() < ALPHA:
                                next_state[node] = 'I'
                                if random.random() < FEB:
                                    fever = True
                            break
    return next_state

#Algoritmo SIR 

def state_transition_SIR(G, current_state, alpha, beta, gamma):
    ALPHA = float(alpha)
    BETA = float(beta)
    GAMMA = float(gamma)
    
    next_state = {}

    for node in G.nodes:
        if current_state[node] == 'I':
            if random.random() < BETA:
                next_state[node] = 'R'
        elif current_state[node] == 'R':
            if random.random() < GAMMA:
                next_state[node] = 'S'
        else:
            for neighbor in G.neighbors(node):
                if current_state[neighbor] == 'I':
                    if random.random() < ALPHA:
                        next_state[node] = 'I'
                        break

    return next_state


#Parte algoritmica

class StopCondition(StopIteration):
    pass

class Simulation:
    def __init__(self, G, initial_state, state_transition, stop_condition=None, name=''):

        self.G = G
        self._initial_state = initial_state
        self._state_transition_function = state_transition #Modificato
        self._stop_condition = stop_condition
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or 'Simulation'

        self._states = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap('tab10')

        self._initialize()

    def _append_state(self, state):
        self._states.append(state)
        for value in set(state.values()):
            if value not in self._value_index:
                self._value_index[value] = len(self._value_index)

    def _initialize(self):
        if self._initial_state:
            if callable(self._initial_state):
                state = self._initial_state(self.G)
            else:
                state = self._initial_state
            for n in self.G.nodes():
                nx.set_node_attributes(self.G, state, 'state')

        if any(self.G.nodes[n].get('state') is None for n in self.G.nodes):
            raise ValueError('All nodes must have an initial state')

        self._append_state(state)

    def _step(self):
        state = nx.get_node_attributes(self.G, 'state')
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopCondition
        
        if self._state_transition_function == state_transition_SIR: #Modificato
            new_state = state_transition_SIR(self.G, state)
        elif self._state_transition_function == state_transition_SIRV:
            new_state = state_transition_SIRV(self.G, state)
        else:
            raise ValueError(f"Invalid state transition function: {self._state_transition_function}")
        
        state.update(new_state)
        nx.set_node_attributes(self.G, state, 'state')
        self._append_state(state)

    def _categorical_color(self, value):
        index = self._value_index[value]
        node_color = self._cmap(index)
        return node_color

    @property
    def steps(self):
        return len(self._states) - 1

    def state(self, step=-1):
        try:
            return self._states[step]
        except IndexError:
            raise IndexError('Simulation step %i out of range' % step)

    def draw(self, step=-1, labels=None, **kwargs):
        state = self.state(step)
        node_colors = [self._categorical_color(state[n]) for n in self.G.nodes]
        nx.draw(self.G, pos=self._pos, node_color=node_colors, **kwargs)

        if labels is None:
            labels = sorted(set(state.values()), key=self._value_index.get)
        patches = [mpl.patches.Patch(color=self._categorical_color(l), label=l)
                   for l in labels]
        plt.legend(handles=patches)

        if step == -1:
            step = self.steps
        if step == 0:
            title = 'initial state'
        else:
            title = 'step %i' % (step)
        if self.name:
            title = '{}: {}'.format(self.name, title)
        plt.title(title)

    def plot(self, min_step=None, max_step=None, labels=None, **kwargs):
        x_range = range(min_step or 0, max_step or len(self._states))
        counts = [Counter(s.values()) for s in self._states[min_step:max_step]]
        if labels is None:
            labels = {k for count in counts for k in count}
            labels = sorted(labels, key=self._value_index.get)

        for label in labels:
            series = [count.get(label, 0) / sum(count.values()) for count in counts]
            plt.plot(x_range, series, label=label, **kwargs)

        title = 'node state proportions'
        if self.name:
            title = '{}: {}'.format(self.name, title)
        plt.title(title)
        plt.xlabel('Simulation step')
        plt.ylabel('Proportion of nodes')
        plt.legend()
        plt.xlim(x_range.start)
        
        plt.show()

        return plt.gca()

    def run(self, steps=1):
        for _ in range(steps):
            try:
                self._step()
            except StopCondition as e:
                print("Stop condition met at step %i." % self.steps)
                break
        return self

#Parte della web app

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
    html.H1([
    "Nemesys Software by University of Catanzaro and ",
    html.A("Quantum Minds Tech", href='https://qmt.pythonanywhere.com')
    ]), 
    html.Label('Select The Model of the Network'), #Pietro 03-12 Change
    dcc.Dropdown(
        id='graph-type',
        options=[
            {'label': 'Erdos Renyi Network', 'value': 'erdos_renyi'},
            {'label': 'A GNP Random Graph', 'value': 'gnp'},# occhio mi sa che alla fine networkX GNP e Erdos Reny sono le stesse
            {'label': 'Barabasi-Albert Network', 'value': 'barabasi_albert'},
            {'label': 'Stochastic Block Model SBM', 'value': 'sbm'},
            {'label': 'Fully Connected Network', 'value': 'full'}
        ],
        value='erdos_renyi', 
        style={'display': 'block', 'marginBottom': '10px'}
    ),
    html.Label('Number of Nodes (or size of clusters SBM)'),
    dcc.Input(
        id='num-nodes',
        type='number',
        value=10,
        min=1, 
        max=1000,
        step=1,
        style={'display': 'block', 'marginBottom': '20px'}
    ),
    html.Label('Edge Probability between Nodes (or number of clusters for SBM)'),
    dcc.Input(
        id='probability',
        type='number',
        value=0.10,
        min=0,
        max=1,
        step=0.1,
        style={'display': 'block', 'marginBottom': '20px'}
    ),
    dcc.Graph(id='graph-image', config={'scrollZoom': True, 'doubleClick': 'reset'}),
    html.Div(children=[
        html.Button("Download Nodes CSV", id='btn-nodes', n_clicks=0),
        html.Span(style={'marginRight': '50px'}),
        html.Button("Download Edges CSV", id='btn-edges', n_clicks=0),
        dcc.Download(id="download-data"),
    ], style={'display': 'flex', 'flexDirection': 'row', 'margin': '20px', 'padding' : '20px'}),

    html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
        html.H1('Select the Simulation Model'), 
        html.Label('Select Model'),
        dcc.Dropdown(
            id='alg-type',
            options=[
                {'label': 'SIR Model', 'value': 'sir'},
                {'label': 'SIRV Model', 'value': 'sirv'},
            ],
            value='sir', 
            style={'display': 'block', 'marginBottom': '10px'}
        ),
        html.Span(style={'margin': '15px'}),
        html.Label('ALPHA (S->I):'),
        dcc.Input(
            id='input-field-1',
            type='number',
            value='0.1',
            min='0',
            max='1',
            step='0.1',
            style={'display': 'block', 'margin': '10px'},
        ),
        html.Label('BETA (I->R):'),
        dcc.Input(
            id='input-field-2',
            type='number',
            value='0.1',
            min='0',
            max='1',
            step='0.1',
            style={'display': 'block', 'margin': '10px'},
        ),
        html.Label('GAMMA (R->S):'),
        dcc.Input(
            id='input-field-3',
            type='number',
            value='0.1',
            min='0',
            max='1',
            step='0.1',
            style={'display': 'block', 'margin': '10px'},
        ),
        html.Label('PVACC (S->V):'),
        dcc.Input(
            id='input-field-4',
            type='number',
            value='0.1',
            min='0',
            max='1',
            step='0.1',
            style={'display': 'block', 'margin': '10px'},
        ),
        html.Label('FEVER (I->F):'),
        dcc.Input(
            id='input-field-5',
            type='number',
            value='0.1',
            min='0',
            max='1',
            step='0.1',
            style={'display': 'block', 'margin': '10px'},
        ),
   
    ]), 
    html.Div([
        html.Span(style={'margin': '15px'}),
        html.Button(id='simulation-button', children='Run Simulation', n_clicks=0),
        html.Span(style={'margin': '15px'}),
        dcc.Graph(id='simulation-graph'),  
    ]),
])

G = None

@app.callback(
   Output('graph-image', 'figure'),
   [Input('graph-type', 'value'), Input('num-nodes', 'value'), Input('probability', 'value')]
)
def update_graph(graph_type, num_nodes, probability):
    global G
    if graph_type == 'erdos_renyi':
        G = nx.erdos_renyi_graph(num_nodes, probability)
    elif graph_type == 'gnp':
        G = nx.gnp_random_graph(num_nodes, probability)
    elif graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(num_nodes, 3)
    elif graph_type == 'full':
        G = nx.complete_graph(num_nodes)
    elif graph_type == 'sbm':# ho scritto con i piedi va ottimizzato, NS deve essere una lista di dimensione PROBABILITY e fatta tutta da elementi uguali a NUM_NODES
        ns = list()# size of clusters
        ps2= list()
        ps=list()
        for i in range(probability):
            ns.append(num_nodes)
            ps2.append(random.random())
        for i in range(probability):
            ps.append(ps2)
        #ps = [[0.3, 0.01, 0.01], [0.01, 0.3, 0.01], [0.01, 0.01, 0.3]] # probability of edge
        G = nx.stochastic_block_model(ns, ps)
        #G = nx.stochastic_block_model(num_nodes, probability, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)
    else:
        raise ValueError('Invalid graph type')

    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(l=50, r=50, b=100, t=100, pad=4),
                        annotations=[dict(
                            text="QMT Github: <a href='https://github.com/QuantumMindsTech'>Link</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
    return fig

@app.callback(
    Output('input-field-1', 'style'),
    Output('input-field-2', 'style'),
    Output('input-field-3', 'style'),
    Output('input-field-4', 'style'),
    Output('input-field-5', 'style'),
    Input('alg-type', 'value')
)
def update_input_fields(alg_type):
    if alg_type == 'sir':
        return {'display': 'block', 'margin': '10px'}, {'display': 'block', 'margin': '10px'}, {'display': 'block', 'margin': '10px'}, {'display': 'none'}, {'display': 'none'}
    elif alg_type == 'sirv':
        return {'display': 'block', 'margin': '10px'}, {'display': 'block', 'margin': '10px'}, {'display': 'block', 'margin': '10px'}, {'display': 'block', 'margin': '10px'}, {'display': 'block', 'margin': '10px'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(
   Output('probability', 'style'),
   [Input('graph-type', 'value')]
)
def update_probability_visibility(graph_type):
   if graph_type == 'full':
       return {'display': 'none'}
   else:
       return {'display': 'block'}
   
@app.callback(
    Output("download-data", "data"),
    [Input('btn-nodes', 'n_clicks'),
    Input('btn-edges', 'n_clicks'),
    Input('btn-graph', 'n_clicks')],
    prevent_initial_call=True
)
def download_data(btn_nodes, btn_edges, btn_graph):
    global G
    ctx = dash.callback_context
    trigger_id = ctx.triggered_id.split('.')[0]

    if trigger_id == 'btn-nodes':
        nodes_data = {'Node': list(G.nodes())}
        nodes_df = pd.DataFrame(nodes_data)
        return dcc.send_data_frame(nodes_df.to_csv, filename="nodes.csv")
    elif trigger_id == 'btn-edges':
        edges_data = {'Node1': [edge[0] for edge in G.edges()], 'Node2': [edge[1] for edge in G.edges()]}
        edges_df = pd.DataFrame(edges_data)
        return dcc.send_data_frame(edges_df.to_csv, filename="edges.csv")
    elif trigger_id == 'btn-graph':
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
            line_width=2))

        figure = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Network graph',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(l=50, r=50, b=100, t=100, pad=4),
                                annotations=[dict(
                                    text="QMT Github: <a href='https://github.com/QuantumMindsTech'>Link</a>",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002)],
                                xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=True, zeroline=False, showticklabels=False))
                            )
        img = figure.to_image(format='png')
        img_str_io = io.BytesIO(img)
        return dcc.send_bytes(img_str_io, filename="graph.png")
    else:
        return None

@app.callback(
    Output('simulation-graph', 'figure'),
    [Input('simulation-button', 'n_clicks')],
    [State('alg-type', 'value')],
    [State('graph-type', 'value'), State('num-nodes', 'value'), State('probability', 'value')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, selected_alg_type, graph_type, num_nodes, probability):
    print("Callback executed")
    if n_clicks > 0:
        # Creazione del grafo
        if graph_type == 'erdos_renyi':
            G = nx.erdos_renyi_graph(num_nodes, probability)
        elif graph_type == 'gnp':
            G = nx.gnp_random_graph(num_nodes, probability)
        elif graph_type == 'barabasi_albert':
            G = nx.barabasi_albert_graph(num_nodes, 3)
        elif graph_type == 'full':
            G = nx.complete_graph(num_nodes)
        elif graph_type == 'sbm':
            ns = [num_nodes] * probability
            ps2 = [random.random() for _ in range(probability)]
            ps = [ps2] * probability
            G = nx.stochastic_block_model(ns, ps)
        else:
            raise ValueError('Invalid graph type')

        # Selezione della funzione di transizione di stato
        if selected_alg_type == 'sir':
            _state_transition_function = state_transition_SIR
        elif selected_alg_type == 'sirv':
            _state_transition_function = state_transition_SIRV
        else:
            raise ValueError(f"Invalid algorithm type: {selected_alg_type}")

        # Esecuzione della simulazione
        simulation = Simulation(G, initial_state, _state_transition_function)
        simulation_result = simulation.run(10)

        # Restituzione del grafico
        return simulation_result.plot()

    # Restituisci un grafico vuoto o None se il pulsante non è stato premuto
    return go.Figure()

if __name__ == '__main__':
  app.run_server(debug=True)

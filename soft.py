import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go
import matplotlib as mpl
import random
from collections import Counter

app = dash.Dash(__name__)

class StopCondition(StopIteration):
    pass

class Simulation:
    def __init__(self, G, initial_state, state_transition, stop_condition=None, name=''):
     
        self.G = G 
        self._initial_state = initial_state
        self._state_transition = state_transition
        self._stop_condition = stop_condition
        #self._fever = False
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or 'Simulation'

        self._states = []
        #self._fevers = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap('tab10')

        self._initialize()

    def _append_state(self, state):
        self._states.append(state)
        #self._fevers.append(fever)
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
                #nx.set_node_attributes(self.G, fever, 'febbre')
                nx.set_node_attributes(self.G, random.randint(1, 80), 'Eta')
                nx.set_node_attributes(self.G, random.choice(["Uomo", "Donna"]), 'Sesso')
                nx.set_node_attributes(self.G, random.choice([True, False]), 'Malattie Pregresse')

        if any(self.G.nodes[n].get('state') is None for n in self.G.nodes):
            raise ValueError('All nodes must have an initial state')

        self._append_state(state)
        print ("Stato iniziale ed attributi: ")
        print(self.G.nodes.data())


    def _step(self):
        state = nx.get_node_attributes(self.G, 'state')
        fever = nx.get_node_attributes(self.G, 'febbre')
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopCondition
        state = nx.get_node_attributes(self.G, 'state')
        fever = nx.get_node_attributes(self.G, 'febbre')
        new_state = self._state_transition(self.G, state)
        state.update(new_state)
        nx.set_node_attributes(self.G, state, 'state')
        nx.set_node_attributes(self.G, fever, 'febbre')
        self._append_state(state)
        nswap = int(0.05* self.G.number_of_edges())
        edges = list (self.G.edges())
        selected=list()
        for i in range(0, nswap):
            random_choice = random.choice(edges)
            if(random_choice not in selected):
                self.G.remove_edge(*random_choice)
            selected.append(random_choice)
        noedges = list (nx.non_edges(self.G))
        for i in range(0, nswap):
            random_choice = random.choice(noedges)
            self.G.add_edge(*random_choice)

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

        return plt.gca()

    def run(self, steps=1):
        for _ in range(steps):
            try:
                self._step();
            except StopCondition as e:
                print("Stop condition met at step %i." % self.steps);
                break;
        print(self.G.edges());
        
def initial_state(G):
    state = {}
    for node in G.nodes:
        state[node] = 'S'
    patient_zero_1 = random.choice(list(G.nodes))
    state[patient_zero_1] = 'I'
    return state

def simulate_sir(graph, num_nodes, probability):
    # Implementazione della simulazione SIR
    pass

def simulate_sirv(graph, num_nodes, probability):
    BETA = 0.1
DELTA= 0.1
ALPHA = 0.1
PVACC=0.05
FEB=0.3
def state_transition_SIRV(G, current_state):
    next_state = {}
    for node in G.nodes:
        if current_state[node] == 'I': #Se è infected può passare a ricovered
            if random.random() < DELTA:
                next_state[node] = 'R'
        elif current_state[node] == 'R': #Da recovered può tornare ad essere susceptible
            if random.random() < ALPHA:
                next_state[node] = 'S'
        else: #Da susceptible può passare a vaccinated
            if current_state[node] == 'S':
                if random.random() < PVACC:
                    next_state[node] = 'V'
                else: #Oppure in base allo stato dei vicini si può infettare
                    for neighbor in G.neighbors(node):
                        if current_state[neighbor] == 'I' :
                            if random.random() < BETA:
                                next_state[node] = 'I'
                                if random.random() < FEB:
                                    fever = True
    return next_state
    # pass

app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column'}, children=[
    html.H1('Nemesys Software by Quantum Minds Tech'), 
    html.Label('Select Graph Type'),
    dcc.Dropdown(
        id='graph-type',
        options=[
            {'label': 'Erdos Renyi Graph', 'value': 'erdos_renyi'},
            {'label': 'GNP Graph', 'value': 'gnp'},
            {'label': 'Barabasi Albert Graph', 'value': 'barabasi_albert'},
            {'label': 'Fully Connected Graph', 'value': 'full'}
        ],
        value='erdos_renyi', 
        style={'display': 'block', 'margin-bottom': '10px'}
    ),
    html.Label('Number of Nodes'),
    dcc.Input(
        id='num-nodes',
        type='number',
        value=10,
        min=1, 
        max=1000,
        step=1,
        style={'display': 'block', 'margin-bottom': '20px'}
    ),
    html.Label('Probability for Edge Creation'),
    dcc.Input(
        id='probability',
        type='number',
        value=0.10,
        min=0,
        max=1,
        step=0.1,
        style={'display': 'block', 'margin-bottom': '20px'}
    ),
    dcc.Graph(id='graph-image', config={'scrollZoom': True, 'doubleClick': 'reset'}), 
    html.Label('Select Simulation Type'),
    dcc.Dropdown(
        id='simulation-type',
        options=[
            {'label': 'SIR', 'value': 'sir'},
            {'label': 'SIRV', 'value': 'sirv'}
        ],
        value='sir',
        style={'display': 'block', 'margin-bottom': '10px'}
    ),
])

@app.callback(
   Output('graph-image', 'figure'),
   [Input('graph-type', 'value'), Input('num-nodes', 'value'), Input('probability', 'value'), Input('simulation-type', 'value')]
)
def update_graph(graph_type, num_nodes, probability):
   if graph_type == 'erdos_renyi':
       G = nx.erdos_renyi_graph(num_nodes, probability)
   elif graph_type == 'gnp':
       G = nx.gnp_random_graph(num_nodes, probability)
   elif graph_type == 'barabasi_albert':
       G = nx.barabasi_albert_graph(num_nodes, 3)
   elif graph_type == 'full':
       G = nx.complete_graph(num_nodes)
   else:
       raise ValueError('Invalid graph type')

   pos = nx.spring_layout(G)
   edge_x = []
   edge_y = []
   for edge in G.edges():
       x0, y0 = pos[edge[0]]
       x1, y1 = pos[edge[1]]
       edge_x.append(x0)
       edge_x.append(x1)
       edge_x.append(None)
       edge_y.append(y0)
       edge_y.append(y1)
       edge_y.append(None)

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
                titleside='right',
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],  
                ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1']
            ),
           line_width=2))

   fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title='Network graph',
                      titlefont_size=16,
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(l=50,
                            r=50,
                            b=100, 
                            t=100,
                            pad=4),
                      annotations=[ dict(
                          text="QMT Github: <a href='https://github.com/CalabianChopper'>Link</a>",
                          showarrow=False,
                          xref="paper", yref="paper",
                          x=0.005, y=-0.002 ) ],
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                      )
   return fig

@app.callback(
   Output('probability', 'style'),
   [Input('graph-type', 'value')]
)
def update_probability_visibility(graph_type):
   if graph_type == 'full':
       return {'display': 'none'}
   else:
       return {'display': 'block'}

if __name__ == '__main__':
   app.run_server(debug=True)

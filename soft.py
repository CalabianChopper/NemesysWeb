import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go

app = dash.Dash(__name__)


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
    dcc.Graph(id='graph-image', config={'scrollZoom': True, 'doubleClick': 'reset'})
])

@app.callback(
   Output('graph-image', 'figure'),
   [Input('graph-type', 'value'), Input('num-nodes', 'value'), Input('probability', 'value')]
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
               titleside='right'
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
                          text="QMT Github: <a href='https://github.com/QuantumMindsTech'>Link</a>",
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

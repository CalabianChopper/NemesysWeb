# -*- coding: utf-8 -*-
"""
Created on 27/12/22

@author: Francesco & Paperella 
"""

from collections import Counter
# from operator import itemgetter
# import pandas as pd
# import os
# import sys
# import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# import copy
import matplotlib as mpl
import random
# import tensorflow as tf
# import stellargraph as sg
# from stellargraph.data import EdgeSplitter
# from stellargraph.layer import GraphSAGE, link_classification
# from stellargraph.mapper import GraphSAGELinkGenerator
# from stellargraph.data import UniformRandomWalk
# from stellargraph.data import UnsupervisedSampler
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing, feature_extraction, model_selection
# from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
# from sklearn.metrics import accuracy_score
# from stellargraph import globalvar
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from stellargraph.mapper import GraphSAGENodeGenerator
# from stellargraph import StellarGraph
# from tensorflow import keras

#Classe di stop
class StopCondition(StopIteration):
    pass

#Classe di simulazione
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
                
#Definizione stato iniziale
def initial_state(G):
    state = {}
    for node in G.nodes:
        state[node] = 'S'
    patient_zero_1 = random.choice(list(G.nodes))
    state[patient_zero_1] = 'I'
    return state

#Algoritmo SIRV
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

#Creazione GRAFO e visualizzazione dati 
G1 = nx.barabasi_albert_graph(1000, 3)
lista_eta = []
lista_sesso = []
lista_mal = []
#for n in G1.nodes():
    #nx.set_node_attributes(G1, random.randint(1, 80), 'eta')
    #nx.set_node_attributes(G1, random.choice(["Uomo", "Donna"]), 'sesso')
    #nx.set_node_attributes(G1, random.choice([True, False]), 'malattie')
#nx.draw(G1,with_labels=True)
"""
for i in range(0, len(G1.nodes())):
    lista_eta.append(random.randint(1, 80))
for i in range(0, len(G1.nodes())):
    lista_sesso.append(random.choice(["Uomo", "Donna"]))
for i in range(0, len(G1.nodes())):
    lista_mal.append(random.choice([True, False]))
"""
print(G1)
print("Numero di archi:")
print(G1.number_of_edges())
print("Numero di nodi: ")
print(G1.number_of_nodes())
print("Connessioni: ")
print(G1.degree())
#print("Attributi legati ai nodi: ")
#print(G1.nodes.data())
#Processo di simulazione
test_state = initial_state(G1)
#print("Test iniziale: ")
#print(state_transition_SIRV(G1, test_state))
sim1 = Simulation(G1, initial_state, state_transition_SIRV, name='SIRV on Barabasi')
#Visualizzazione stato
#print(sim1.state())
#Simulazione

print("Simulazione 0-9: ")
print("Rewinding archi:")
print(sim1.run(10))
print("Evoluzione stato:")
print(sim1.state())
print("Simulazione 10-19: ")
print("Rewinding archi:")
print(sim1.run(10))
print("Evoluzione stato:")
print(sim1.state())
print("Simulazione 10-29: ")
print("Rewinding archi:")
print(sim1.run(10))
print("Evoluzione stato:")
print(sim1.state())
print("Simulazione 30-39: ")
print("Rewinding archi:")
print(sim1.run(10))
print("Evoluzione stato:")
print(sim1.state())
print("Simulazione 40-49: ")
print("Rewinding archi:")
print(sim1.run(10))
print("Evoluzione stato:")
print(sim1.state())

sim1.plot()

f.write("Lista nodi: ")
f.write("\n")
for i in G1.nodes:
    f.write(format(i)) 
    f.write("\n")
f.write('Archi')
f.write("\n")
for i in G1.edges:
    f.write(format(i))
    f.write('\n')




f.write("\n")
f.close()


"""
Parte di embedding da vedere dopo
#Embedding e Deep Learning
nodi = list(G1.nodes())
cammini = 1
lunghezza = 5
f = pd.DataFrame({ "stati" : lista_stati, "eta" : lista_eta, "malattie pregresse" : lista_malattie, "sesso" : lista_sesso})
Gs = StellarGraph.from_networkx(G1, node_features = sim1.infograph())
Gs.info()
u_s = UnsupervisedSampler(Gs, nodes=nodi, length=lunghezza, number_of_walks=cammini)
batch = 20
epoche = 20
samples = [5, 5]
generator = GraphSAGELinkGenerator(Gs, batch, samples)
train_gen = generator.flow(u_s)
layer_sizes = [50, 50]
graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2")
x_inp, x_out = graphsage.in_out_tensors()
prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)
model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)
history = model.fit(
    train_gen,
    epochs=epoche,
    verbose=1,
    use_multiprocessing=False,
    workers=8,
    shuffle=True,
)
x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
node_ids = f.index
node_gen = GraphSAGENodeGenerator(Gs, batch, samples).flow(node_ids)
node_embeddings = embedding_model.predict(node_gen, workers=10, verbose=1)
node_subject = f.astype("category")

X = node_embeddings
if X.shape[1] > 2:
    transform = TSNE  # PCA

    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_ids)
    emb_transformed["label"] = node_subject
else:
    emb_transformed = pd.DataFrame(X, index=node_ids)
    emb_transformed = emb_transformed.rename(columns={"0": 0, "1": 1})
    emb_transformed["label"] = node_subject
alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="Set1",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of GraphSAGE embeddings for SIRV data".format(transform.__name__)
)
import matplotlib.patches as mpatches
a = mpatches.Patch(color='grey', label='V')
b = mpatches.Patch(color='brown', label='I')
c = mpatches.Patch(color='purple', label='S')
d = mpatches.Patch(color='red', label='R')
plt.legend(handles=[a,b,c,d])
plt.show()
X = node_embeddings
y = np.array(f)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, test_size=None, stratify=y)
clf = LogisticRegression(verbose=0, solver="lbfgs", multi_class="auto")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
pd.Series(y_pred).value_counts()
"""

###
### This is the grossest thing ever
### I'm so sorry to anyone who sees this...
### all I can say is, I'm in a huge rush... lol
### John, March 17, 2021
###


import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
from ndlib.utils import multi_runs

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import os
import random
from collections import Counter

# from dcss import set_style
# set_style()

with open('for_viz.pickle', 'rb') as handle:
    for_viz = pickle.load(handle)

# population_size = 300
# G = nx.watts_strogatz_graph(population_size, 4, 0.15)





layout = nx.nx_pydot.graphviz_layout(G)



def sir_model(network, beta, gamma, fraction_infected):
    model = ep.SIRModel(network)

    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_parameter('gamma', gamma)
    config.add_model_parameter("fraction_infected", fraction_infected)

    model.set_initial_status(config)
    return model



sir_model_1 = sir_model(G, beta=0.05, gamma=0.01, fraction_infected=0.1)
sir_model_1


sir_1_iterations = sir_model_1.iteration_bunch(200, node_status=True)



def who_is_infected(iterations):
    """
    Returns a list of lists where the outter list is an
    iteration of the model and the inner list is a list of
    newly infected nodes.
    """
    infections_over_time = []

    for i, each in enumerate(iterations):
        infected = []
        for k, v in each['status'].items():
            if v == 1:
                infected.append(k)
        infections_over_time.append(infected)
    return infections_over_time


def who_is_infected_so_far(iterations, until_iteration=2):
    infection_lol = who_is_infected(iterations)
    until = infection_lol[:until_iteration]
    return until


def flatten(lol):
    return [item for sublist in lol for item in sublist]


to_highlight = {}

for i in range(100):
    to_highlight[i] = flatten(who_is_infected_so_far(sir_1_iterations, i))







fig, axs = plt.subplots(5,4, figsize=(23,33))

# THE FIRST ROW
nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[0,0])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[0,0],
                      nodelist=to_highlight[1])
axs[0,0].set_title('T0', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[0,1])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[0,1],
                      nodelist=to_highlight[2])
axs[0,1].set_title('T1', fontsize=23)



nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[0,2])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[0,2],
                      nodelist=to_highlight[3])
axs[0,2].set_title('T2', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[0,3])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[0,3],
                      nodelist=to_highlight[4])
axs[0,3].set_title('T3', fontsize=23)


# THE SECOND ROW
nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[1,0])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[1,0],
                      nodelist=to_highlight[5])
axs[1,0].set_title('T4', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[1,1])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[1,1],
                      nodelist=to_highlight[6])
axs[1,1].set_title('T5', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[1,2])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[1,2],
                      nodelist=to_highlight[7])
axs[1,2].set_title('T6', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[1,3])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[1,3],
                      nodelist=to_highlight[8])
axs[1,3].set_title('T7', fontsize=23)


# THE THIRD ROW
nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[2,0])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[2,0],
                      nodelist=to_highlight[9])
axs[2,0].set_title('T8', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[2,1])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[2,1],
                      nodelist=to_highlight[10])
axs[2,1].set_title('T9', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[2,2])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[2,2],
                      nodelist=to_highlight[11])
axs[2,2].set_title('T10', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[2,3])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[2,3],
                      nodelist=to_highlight[12])
axs[2,3].set_title('T11', fontsize=23)


# THE FOURTH ROW
nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[3,0])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[3,0],
                      nodelist=to_highlight[13])
axs[3,0].set_title('T12', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[3,1])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[3,1],
                      nodelist=to_highlight[14])
axs[3,1].set_title('T13', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[3,2])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[3,2],
                      nodelist=to_highlight[15])
axs[3,2].set_title('T14', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[3,3])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[3,3],
                      nodelist=to_highlight[16])
axs[3,3].set_title('T15', fontsize=23)


# THE FIFTH ROW
nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[4,0])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[4,0],
                      nodelist=to_highlight[17])
axs[4,0].set_title('T16', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[4,1])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[4,1],
                      nodelist=to_highlight[18])
axs[4,1].set_title('T17', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[4,2])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[4,2],
                      nodelist=to_highlight[19])
axs[4,2].set_title('T18', fontsize=23)


nx.draw(G, pos=layout, node_color='gray', edge_color='gray',
            node_size=100, width=.5, ax=axs[4,3])
nx.draw_networkx_nodes(G, pos=layout, node_color='crimson',
                      node_size=100, ax=axs[4,3],
                      nodelist=to_highlight[20])
axs[4,3].set_title('T19', fontsize=23)

plt.savefig('activations_first_20.pdf')
plt.savefig('activations_first_20.png', dpi=300)

# The Garden of Forking Data

This tutorial is adapted from McElreath's *Statistical Rethinking*. 

## Imports

```python
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import pandas as pd
```

```python
L0_Root = ["0", '1', '2', '3', '4']
L1_SM01 = ['1', '5', '6', '7', '8']
L1_SM02 = ['2', '9', '10', '11', '12']
L1_SM03 = ['3', '13', '14', '15', '16']
L1_SM04 = ['4', '17', '18', '19', '20']
L2_SM05 = ['5', '21', '22', '23', '24']
L2_SM06 = ['6', '25', '26', '27', '28']
L2_SM07 = ['7', '29', '30', '31', '32']
L2_SM08 = ['8', '33', '34', '35', '36']
L2_SM09 = ['9', '37', '38', '39', '40']
L2_SM10 = ['10', '41', '42', '43', '44']
L2_SM11 = ['11', '45', '46', '47', '48']
L2_SM12 = ['12', '49', '50', '51', '52']
L2_SM13 = ['13', '53', '54', '55', '56']
L2_SM14 = ['14', '57', '58', '59', '60']
L2_SM15 = ['15', '61', '62', '63', '64']
L2_SM16 = ['16', '65', '66', '67', '68']
L2_SM17 = ['17', '69', '70', '71', '72']
L2_SM18 = ['18', '73', '74', '75', '76']
L2_SM19 = ['19', '77', '78', '79', '80']
L2_SM20 = ['20', '81', '82', '83', '84']

watercolor = '#45A0F8'
landcolor = '#648225'
```

```python
every_draw = [L0_Root, L1_SM01, L1_SM02, L1_SM03, L1_SM04, L2_SM05, L2_SM06, L2_SM07, L2_SM08, L2_SM09,
              L2_SM10, L2_SM11, L2_SM12, L2_SM13, L2_SM14, L2_SM15, L2_SM16, L2_SM17, L2_SM18, L2_SM19, L2_SM20]
edges = []
for draw in every_draw:
    edges.append([draw[0], draw[1]])
    edges.append([draw[0], draw[2]])
    edges.append([draw[0], draw[3]])
    edges.append([draw[0], draw[4]])

edges = pd.DataFrame(edges)
edges.columns = ['source', 'target']

G = nx.from_pandas_edgelist(edges)
pos = graphviz_layout(G, prog="twopi")
```

```python
water_samples = ['1', '5', '9', '13', '17', '21', '25', '29', '33', '37', '41', '45', '49', '53', '57', '61', '65', '69', '73', '77', '81']
```


```python
def assign_color(nodelist, waterlist, watercolor=watercolor, landcolor=landcolor):
    return [watercolor if node in waterlist else landcolor for node in nodelist]

draw_one_nodes = ['1', '2', '3', '4']
draw_one_colors = assign_color(draw_one_nodes, water_samples)

draw_two_nodes = ['5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
draw_two_colors = assign_color(draw_two_nodes, water_samples)
```



```python
draw_three_nodes = []
for node in G.nodes():
    if node not in draw_one_nodes:
        if node not in draw_two_nodes:
            draw_three_nodes.append(node)
draw_three_colors = assign_color(draw_three_nodes, water_samples)

draw_one_all_paths, draw_two_all_paths, draw_three_all_paths = [('0', '1'), ('0', '2'), ('0', '3'), ('0', '4')], [], []
```

```python
for edge in G.edges():
    if edge[0] in draw_two_nodes:
        draw_two_all_paths.append(edge)
    if edge[1] in draw_two_nodes:
        draw_two_all_paths.append(edge)

for edge in G.edges():
    if edge[0] in draw_three_nodes:
        draw_three_all_paths.append(edge)
    if edge[1] in draw_three_nodes:
        draw_three_all_paths.append(edge)
```

```python
paths_1 = [
    ('0', '1')
]

paths_2 = [
    ('0', '1'),
    ('1', '6'),
    ('1', '7'),
    ('1', '8')
]

paths_3 = [
    ('0', '1'),
    ('1', '6'),
    ('1', '7'),
    ('1', '8'),
    ('6', '25'),
    ('7', '29'),
    ('8', '33')
]
```

```python
def plot_garden(G, pos, draw_one, draw_two, draw_three, paths, filename):
    plt.figure(figsize=(12, 12))
    # Draw everything to ensure layout doesn't change when you produce a series of images
    nx.draw_networkx_nodes(G, pos, nodelist=['0'], node_size=1_000, alpha=1, node_color='black', node_shape='8')
    nx.draw_networkx_nodes(G, pos, node_size=320, alpha=1, node_color='gray')
    nx.draw_networkx_edges(G, pos, style='dashed', width=1, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='white')

    if draw_one is True:
        nx.draw_networkx_nodes(G, pos, nodelist=draw_one_nodes, node_size=320, alpha=1, node_color=draw_one_colors)
        nx.draw_networkx_edges(G, pos, style='dashed', width=1, edgelist=draw_one_all_paths, edge_color=landcolor)

    if draw_two is True:
        nx.draw_networkx_nodes(G, pos, nodelist=draw_two_nodes, node_size=320, alpha=1, node_color=draw_two_colors)
        nx.draw_networkx_edges(G, pos, style='dashed', width=1, edgelist=draw_two_all_paths, edge_color=landcolor)

    if draw_three is True:
        nx.draw_networkx_nodes(G, pos, nodelist=draw_three_nodes, node_size=320, alpha=1, node_color=draw_three_colors)
        nx.draw_networkx_edges(G, pos, style='dashed', width=1, edgelist=draw_three_all_paths, edge_color=landcolor)

    if paths:
        nx.draw_networkx_edges(G, pos, edge_color='black', edgelist=paths, width=6)
    
    nx.draw_networkx_nodes(G, pos, node_size=320, alpha=1, node_color='white', nodelist=[n for n in G.nodes()][0])
    plt.axis("equal")
    plt.axis("off")
    plt.savefig(f'../figures/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
```



```python
plot_garden(G=G, pos=pos, draw_one=False, draw_two=False, draw_three=False, paths=False, filename="garden_of_forking_data_0")

plot_garden(G=G, pos=pos, draw_one=True, draw_two=False, draw_three=False, paths=False, filename="garden_of_forking_data_1")

plot_garden(G=G, pos=pos, draw_one=True, draw_two=True, draw_three=False, paths=False, filename="garden_of_forking_data_2")

plot_garden(G=G, pos=pos, draw_one=True, draw_two=True, draw_three=True, paths=False, filename="garden_of_forking_data_3")

plot_garden(G=G, pos=pos, draw_one=True, draw_two=True, draw_three=True, paths=paths_1, filename="garden_of_forking_data_4")

plot_garden(G=G, pos=pos, draw_one=True, draw_two=True, draw_three=True, paths=paths_2, filename="garden_of_forking_data_5")

plot_garden(G=G, pos=pos, draw_one=True, draw_two=True, draw_three=True, paths=paths_3, filename="garden_of_forking_data_6")
```
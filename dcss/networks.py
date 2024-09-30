import pandas as pd
import random
import math
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)


def check_beta_validity(beta_values, lambda_max, lambda_min):
    """
    Checks the validity of beta values based on the eigenvalues of the adjacency matrix.
    
    Parameters:
    - beta_values: list of floats, attenuation factors.
    - lambda_max: float, maximum absolute eigenvalue of the adjacency matrix.
    - lambda_min: float, minimum eigenvalue of the adjacency matrix.
    
    Returns:
    - valid_betas: list of valid beta values.
    - invalid_betas: list of invalid beta values.
    """
    valid_betas = []
    invalid_betas = []
    
    for beta in beta_values:
        if beta >= 0:
            if beta < 1 / lambda_max:
                valid_betas.append(beta)
            else:
                print(f"Beta {beta} is too large (>= 1/lambda_max). Skipping.")
                invalid_betas.append(beta)
        else:
            if beta > 1 / lambda_min:
                valid_betas.append(beta)
            else:
                print(f"Beta {beta} is too small (<= 1/lambda_min). Skipping.")
                invalid_betas.append(beta)
    
    if valid_betas:
        print(f"\nValid beta values that will be used: {valid_betas}\n")
    else:
        print("\nNo valid beta values to use.\n")
    
    return valid_betas, invalid_betas


def bonacich_centrality_multiple_betas(G, beta_values, normalized=True):
    """
    Computes the Bonacich centrality for an undirected graph G for multiple beta values.
    
    Parameters:
    - G: NetworkX graph
    - beta_values: list of floats, attenuation factors.
    - normalized: bool, whether to normalize the centrality vector.
    
    Returns:
    - centralities: dict, mapping beta values to centrality score dictionaries.
    """
    # Get the adjacency matrix
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    N = A.shape[0]
    
    # Compute eigenvalues to find lambda_max and lambda_min
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = max(abs(eigenvalues))
    lambda_min = min(eigenvalues)
    
    # Check beta validity
    valid_betas, invalid_betas = check_beta_validity(beta_values, lambda_max, lambda_min)
    
    centralities = {}
    
    for beta in valid_betas:
        # Compute centrality vector c
        I = np.eye(N)
        try:
            inv_matrix = np.linalg.inv(I - beta * A)
        except np.linalg.LinAlgError:
            print(f"Matrix inversion failed for beta {beta}. Skipping.")
            continue
        
        c = np.dot(inv_matrix, np.ones(N))
        
        # Normalize c if requested
        if normalized:
            c = c / np.linalg.norm(c)
        
        # Map centrality scores to nodes
        centrality = dict(zip(sorted(G.nodes()), c))
        
        centralities[beta] = centrality
    
    return centralities



def get_attenuation_bounds(G):
    A = nx.to_numpy_array(G)
    eigenvalues = np.linalg.eigvals(A)
    lambda_max = max(abs(eigenvalues))
    lambda_min = min(eigenvalues)
    return (lambda_min, lambda_max)



def generate_attenuation_factors(
    attenuation_bounds, num_points, include_zero=True
):
    lambda_min, lambda_max = attenuation_bounds
    
    # Calculate beta bounds
    beta_positive_max = (1 / lambda_max) * 0.99  # Slightly less to avoid numerical issues
    beta_negative_min = (1 / lambda_min) * 0.99 if lambda_min != 0 else -np.inf  # Slightly more to avoid numerical issues
    
    # Generate positive beta values
    beta_positive_values = np.linspace(
        0, beta_positive_max, num=num_points // 2 + 1
    )
    
    # Generate negative beta values
    beta_negative_values = np.linspace(
        beta_negative_min, 0, num=num_points // 2 + 1
    )
    
    # Combine and sort beta values
    beta_values = np.unique(
        np.concatenate((beta_negative_values, beta_positive_values))
    )
    
    return beta_values








def simulation_by_iteration_matrices(multi_runs, population_size, state, proportion=False):
    """
    Constructs matrices for each status category in the model.
    Each row in the matrix represents a single simulation run.
    Each column in the matrix represents an iteration within a simulation.
    Each cell is the count (or proportion) of nodes that were in
    the state in question for that iteration of that simulation.
    """
    results = []
    for i, simulation in enumerate(multi_runs):
        sims = multi_runs[i]['trends']['node_count']
        results.append(sims[state])
    results = pd.DataFrame(results)
    if proportion is True:
        results = results.div(population_size)
    return results

def visualize_trends(multi_runs,
                     network,
                     states=[0, 1, 2],
                     labels=['Susceptible', 'Infected', 'Removed'],
                     highlight_state=1,
                     proportion=False,
                     return_data=False):
    """
    Gets the simulation by iteration matrix for each possible node state.
    """
    population_size = network.number_of_nodes()

    state_matrices = []
    medians = []

    for state in states:
        df = simulation_by_iteration_matrices(multi_runs,
                                              population_size=population_size,
                                              state=state,
                                              proportion=True)
        medians.append(df.median())
        df = df.T
        state_matrices.append(df)


    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    lstyles = ['--', '-', '-.']

    for i in range(len(state_matrices)):
        for column in state_matrices[i].T:
            plt.plot(state_matrices[i][column], alpha=.1, linewidth=.5)

    for i in range(len(medians)):
        if i is highlight_state:
            plt.plot(medians[i],
                     c='crimson',
                     label=labels[i],
                     linestyle=lstyles[i])
        else:
            plt.plot(medians[i],
                     c='black',
                     label=labels[i],
                     linestyle=lstyles[i])

    ax.set(xlabel='Iteration', ylabel='Proportion of nodes')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.show()

    if return_data is True:
        return state_matrices, medians



def rand_infection_set(network, frac):
    node_list = list(network.nodes())
    return random.sample(node_list, int(round(len(node_list)*frac, 0))) # randomly select nodes from node_list without replacement

def add_to_infection_set(infection_sets, fraction_increase, network):
    num_adds = int(round(network.number_of_nodes()*fraction_increase, 0)) # Number of new initial nodes needed to be added
    new_infection_sets = []
    for inf_set in infection_sets:
        new_set = copy.deepcopy(inf_set)
        while len(new_set) < len(inf_set) + num_adds: # Keep randomly selecting nodes, checking if they're already in the list, and adding if they haven't until the new set is as long as needed.
            new_add = random.choice(list(network.nodes()))
            if new_add not in new_set:
                new_set.append(new_add)
        new_infection_sets.append(new_set)
    return new_infection_sets




def create_enron_graph(df, employee_df):
    try:
        import graph_tool as gt
    except:
        print("Error importing graph-tool. Make sure that it's correctly installed.")
    G = gt.Graph(directed = True)

    employee_list = employee_df['id'].tolist()
    position_list = employee_df['position'].tolist()
    core_employee_set = set(employee_list)

    vertex_lookup = {}

    label = G.new_vertex_property('string')
    core = G.new_vertex_property('bool')
    position = G.new_vertex_property('string')

    for vertex in zip(employee_list, position_list):
        v = G.add_vertex()
        label[v] = vertex[0]
        position[v] = vertex[1]
        core[v] = True
        vertex_lookup[vertex[0]] = v

    edges_df = df.value_counts(['source', 'target']).reset_index(name='count').copy()
    source_nodes = edges_df['source'].tolist()
    target_nodes = edges_df['target'].tolist()
    source_nodes.extend(target_nodes)
    node_set = set(source_nodes)
    node_list = [x for x in node_set if x not in core_employee_set]

    for vertex in node_list:
        v = G.add_vertex()
        label[v] = vertex
        core[v] = False
        vertex_lookup[vertex] = v

    edge_weight = G.new_edge_property('int')

    source_list = edges_df['source'].tolist()
    target_list = edges_df['target'].tolist()
    weight_list = edges_df['count'].tolist()
    for nodes in zip(source_list, target_list, weight_list):
        from_idx = vertex_lookup[nodes[0]]
        to_idx = vertex_lookup[nodes[1]]
        if from_idx != to_idx:
            edge = G.add_edge(from_idx, to_idx)
            edge_weight[edge] = nodes[2]


    G.vertex_properties['label'] = label
    G.vertex_properties['core'] = core
    G.vertex_properties['position'] = position
    G.edge_properties['edge_weight'] = edge_weight
    return G



def get_block_membership(state, graph, employee_df, column_prefix):
    lookup = graph.graph_properties['vertex_lookup']

    levels = state.get_levels()
    base_level = levels[0].get_blocks()
    block_list = []

    for index, row in employee_df.iterrows():
        email = row['id']
        node = lookup[email]
        block_list.append(base_level[node])

    employee_df[column_prefix + '_block_id'] = block_list

    return employee_df



def get_shortest_paths(G, source, target):
    """
    Returns the nodes that are on the shortest path and
    the list of edge tuples that make up that path.
    """
    spath = nx.shortest_path(G, source, target)
    spathedges = set(zip(spath, spath[1:]))
    return spath, spathedges

def plot_path(G, layout, nodes_to_highlight, edges_to_highlight):
    fig, ax = plt.subplots(figsize=(12, 8))
    # the base network nodes
    nx.draw_networkx_nodes(G,
                           pos=layout,
                           node_size=250,
                           node_color='#32363A')

    # the base network edges
    nx.draw_networkx_edges(G,
                           pos=layout,
                           edge_color='darkgray',
                           width=1)

    # the path to highlight
    nx.draw_networkx_nodes(G,
                           pos=layout,
                           node_size=300,
                           node_color='crimson',
                           nodelist=nodes_to_highlight)
    nx.draw_networkx_edges(G,
                           pos=layout,
                           edge_color='crimson',
                           width=4,
                           edgelist=edges_to_highlight)

    # labels for nodes on the path
    path_labels = {}
    for node in nodes_to_highlight:
        path_labels[node] = node

    nx.draw_networkx_labels(G,
                            pos=layout,
                            font_size = 8,
                            labels=path_labels)
    plt.axis('off')







def label_radial_blockmodel(G, state):
    try:
        import graph_tool as gt
    except:
        print("Error importing graph-tool. Make sure that it's correctly installed.")

    t = gt.all.get_hierarchy_tree(state)[0]

    # calculate the graph positions for our nodes as well as the control points where our edges will bend
    # so they align more closely with the square hierarchy markers
    tpos = pos = gt.all.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
    cts = gt.all.get_hierarchy_control_points(G, t, tpos)
    pos = G.own_property(tpos)

    # we need a property map to rotate our node labels around the radial tree
    text_rot = G.new_vertex_property('double')

    for v in G.vertices():
        if pos[v][0] >0:
            text_rot[v] = math.atan(pos[v][1]/pos[v][0])
        else:
            text_rot[v] = math.pi + math.atan(pos[v][1]/pos[v][0])

    G.vertex_properties['text_rot'] = text_rot
    G.vertex_properties['pos'] = pos
    G.edge_properties['cts'] = cts

    return G

def get_block_membership(state, graph, employee_df, column_prefix):

    lookup = graph.graph_properties['vertex_lookup']

    levels = state.get_levels()
    base_level = levels[0].get_blocks()
    block_list = []

    for index, row in employee_df.iterrows():
        email = row['id']
        node = lookup[email]
        block_list.append(base_level[node])

    employee_df[column_prefix + '_block_id'] = block_list

    return employee_df


def blockmodel_from_edge_df(df, n_edges = None, use_weights = False):
    try:
        import graph_tool as gt
    except:
        print("Error importing graph-tool. Make sure that it's correctly installed.")

    G = gt.all.Graph(directed = False)
    weight = G.new_ep('int')
    avg_sent = G.new_ep('float')
    all_sents = G.new_ep('vector<float>')



    labels = G.add_edge_list(df[['source','target']].values, hashed = True, hash_type='string', eprops=[weight])
    labels = labels.coerce_type()

    if n_edges:
        mask_n = [True]*n_edges + [False]*(len(df) - 200)
    else:
        mask_n = [True]*len(df)
    top_mask = G.new_ep('bool')
    top_mask.a = mask_n

    G.ep['weight'] = weight
    G.ep['avg_sent'] = avg_sent
    G.ep['all_sents'] = all_sents
    G.ep['mask'] = top_mask
    G.vp['labels'] = labels

    if use_weights:
        blocks = gt.all.minimize_nested_blockmodel_dl(G, deg_corr = True, state_args=dict(recs=[G.ep['weight']],
                                                                               rec_types=['normal-real']))
    else:
        blocks = gt.all.minimize_nested_blockmodel_dl(G, deg_corr = True)



    return G, blocks

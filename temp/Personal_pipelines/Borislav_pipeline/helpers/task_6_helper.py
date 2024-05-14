import cv2
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.flow import shortest_augmenting_path


def create_graph(nodes_coordinates):
    """
    Create an undirected graph based on given nodes' coordinates.

    :param nodes_coordinates: (list(tuple)) A list of 2D coordinates (x, y) representing the nodes.

    :return:
    (networkx.Graph): An undirected graph where nodes are created based on the given coordinates,
      and edges are added if the Euclidean distance between nodes is less than or equal to 1.5.
    """
    G = nx.Graph()

    for idx, (x, y) in enumerate(nodes_coordinates):
        G.add_node((x, y))  # Adding nodes based on contour points

    for u in nodes_coordinates:
        for v in nodes_coordinates:
            if u != v:
                x1, y1 = u
                x2, y2 = v
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if distance <= 1.5:
                    G.add_edge(u, v)
    return G


def draw_graph_t0_img(graph, title, fig_size, color="yellow", node_size=1):
    """
    Converts the built-in networkx.draw() to an rgb array.

    :param graph: (networkx.Graph) object representing the root.
    :param title: (str) the of the chart.
    :param color: (array) the string representation of the colors of the nodes.
    :param node_size: (int) the size of each node.
    :param fig_size: (tuple) with shape (,2) in the format [width, height] of the figure.
    :return:
        (array) the representation of the built-in networkx.draw() function in an rgb pixel format.
    """
    pos = {node: node for node in graph.nodes()}

    from PIL import Image
    import io

    plt.figure()
    nx.draw(graph, pos=pos, with_labels=False, node_size=node_size, node_color=color)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)

    # im.show()
    arr = np.array(im)

    buf.close()
    return cv2.resize(arr, (fig_size[1], fig_size[0]), interpolation=cv2.INTER_LINEAR)


def draw_graph(graph, title, color="yellow", node_size=1, fig_size=(8, 6)):
    """
    An abstract function using the built-in networkx.draw() function which output can be seen only in a Jupyter notebook.

    :param graph: (networkx.Graph) object representing the root.
    :param title: (str) the of the chart.
    :param color: (array) the string representation of the colors of the nodes.
    :param node_size: (int) the size of each node.
    :param fig_size: (tuple) with shape (,2) in the format [width, height] of the figure.
    :return:
        NoneType (the output of the function can be seen only in a Jupyter Notebook).
    """
    plt.figure(figsize=fig_size)
    pos = {node: node for node in graph.nodes()}
    nx.draw(graph, pos=pos, with_labels=False, node_size=node_size, node_color=color)
    plt.title(title)
    plt.show()


def get_edges(nodes):
    """
    Layer of abstraction over the get_degrees() method which eliminates mistakes in future iterations. The main purpose
    of this function is to extract the endpoints of each root mask.

    :param nodes: list((tuple)) containing the coordinates of each node with its degree (how many nodes are connected
    to that node)
    :return:
    (array) containing the coordinates of the endpoints.
    """
    return get_degrees(nodes, 1)


def get_connection_nodes(nodes):
    """
    Layer of abstraction over the get_degrees() method which eliminates mistakes in future iterations. The main purpose
    of this function is to extract the connections of each root mask.

    :param nodes: list((tuple)) containing the coordinates of each node with its degree (how many nodes are connected
    to that node)
    :return:
    (array) containing the coordinates of the connections.
    """
    return get_degrees(nodes, 3)


def get_degrees(nodes, degrees):
    """
    Low level functions filtering the passed nodes as an input parameter based on their degree.

    :param nodes: list((tuple)) containing the coordinates of each node with its degree (how many nodes are connected
    to that node)
    :param degrees: (int) the desired degree that each node should have.
    :return:
    """
    return [node for node, degree in nodes if degree == degrees]


def get_connections(nodes):
    """
    Calculates the coordinates of the connections from the root mask.

    :param nodes: list((tuple)) containing the coordinates of each node with its degree (how many nodes are connected
    to that node)
    :return:
        (array) containing the coordinates of the connections.
    """
    connections = np.array(get_connection_nodes(nodes))

    sS = [np.sum(t) for t in connections]
    differences = np.abs(np.diff(sS))

    mask = np.concatenate(([True], differences > 1))
    # maskToEliminate = np.concatenate(([False], differences > 1))
    if len(mask) == 1:
        return -1
    filtered_array = connections[mask]
    return filtered_array


def morphometric_naive(df, G, edges_nodes, filtered_array):
    """
    Performs morphometric analysis on the given Graph object using a naive approach by taking the top most and the
    bottom most pixel considering the distance as the main root.

    :param df: (pandas.DataFrame) containing the x and y coordinates of each pixel recognized as root and a default
    color columns.
    :param G_init: (networkx.Graph) object representing the root.
    :param edges_nodes: (array) containing the coordinates of each pixel marked as an edge of the current Graph object.
    :param filtered_array: (array) the connections between the task_8_kaggle_evidence_main.py and the lateral roots.
    :return:
        (pandas.DataFrame) object containing the information for the main and lateral roots with their color
    representation.
    """
    try:
        start_node_demo = df.iloc[0][0]
        end_node_demo = df.iloc[-1][0]

        main_rooot_length = nx.shortest_path(G, start_node_demo, end_node_demo)
        print(len(main_rooot_length))
        df.loc[df['nodes'].isin(main_rooot_length), 'color'] = "green"
    except Exception:
        return -11
    return df


def morphometric(df, G_init, edges_nodes, filtered_array):
    """
    Performs morphometric analysis on the given Graph object by calculating the root with the most connections,
    therefore that root is considered the main root.

    :param df: (pandas.DataFrame) containing the x and y coordinates of each pixel recognized as root and a default
    color columns.
    :param G_init:  (networkx.Graph) object representing the root.
    :param edges_nodes: (array) containing the coordinates of each pixel marked as an edge of the current Graph object.
    :param filtered_array: (array) the connections between the task_8_kaggle_evidence_main.py and the lateral roots.
    :return:
        (pandas.DataFrame) object containing the information for the main and lateral roots with their color
    representation.
    """
    init_color = "green"
    start_node = edges_nodes[0]
    df.loc[df['nodes'] == start_node, 'color'] = str(init_color)

    if len(edges_nodes) == 2:
        main_rooot_length = nx.shortest_path(G_init, edges_nodes[0], edges_nodes[1])
        print("MAIN")
        print(len(main_rooot_length))
        df.loc[df['nodes'].isin(main_rooot_length), 'color'] = "green"
        return df

    for idx, edge in enumerate(edges_nodes[1:]):
        try:
            sh_path = nx.shortest_path(G_init, start_node, edge)
            sum = 0
            connection = -1
            print("Start ", start_node)
            for el in filtered_array:
                connections = int(tuple(el) in sh_path)
                if connections == 1:
                    connection = tuple(el)
                    print("Connection: ", connection)
                sum += connections
            if connection != -1 and idx != len(edges_nodes) - 2:
                second_rooot_length = nx.shortest_path(G_init, connection, edge)
                df.loc[df['nodes'].isin(second_rooot_length), 'color'] = "blue"
                # df.loc[df['nodes'] == connection, 'color'] = "red"
                # df.loc[df['nodes'] == edge, 'color'] = colors[color_idx]
                start_node = tuple(connection)
            print("Length: ", len(second_rooot_length), " - ")
            print("End: ", edge)
            if idx > 0:
                sum -= 1
            if sum == 0:
                main_rooot_length = nx.shortest_path(G_init, edges_nodes[0], edge)
                print(f"Main root length is {len(main_rooot_length)}")
                df.loc[df['nodes'].isin(main_rooot_length), 'color'] = init_color
                # df.loc[df['nodes'] == edge, 'color'] = init_color
                # df.loc[df['nodes'] == edge, 'color'] = init_color
            print("Sum: ", sum)
        except Exception:
            pass

    return df


def measure(G, is_naive=False):
    """
    Measures the main and lateral root lengths using either a naive or the custom algorithm developed for this project
    to find the main root.

    :param G: (networkx.Graph) object representing the root.
    :param is_naive: (bool) parameter specifying the type of algorithm that should be used to measure the plants.
    :return:
     (pandas.DataFrame) object containing the information for the main and lateral roots with their color
     representation.
    """
    nodes = dict(G.degree())

    filtered_array = get_connections(nodes.items())

    edges_nodes = get_edges(nodes.items())

    data = {"nodes": G.nodes}
    if len(G.nodes) == 0:
        return -1
    df = pd.DataFrame(data)
    df['color'] = "yellow"
    df['x'], df['y'] = list(zip(*df['nodes']))
    if is_naive:
        return morphometric_naive(df, G, edges_nodes, filtered_array)
    else:
        return morphometric(df, G, edges_nodes, filtered_array)

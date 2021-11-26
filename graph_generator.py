import random as rnd

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def generateRandomCenters(n, limits, spacing):
    centers = []

    while (len(centers) != n):
        # x are for columns
        # y are for rows
        center_x = rnd.randint(2, limits[0] - 2)
        center_y = rnd.randint(2, limits[1] - 2)

        # If its the first center
        if np.size(centers) == 0:
            centers.append([center_x, center_y])

        # Calculate distance between points
        flag = True
        for element in centers:
            dist = int(np.floor(np.sqrt((center_x - element[0]) ** 2 + (center_y - element[1]) ** 2)))
            # If for some element spacing is not valid break check out
            if dist <= spacing:
                flag = False
                break
        if flag == True:
            centers.append(([center_x, center_y]))

    return centers


def generateAdjacencyCA(centers, grid_shape):
    A = np.zeros((grid_shape[0], grid_shape[1]), dtype=int)

    for center in centers:
        A[center[0], center[1]] = 1

    return A


def generateAdjacencyGraph(centers, prob=0.2):
    n_nodes = len(centers)
    A = np.zeros((n_nodes, n_nodes), dtype=int)

    # Create Adjacency Matrix with only disconnected nodes
    for i in range(0, n_nodes):
        A[i][i] = 1
        for j in range(i + 1, n_nodes):
            r = rnd.random()
            if r <= prob:
                A[i][j] = 1
    return A


def show_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, with_labels=True)
    plt.show()


def createEdges(CAGrid, centers, Adjacency, n_nodes):
    edges = []
    edge_centers = []

    # Get Edge indexes from Adjacency
    for i in range(0, n_nodes):
        for j in range(i + 1, n_nodes):
            if Adjacency[i][j] == 1:
                edges.append([i, j])

    # Get centers of edges in Grid
    for edge in edges:
        edge_centers.append([centers[edge[0]], centers[edge[1]]])

    print(edge_centers)
    for edge in edge_centers:
        print(edge)
        x_direction = 0
        y_direction = 0

        # Calculate Distance between nodes in columns
        x_dist = edge[0][0] - edge[1][0]
        if x_dist >= 0:
            x_direction = 1
        else:
            x_direction = -1

        # Calculate Distance between nodes in rows
        y_dist = edge[0][1] - edge[1][1]
        if y_dist >= 0:
            y_direction = -1
        else:
            y_direction = 1

        print('x_direction: {}'.format(x_direction))
        print('y_direction: {}'.format(y_direction))

        print('x_dist: {}'.format(x_dist))
        print('y_dist: {}'.format(y_dist))

        x_ratio = np.abs(x_dist / y_dist)
        y_ratio = np.abs(y_dist / x_dist)

        if np.abs(x_dist) >= np.abs(y_dist):
            x_ratio = 1
        else:
            y_ratio = 1

        max_dist = np.maximum(np.abs(x_dist), np.abs(y_dist))

        x = edge[0][0]
        y = edge[0][1]
        x_buffer = 0
        y_buffer = 0
        print('x_ratio: {}'.format(x_ratio))
        print('y_ratio: {}'.format(y_ratio))
        print('max_dist: {}'.format(max_dist))
        for i in range(1, max_dist):
            # Start cumulative buffers
            if x_ratio < 1:
                x_buffer += x_ratio
            else:
                x_buffer = 1

            if y_ratio < 1:
                y_buffer += y_ratio
            else:
                y_buffer = 1

            print('buffer x: {}'.format(x_buffer))
            print('buffer y: {}'.format(y_buffer))

            # New coordinates
            if x_direction > 0:
                x = int(x + np.floor(x_buffer * x_direction))
            else:
                x = int(x + np.ceil(x_buffer * x_direction))

            if y_direction > 0:
                y = int(y + np.floor(y_buffer * y_direction))
            else:
                y = int(y + np.ceil(y_buffer * y_direction))
            print('new x: {}'.format(x))
            print('new y: {}'.format(y))

            # Construct tree path in Grid
            CAGrid[x - 1][y - 1] = 3

            # Reset Buffer after make a step
            if x_buffer > 1:
                x_buffer = 0
            if y_buffer > 1:
                y_buffer = 0

    print(CAGrid)


if __name__ == '__main__':
    n_centers = 10
    grid_shape = [64, 64]
    spacing = 8

    # Generate Random Spacing Centers
    points = generateRandomCenters(n_centers, grid_shape, spacing)

    # Generate the adjacency Matrix for CA
    CAMatrix = generateAdjacencyCA(points, grid_shape)

    # Generate the adjacency Matrix for Graph
    GraphMatrix = generateAdjacencyGraph(points)

    # Create Edges In Grid
    createEdges(CAMatrix, points, GraphMatrix, n_centers)

    # Create Labels for nodes
    node_labels = list()
    for i in range(0, n_centers):
        node_labels.append(str(i))
    print(node_labels)

    # Save Matrix as a CSV file
    np.savetxt("Adj_Matrix_Graph.csv", GraphMatrix, delimiter=",")
    new_Matrix = np.loadtxt("Adj_Matrix_Graph.csv", delimiter=",")

    xs = [x[0] for x in points]
    ys = [x[1] for x in points]
    plt.plot(xs, ys, 'o', color='black');
    plt.show()

    show_graph_with_labels(GraphMatrix, node_labels)

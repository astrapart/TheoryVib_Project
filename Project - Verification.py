import numpy as np
from math import radians

n_elem = 3
def create_node_elem(n_elem) :
    elem = np.array([[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0],
                     [5, 9, 0], [6, 10, 0], [7, 11, 0], [8, 12, 0],
                     [9, 13, 0], [10, 14, 0], [11, 15, 0], [12, 16, 0],
                     [13, 17, 0], [14, 18, 0], [15, 19, 0], [16, 20, 0],
                     [5, 6, 1], [5, 7, 1], [6, 8, 1], [8, 7, 1],
                     [9, 10, 1], [9, 11, 1], [10, 12, 1], [12, 11, 1],
                     [13, 14, 1], [13, 15, 1], [14, 16, 1], [16, 15, 1],
                     [17, 18, 1], [17, 19, 1], [18, 20, 1], [20, 19, 1],
                     [9, 6, 1], [6, 12, 1], [12, 7, 1], [7, 9, 1],
                     [9, 14, 1], [14, 12, 1], [12, 15, 1], [15, 9, 1],
                     [17, 15, 1], [15, 20, 1], [20, 14, 1], [14, 17, 1],
                     [17, 21, 2], [18, 21, 2], [19, 21, 2], [20, 21, 2], [21, 22, 2]])

    z_values = np.array([0, 1, 9, 17, 25])

    node = np.empty((0, 3))

    for z in z_values:
        floor_nodes = np.array([
            [z * np.tan(radians(3)), z * np.tan(radians(3)), z],
            [5 - z * np.tan(radians(3)), z * np.tan(radians(3)), z],
            [z * np.tan(radians(3)), 5 - z * np.tan(radians(3)), z],
            [5 - z * np.tan(radians(3)), 5 - z * np.tan(radians(3)), z]
        ])
        # Add floor_nodes to the global node array
        node = np.vstack((node, floor_nodes))

      # Add the last two nodes
    nodeList = np.vstack((node, np.array([[2.5, 2.5, 25], [2.5, 2.5, 80]])))

    n_nodes = len(node)
    n_elem_tot = len(elem)

    elemList = np.empty((0, 3), dtype=int)
    nodeList = node
    next_node = n_nodes + 1

    for i in range(n_elem_tot):
        current_elem = elem[i, :]
        print(current_elem)

        if (n_elem < 2 or current_elem[1] == 21 or current_elem[0] == 21):  # rigid beams : only one element
            new_elem = current_elem

        else:
            new_elem = [current_elem[0], next_node]

            for j in range(1, n_elem - 1):
                new_elem = np.vstack((new_elem, [next_node, next_node + 1]))

                next_node += 1

                x = node[current_elem[0] - 1, 0] + (
                            node[current_elem[1] - 1, 0] - node[current_elem[0] - 1, 0]) * j / n_elem
                y = node[current_elem[0] - 1, 1] + (
                            node[current_elem[1] - 1, 1] - node[current_elem[0] - 1, 1]) * j / n_elem
                z = node[current_elem[0] - 1, 2] + (
                            node[current_elem[1] - 1, 2] - node[current_elem[0] - 1, 2]) * j / n_elem
                nodeList = np.vstack((nodeList, [x, y, z]))

            new_elem = np.vstack((new_elem, [next_node, current_elem[1]]))
            next_node += 1

            x = node[current_elem[0] - 1, 0] + (node[current_elem[1] - 1, 0] - node[current_elem[0] - 1, 0]) * (
                        n_elem - 1) / n_elem
            y = node[current_elem[0] - 1, 1] + (node[current_elem[1] - 1, 1] - node[current_elem[0] - 1, 1]) * (
                        n_elem - 1) / n_elem
            z = node[current_elem[0] - 1, 2] + (node[current_elem[1] - 1, 2] - node[current_elem[0] - 1, 2]) * (
                        n_elem - 1) / n_elem
            nodeList = np.vstack((nodeList, [x, y, z]))

        elemList = np.vstack((elemList, new_elem))

    return elemList,nodeList

print(create_node_elem(3))
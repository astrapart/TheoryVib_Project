import data
import numpy as np
from math import radians
import fct
from scipy.linalg import eigvals

def FE(n_elem):
    elem = np.array([[1, 5], [2, 6], [3, 7], [4, 8],  # main beams
                     [5, 9], [6, 10], [7, 11], [8, 12],  # main beams
                     [9, 13], [10, 14], [11, 15], [12, 16],  # main beams
                     [13, 17], [14, 18], [15, 19], [16, 20],  # main beams
                     [5, 6], [6, 7], [7, 8], [8, 5],  # secondary horizontal beams
                     [9, 10], [10, 11], [11, 12], [12, 9],  # secondary horizontal beams
                     [13, 14], [14, 15], [15, 16], [16, 13],  # secondary horizontal beams
                     [17, 18], [18, 19], [19, 20], [20, 17],  # secondary horizontal beams
                     [6, 9], [6, 11], [8, 9], [8, 11],  # secondary diagonal beams
                     [9, 14], [9, 16], [11, 14], [11, 16],  # secondary diagonal beams
                     [14, 17], [14, 19], [16, 17], [16, 19],  # secondary diagonal beams
                     [17, 21], [18, 21], [19, 21], [20, 21], [21, 22]  # rigid beams
                     ])

    elem10011 = np.array([[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0],  # main beams
                          [5, 9, 0], [6, 10, 0], [7, 11, 0], [8, 12, 0],  # main beams
                          [9, 13, 0], [10, 14, 0], [11, 15, 0], [12, 16, 0],  # main beams
                          [13, 17, 0], [14, 18, 0], [15, 19, 0], [16, 20, 0],  # main beams
                          [5, 6, 1], [6, 7, 1], [7, 8, 1], [8, 5, 1],  # secondary horizontal beams
                          [9, 10, 1], [10, 11, 1], [11, 12, 1], [12, 9, 1],  # secondary horizontal beams
                          [13, 14, 1], [14, 15, 1], [15, 16, 1], [16, 13, 1],  # secondary horizontal beams
                          [17, 18, 1], [18, 19, 1], [19, 20, 1], [20, 17, 1],  # secondary horizontal beams
                          [6, 9, 1], [6, 11, 1], [8, 9, 1], [8, 11, 1],  # secondary diagonal beams
                          [9, 14, 1], [9, 16, 1], [11, 14, 1], [11, 16, 1],  # secondary diagonal beams
                          [14, 17, 1], [14, 19, 1], [16, 17, 1], [16, 19, 1],  # secondary diagonal beams
                          [17, 21, 2], [18, 21, 2], [19, 21, 2], [20, 21, 2], [21, 22, 2]])  # rigid beams
    elem_type = np.array(['m', 'm', 'm', 'm',  # main beams
                          'm', 'm', 'm', 'm',  # main beams
                          'm', 'm', 'm', 'm',  # main beams
                          'm', 'm', 'm', 'm',  # main beams
                          's', 's', 's', 's',  # secondary horizontal beams
                          's', 's', 's', 's',  # secondary horizontal beams
                          's', 's', 's', 's',  # secondary horizontal beams
                          's', 's', 's', 's',  # secondary horizontal beams
                          's', 's', 's', 's',  # secondary diagonal beams
                          's', 's', 's', 's',  # secondary diagonal beams
                          's', 's', 's', 's',  # secondary diagonal beams
                          'r', 'r', 'r', 'r', 'r'  # rigid beams
                          ])

    # Define z_values array
    z_values = np.array([0, 1, 9, 17, 25])

    # Initialize an empty array for node coordinates
    node = np.empty((0, 3))

    # Loop to generate coordinates for each floor
    for z in z_values:
        floor_nodes = np.array([
            [z * np.tan(radians(3)), z * np.tan(radians(3)), z],
            [5 - z * np.tan(radians(3)), z * np.tan(radians(3)), z],
            [5 - z * np.tan(radians(3)), 5 - z * np.tan(radians(3)), z],
            [z * np.tan(radians(3)), 5 - z * np.tan(radians(3)), z]
        ])

        # Add floor_nodes to the global node array
        node = np.vstack((node, floor_nodes))

    # Add the last two nodes
    node = np.vstack((node, np.array([[2.5, 2.5, 25], [2.5, 2.5, 80]])))

    n_nodes = len(node)
    n_elem_tot = len(elem)

    elemList = np.empty((0, 2), dtype=int)
    elemTypeList = np.empty(0)
    nodeList = node
    next_node = n_nodes + 1

    for i in range(n_elem_tot):
        current_elem = elem[i, :]
        current_elem_type = elem_type[i]

        if (n_elem < 2 or current_elem[1] == 21 or current_elem[0] == 21):  # rigid beams : only one element
            new_elem = current_elem
            new_elem_type = current_elem_type

        else:
            new_elem = [current_elem[0], next_node]
            new_elem_type = current_elem_type

            for j in range(1, n_elem - 1):
                new_elem = np.vstack((new_elem, [next_node, next_node + 1]))
                new_elem_type = np.append(new_elem_type, current_elem_type)

                next_node += 1

                x = node[current_elem[0] - 1, 0] + (
                            node[current_elem[1] - 1, 0] - node[current_elem[0] - 1, 0]) * j / n_elem
                y = node[current_elem[0] - 1, 1] + (
                            node[current_elem[1] - 1, 1] - node[current_elem[0] - 1, 1]) * j / n_elem
                z = node[current_elem[0] - 1, 2] + (
                            node[current_elem[1] - 1, 2] - node[current_elem[0] - 1, 2]) * j / n_elem
                nodeList = np.vstack((nodeList, [x, y, z]))

            new_elem = np.vstack((new_elem, [next_node, current_elem[1]]))
            new_elem_type = np.append(new_elem_type, current_elem_type)
            next_node += 1

            x = node[current_elem[0] - 1, 0] + (node[current_elem[1] - 1, 0] - node[current_elem[0] - 1, 0]) * (
                        n_elem - 1) / n_elem
            y = node[current_elem[0] - 1, 1] + (node[current_elem[1] - 1, 1] - node[current_elem[0] - 1, 1]) * (
                        n_elem - 1) / n_elem
            z = node[current_elem[0] - 1, 2] + (node[current_elem[1] - 1, 2] - node[current_elem[0] - 1, 2]) * (
                        n_elem - 1) / n_elem
            nodeList = np.vstack((nodeList, [x, y, z]))

        elemList = np.vstack((elemList, new_elem))
        elemTypeList = np.append(elemTypeList, new_elem_type)

    n_nodes = len(nodeList)
    dofList = np.zeros((n_nodes, 6), dtype=int)

    for i in range(n_nodes):
        for j in range(6):
            dofList[i, j] = 6 * i + j + 1

    n_elem_tot = len(elemList)
    locel = np.zeros((n_elem_tot, 12), dtype=int)

    for i in range(n_elem_tot):
        locel[i, 0:6] = dofList[elemList[i][0] - 1]
        locel[i, 6:12] = dofList[elemList[i][1] - 1]

    return elemList, dofList, nodeList, locel, elemTypeList

def calculate_length(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)

def properties(type_beam, l):
    rho = data.density_beam                                     # [kg/m3]
    v = data.poisson_ratio                                      # [-]
    E = data.young_mod                                          # [Pa]
    D = data.diam_beam[type_beam]                               # [m]
    A = np.pi * (D*D - (D - 2 * data.thickness_beam) ** 2)/4    # [m2]
    Ix = (np.pi / 64) * (D ** 4 - (D - 2 * data.thickness_beam) ** 4)  # [m4]
    Jx = Ix * 2  # [m4]
    Iy = Ix  # [m4]
    Iz = Iy  # [m4]

    if type_beam == 2:
        rho = rho * 10 ** (-4)
        A = A * 10 ** (-2)
        E = E * 10 ** 4
        Jx = Jx*10**4
        Iy = Iy*10**4
        Iz = Iz*10**4

    m = rho * A * l                         # [kg]
    G = E / (2 * (1 + v))                   # [GPa]
    r = np.sqrt(Jx / A)                     # [m]
    return rho, v, E, A, m, Jx, Iy, Iz, G, r


elemList, dofList, nodeList, locel, elemTypeList = FE(3)


nodeLumped = 22
M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))



for i in range(len(elemList)):
    node1 = elemList[i][0]-1
    node2 = elemList[i][1]-1
    if elemTypeList[i] == 'm' :
        type_beam = 0
    if elemTypeList[i] == 's' :
        type_beam = 1
    if elemTypeList[i] == 'r' :
        type_beam = 2

    coord1 = nodeList[node1]
    coord2 = nodeList[node2]
    l = calculate_length(coord1, coord2)

    rho, v, E, A, m, Jx, Iy, Iz, G, r = properties(type_beam, l)

    Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
    Mel = fct.create_Mel(m, r, l)

    T = fct.create_T(coord1, coord2, l)

    Kes = np.transpose(T) @ Kel @ T
    Mes = np.transpose(T) @ Mel @ T

    for j in range(len(locel[i])):
        for k in range(len(locel[i])):
            M[locel[i][j] - 1][locel[i][k] - 1] += Mes[j][k]
            K[locel[i][j] - 1][locel[i][k] - 1] += Kes[j][k]

fct.Add_lumped_mass(nodeLumped, dofList, M)

M, K = fct.Add_const_emboit(dofList, M, K)
"""
# Boundary conditions : Nodes 1 to 4 are clamped
clamped_dof = np.concatenate((dofList[0], dofList[1], dofList[2], dofList[3]))
clamped_dof = np.sort(clamped_dof)[::-1]

K = np.delete(K, clamped_dof-1, axis=0)
K = np.delete(K, clamped_dof-1, axis=1)
M = np.delete(M, clamped_dof-1, axis=0)
M = np.delete(M, clamped_dof-1, axis=1)

"""



Freq = eigvals(K, M)
Freq = np.sort(np.real(Freq))
Freq_Hz = np.sqrt(Freq) / (2 * np.pi)

print(Freq_Hz[:8])


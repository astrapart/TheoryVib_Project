import matplotlib.pyplot as plt
import numpy as np
import scipy
import fct
import copy

a = 5.49
b = 3.66
nodeListsimple = [[0, 0, 0],    # node 0
                  [a, 0, 0],    # node 1
                  [0, a, 0],    # node 2
                  [a, a, 0],    # node 3
                  [0, 0, b],    # node 4
                  [a, 0, b],    # node 5
                  [0, a, b],    # node 6
                  [a, a, b],    # node 7
                  [0, 0, 2*b],  # node 8
                  [a, 0, 2*b],  # node 9
                  [0, a, 2*b],  # node 10
                  [a, a, 2*b]]  # node 11

elemList0simple = [[0, 4, 0], [1, 5, 0], [2, 6, 0], [3, 7, 0],
                   [4, 8, 0], [5, 9, 0], [6, 10, 0], [7, 11, 0],
                   [4, 5, 1], [5, 7, 1], [7, 6, 1], [6, 4, 1],
                   [8, 9, 1], [9, 11, 1], [11, 10, 1], [10, 8, 1]]

#fct.plot(elemList0simple, nodeListsimple)

# proprieties = [rho [kg/m3], poisson [-], Young [Pa], Jx [m4], Iy [m4], Iz [m4]]
vertical_beams = [7800, 0.3, 211e9, 5.14e-3, 1.73e-7, 6.9e-6, 8.49e-5]
horizontal_beams = [7800, 0.3, 211e9, 5.68e-3, 1.76e-7, 1.2e-4, 7.3e-6]

proprieties = [vertical_beams, horizontal_beams]

numberElem = 2

elemList = fct.create_elemList(elemList0simple, nodeListsimple, numberElem)
dofList = fct.create_dofList(nodeListsimple)
locel = fct.create_locel(elemList, dofList)

nodeConstraint = [0, 1, 2, 3]

numberNode = len(nodeListsimple)
M = np.zeros([numberNode*6, numberNode*6])
K = np.zeros([numberNode*6, numberNode*6])

for i in range(len(elemList)):

    node1 = elemList[i][0]
    node2 = elemList[i][1]
    indexProp = elemList[i][2]

    coord1 = nodeListsimple[node1]
    coord2 = nodeListsimple[node2]

    l = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

    propriety = proprieties[indexProp]

    rho = propriety[0]     # [kg/m3]
    v = propriety[1]       # [-]
    E = propriety[2]       # [Pa]
    A = propriety[3]       # [m2]
    Jx = propriety[4]      # [m4]
    Iy = propriety[5]      # [m4]
    Iz = propriety[6]      # [m4]
    G = E / (2 * (1 + v))  # [Pa]
    r = np.sqrt(Iy/A)      # [m]
    m = rho * l * A        # [kg]

    Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
    Mel = fct.create_Mel(m, r, l)

    T = fct.create_T(coord1, coord2, l)

    Kes = T.T @ Kel @ T
    Mes = T.T @ Mel @ T

    for j in range(len(locel[i])):
        for k in range(len(locel[i])):

            M[locel[i][j]][locel[i][k]] = M[locel[i][j]][locel[i][k]] + Mes[j][k]

            K[locel[i][j]][locel[i][k]] = K[locel[i][j]][locel[i][k]] + Kes[j][k]

fct.Add_const_emboit(nodeConstraint, dofList, M, K)

eigenvals, eigenvects = scipy.linalg.eigh(K, M)

print(sorted(eigenvals)[:8])
index_sort = np.argsort(eigenvals)

fig = plt.figure()
newNodeList = []
for i in range(8):
    for j in range(len(nodeListsimple)):
        coord = nodeListsimple[j]
        if j not in nodeConstraint:

            dx, dy, dz = eigenvects[i][6*j], eigenvects[i][6*j+1], eigenvects[i][6*j+2]

            new_coord = [coord[0]+dx, coord[1]+dy, coord[2]+dz]
            newNodeList.append(new_coord)
        else:
            newNodeList.append(coord)

    ax = fig.add_subplot(2, 4, i+1, projection='3d')

    for elem in elemList0simple:
        newnode1 = newNodeList[elem[0]]
        newnode2 = newNodeList[elem[1]]
        node1 = nodeListsimple[elem[0]]
        node2 = nodeListsimple[elem[1]]

        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]],'--', c='b')
        ax.plot([newnode1[0], newnode2[0]], [newnode1[1], newnode2[1]], [newnode1[2], newnode2[2]], c='r')


plt.show()
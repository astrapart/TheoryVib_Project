"""
########################################################################################################################
IMPORT
########################################################################################################################
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import data

"""
########################################################################################################################
Fonction create
########################################################################################################################
"""


def create_elemList(elemList0, nodeList, numberElem):
    elemList = []

    for elem in elemList0:
        i = elem[0]-1
        j = elem[1]-1
        propriety = elem[2]

        if propriety != 2:
            current = i + 1
            len_x = abs(nodeList[i][0] - nodeList[j][0]) / numberElem
            if nodeList[i][0] > nodeList[j][0]:
                len_x *= -1
            len_y = abs(nodeList[i][1] - nodeList[j][1]) / numberElem
            if nodeList[i][1] > nodeList[j][1]:
                len_y *= -1
            len_z = abs(nodeList[i][2] - nodeList[j][2]) / numberElem
            if nodeList[i][2] > nodeList[j][2]:
                len_z *= -1

            for m in range(numberElem):
                new = len(nodeList) + 1
                if m != (numberElem - 2):
                    elemList.append([current, new, propriety])
                    nodeList.append([nodeList[current-1][0] + len_x, nodeList[current-1][1] + len_y,
                                 nodeList[current-1][2] + len_z])

                    current = new
                else:
                    elemList.append([new, j+1, propriety])
        else:
            elemList.append(elem)

    return np.array(elemList)

    
def create_dofList(nodeList):
    dofList = []
    dof = 1
    for i in range(len(nodeList)):
        tmp = []
        for j in range(6):
            tmp.append(dof)
            dof += 1
        dofList.append(tmp)
    return np.array(dofList)


def create_locel(elemList, dofList):
    locel = []
    for i in range(len(elemList)):
        dofNode1 = dofList[elemList[i][0]-1]
        dofNode2 = dofList[elemList[i][1]-1]
        locel.append(np.concatenate((dofNode1, dofNode2), axis=0))
    
    return np.array(locel)

def create_T(coord1, coord2, l):
    P1 = np.array(coord1)
    P2 = np.array(coord2)
    P3 = np.array([1,1,1])

    d2 = P2 - P1
    d3 = P3 - P1

    ex = d2 / l
    ey = np.cross(d3, d2) / np.linalg.norm(np.cross(d3, d2))
    ez = np.cross(ex, ey)

    eX = [1, 0, 0]
    eY = [0, 1, 0]
    eZ = [0, 0, 1]

    R = [[np.dot(eX, ex), np.dot(eY, ex), np.dot(eZ, ex)],
         [np.dot(eX, ey), np.dot(eY, ey), np.dot(eZ, ey)],
         [np.dot(eX, ez), np.dot(eY, ez), np.dot(eZ, ez)]]

    T = scipy.linalg.block_diag(R, R, R, R)

    return np.array(T)


def create_Kel(E, A, Jx, Iy, Iz, G, l) :
    Kel = [[E*A/l],
           [  0, 12*E*Iz/(l*l*l)],
           [  0,       0,       12*E*Iy/(l*l*l)],
           [  0,       0,              0,       G*Jx/l],
           [  0,       0,        -6*E*Iy/(l*l),    0,   4*E*Iy/l],
           [  0,   6*E*Iz/(l*l),       0,          0,       0,     4*E*Iz/l],
           [-E*A/l,    0,              0,          0,       0,         0,         E*A/l],
           [  0, -12*E*Iz/(l*l*l),     0,          0,       0,   -6*E*Iz/(l*l),     0,     12*E*Iz/(l*l*l)],
           [  0,       0,       -12*E*Iy/(l*l*l),  0, 6*E*Iy/(l*l),    0,           0,            0,        12*E*Iy/(l*l*l)],
           [  0,       0,              0,       -G*Jx/l,    0,         0,           0,            0,               0,          G*Jx/l],
           [  0,       0,         -6*E*Iy/(l*l),   0,    2*E*Iy/l,     0,           0,            0,          6*E*Iy/(l*l),       0,   4*E*Iy/l],
           [  0,   6*E*Iz/(l*l),       0,          0,       0,     2*E*Iz/l,        0,      -6*E*Iz/(l*l),         0,             0,       0,    4*E*Iz/l]]

    for i in range(len(Kel)):
        for j in range(i+1, len(Kel)):

            Kel[i].append(Kel[j][i])

    return np.array(Kel)


def create_Mel(m, r, l) :

    Mel = [[1/3],
           [0, 13/35],
           [0, 0, 13/35],
           [0, 0, 0, r*r/3],
           [0, 0, -11*l/210, 0, l*l/105],
           [0, 11*l/210, 0, 0, 0, l*l/105],
           [1/6, 0, 0, 0, 0, 0, 1/3],
           [0, 9/70, 0, 0, 0, 13*l/420, 0, 13/35],
           [0, 0, 9/70, 0, -13*l/420, 0, 0, 0, 13/35],
           [0, 0, 0, r*r/6, 0, 0, 0, 0, 0, r*r/3],
           [0, 0, 13*l/420, 0, -l*l/140, 0, 0, 0, 11*l/210, 0, l*l/105],
           [0, -13*l/420, 0, 0, 0, -l*l/140, 0, -11*l/210, 0, 0, 0, l*l/105]]

    for i in range(len(Mel)):
        for j in range(i+1, len(Mel)):
            Mel[i].append(Mel[j][i])

    return m * np.array(Mel)


def Kel(A, l, E, I, J, G):
    K_el = np.zeros((12,12))

    K_el[(0,0)] = E*A/l
    K_el[(1,1)] = 12*E*I/l**3
    K_el[(2,2)] = 12*E*I/l**3
    K_el[(3,3)] = G*J/l
    K_el[(4,4)] = 4*E*I/l
    K_el[(5,5)] = 4*E*I/l
    K_el[(6,6)] = E*A/l
    K_el[(7,7)] = 12*E*I/l**3
    K_el[(8,8)] = 12*E*I/l**3
    K_el[(9,9)] = G*J/l
    K_el[(10,10)] = 4*E*I/l
    K_el[(11,11)] = 4*E*I/l
    K_el[(0,6)] = K_el[(6,0)] = -E*A/l
    K_el[(1,5)] = K_el[(5,1)] = 6*E*I/l**2
    K_el[(1,7)] = K_el[(7,1)] = -12*E*I/l**3
    K_el[(1,11)] = K_el[(11,1)] = 6*E*I/l**2
    K_el[(2,4)] = K_el[(4,2)] = -6*E*I/l**2
    K_el[(2,8)] = K_el[(8,2)] = -12*E*I/l**3
    K_el[(2,10)] = K_el[(10,2)] = -6*E*I/l**2
    K_el[(3,9)] = K_el[(9,3)] = -G*J/l
    K_el[(4,8)] = K_el[(8,4)] = 6*E*I/l**2
    K_el[(4,10)] = K_el[(10,4)] = 2*E*I/l
    K_el[(5,7)] = K_el[(7,5)] = -6*E*I/l**2
    K_el[(5,11)] = K_el[(11,5)] = 2*E*I/l
    K_el[(7,11)] = K_el[(11,7)] = -6*E*I/l**2
    K_el[(8,10)] = K_el[(10,8)] = 6*E*I/l**2

    return K_el


def Mel(A, l, r2, rho):
    M_el = np.zeros((12,12))

    M_el[(0,0)] = 1/3
    M_el[(1,1)] = 13/35
    M_el[(2,2)] = 13/35
    M_el[(3,3)] = r2/3
    M_el[(4,4)] = l**2/105
    M_el[(5,5)] = l**2/105
    M_el[(6,6)] = 1/3
    M_el[(7,7)] = 13/35
    M_el[(8,8)] = 13/35
    M_el[(9,9)] = r2/3
    M_el[(10,10)] = l**2/105
    M_el[(11,11)] = l**2/105
    M_el[(0,6)] = M_el[(6,0)] = 1/6
    M_el[(1,5)] = M_el[(5,1)] = 11*l/210
    M_el[(1,7)] = M_el[(7,1)] = 9/70
    M_el[(1,11)] = M_el[(11,1)] = -13*l/420
    M_el[(2,4)] = M_el[(4,2)] = -11*l/210
    M_el[(2,8)] = M_el[(8,2)] = 9/70
    M_el[(2,10)] = M_el[(10,2)] = 13*l/420
    M_el[(3,9)] = M_el[(9,3)] = r2/6
    M_el[(4,8)] = M_el[(8,4)] = -13*l/420
    M_el[(4,10)] = M_el[(10,4)] = -l**2/140
    M_el[(5,7)] = M_el[(7,5)] = 13*l/420
    M_el[(5,11)] = M_el[(11,5)] = -l**2/140
    M_el[(7,11)] = M_el[(11,7)] = -11*l/210
    M_el[(8,10)] = M_el[(10,8)] = 11*l/210
    M_el = rho*A*l*M_el

    return M_el


def Add_const_emboit(nodeConstraint, dofList, M, K):

    clamped_dof = np.concatenate((dofList[0], dofList[1], dofList[2], dofList[3]))
    clamped_dof = np.sort(clamped_dof)[::-1]

    K = np.delete(K, clamped_dof -1, axis=0)
    K = np.delete(K, clamped_dof -1, axis=1)
    M = np.delete(M, clamped_dof -1, axis=0)
    M = np.delete(M, clamped_dof -1, axis=1)
    """
    for node in nodeConstraint:
        for tmp in dofList[node-1]:
            M = np.delete(M, tmp-1, 0)
            M = np.delete(M, tmp-1, 1)

            K = np.delete(K, tmp-1, 0)
            K = np.delete(K, tmp-1, 1)
    """
    return M, K


def Add_lumped_mass(nodeLumped, dofList, M):
    mass = data.mass_lumped
    J = data.node_lumped_J

    count = 0
    for tmp in dofList[nodeLumped-1]:
        i = tmp-1
        if count <= 2:
            M[i][i] += mass
        else:
            M[i][i] += J
        count += 1
    return M

"""
########################################################################################################################
Fonction Calculate
########################################################################################################################
"""


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


def calculate_mtot_rigid(M):

    ue = np.zeros(len(M))
    for i in range(len(M)//6):
        ue[i*6] = 1

    return np.transpose(ue) @ M @ ue

"""
########################################################################################################################
Fonctions PLOT 
########################################################################################################################
"""
def plot_structure(elemList, nodeList):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for elem in elemList:
        elem_1 = nodeList[elem[0]-1]
        elem_2 = nodeList[elem[1]-1]
        ax.plot([elem_1[0], elem_2[0]], [elem_1[1], elem_2[1]], [elem_1[2], elem_2[2]], c='b')

    for node in nodeList:
        x = node[0]
        y = node[1]
        z = node[2]
        ax.scatter(x, y, z, c='g')

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)

    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')

    plt.show()


def plot_result(nodeList, nodeConstraint, eigenvects, elemList0):
    fig = plt.figure()
    for i in range(len(eigenvects)):
        newNodeList = []
        for j in range(len(nodeList)):
            coord = nodeList[j]
            if j+1 not in nodeConstraint:

                dx, dy, dz = eigenvects[i][6 * j], eigenvects[i][6 * j + 1], eigenvects[i][6 * j + 2]

                factor = 5
                new_coord = [coord[0] + dx*factor, coord[1] + dy*factor, coord[2] + dz*factor]
                newNodeList.append(new_coord)
            else:
                newNodeList.append(coord)

        ax = fig.add_subplot(2, 4, i + 1, projection='3d')

        for elem in elemList0:
            if elem[2] != 2:
                newnode1 = newNodeList[elem[0]-1]
                newnode2 = newNodeList[elem[1]-1]
                node1 = nodeList[elem[0]-1]
                node2 = nodeList[elem[1]-1]

                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], '--', c='b')
                ax.plot([newnode1[0], newnode2[0]], [newnode1[1], newnode2[1]], [newnode1[2], newnode2[2]], c='r')

    plt.show()


def print_freq(list_eign):

    for i in range(len(list_eign)):
        f = np.real(np.sqrt(list_eign[i]))/(2*np.pi)
        #print("La fréquence pour la valeur propre", i, "vaut :", f, "Hz")
        # 0.44 valeur propre 1
        # 0.45 valeur propre 2
        # 0.9  valeur propre 3
        print("La fréquence pour la valeur propre {index} vaut : {val:.5f} [Hz]".format(index=i, val=f))

    return

def print_data_beam(node1, node2, type_beam, rho, v, E, A, m, Jx, Iy, Iz, G, r,l) :
    print("--------Propriety {beam} for elem [{node1}, {node2}]--------".format(beam=type_beam, node1=node1 + 1,
                                                                                node2=node2 + 1))
    print("rho = {rho}, E = {E}  A = {A}  l = {l}".format(rho=rho, E=E, A=A, l=l))
    print("G = {G}  r = {r}  m = {m}".format(G=G, r=r, m=m))
    print("Jx = {Jx}  I = {Iy}".format(Jx=Jx, Iy=Iy))
    print()

def print_matrix(matrix):

    for line in matrix:
        print("      ".join(map(str, line)))

    return
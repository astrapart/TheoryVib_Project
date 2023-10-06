import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import fct

tan_3 = np.tan(np.radians(3))
nodeList = [[0, 0, 0],  # node 1
            [5, 0, 0],  # node 2
            [0, 5, 0],  # node 3
            [5, 5, 0],  # node 4
            [tan_3, tan_3, 1],  # node 5
            [5 - tan_3, tan_3, 1],  # node 6
            [tan_3, 5 - tan_3, 1],  # node 7
            [5 - tan_3, 5 - tan_3, 1],  # node 8
            [9 * tan_3, 9 * tan_3, 9],  # node 9
            [5 - 9 * tan_3, 9 * tan_3, 9],  # node 10
            [9 * tan_3, 5 - 9 * tan_3, 9],  # node 11
            [5 - 9 * tan_3, 5 - 9 * tan_3, 9],  # node 12
            [17 * tan_3, 17 * tan_3, 17],  # node 13
            [5 - 17 * tan_3, 17 * tan_3, 17],  # node 14
            [17 * tan_3, 5 - 17 * tan_3, 17],  # node 15
            [5 - 17 * tan_3, 5 - 17 * tan_3, 17],  # node 16
            [25 * tan_3, 25 * tan_3, 25],  # node 17
            [5 - 25 * tan_3, 25 * tan_3, 25],  # node 18
            [25 * tan_3, 5 - 25 * tan_3, 25],  # node 19
            [5 - 25 * tan_3, 5 - 25 * tan_3, 25]]  # node 20
"""[2.5, 2.5, 25],  # node 21 [2.5, 2.5, 80]]  # node 22"""

elemList0 = [[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0],
             [5, 9, 0], [6, 10, 0], [7, 11, 0], [8, 12, 0],
             [9, 13, 0], [10, 14, 0], [11, 15, 0], [12, 16, 0],
             [13, 17, 0], [14, 18, 0], [15, 19, 0], [16, 20, 0],
             [5, 6, 1], [5, 7, 1], [6, 8, 1], [8, 7, 1],
             [9, 10, 1], [9, 11, 1], [11, 12, 1], [12, 10, 1],
             [13, 14, 1], [13, 15, 1], [15, 16, 1], [16, 14, 1],
             [17, 18, 1], [17, 19, 1], [19, 20, 1], [20, 18, 1],
             [9, 6, 1], [6, 12, 1], [12, 7, 1], [7, 9, 1],
             [14, 9, 1], [9, 15, 1], [15, 12, 1], [12, 14, 1],
             [17, 15, 1], [15, 20, 1], [20, 14, 1], [14, 17, 1]]
"""[17, 21, 2], [18, 21, 2], [19, 21, 2], [20, 21, 2],[21, 22, 2]]"""

numberElem = 3
dof = 1
elemList = fct.create_elemList(elemList0, nodeList, numberElem)
dofList = fct.create_dofList(dof, nodeList, numberElem)
locel = fct.create_locel(elemList,dofList)

nodeConstraint = np.array([1, 2, 3, 4])
nodeLumped = np.array([[22, 200000]])

#fct.plot(elemList, nodeList)

# Define properties
#[densité [kg/m3], poisson [-], young [GPa], air section [m2], Rayon interne [m], Rayon externe[]
# en SI
mainBeam_d = 1  # m
othbeam_d = 0.6  # m
thickn = 0.02  # m
proprieties = fct.create_properties(mainBeam_d, othbeam_d, thickn)

M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))

for i in range(len(elemList)):
    node1 = elemList[i][0] - 1
    node2 = elemList[i][1] - 1
    propriety = elemList[i][2]

    coord1 = nodeList[node1]
    coord2 = nodeList[node2]

    l = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

    prop = proprieties[propriety]
    rho = prop[0]
    v = prop[1]  # [-]
    E = prop[2]  # [GPa]
    A = prop[3]  # [m2]
    Re = prop[4]  # [m]
    Ri = prop[5]  # [m]

    m = prop[0] * prop[3] * l
    Ix = (np.pi / 64) * (Re ** 4 - Ri ** 4)  # [kg*m2]
    Iy = (np.pi / 64) * (Re ** 4 - Ri ** 4)  # [kg*m2]
    Iz = (np.pi / 64) * (Re ** 4 - Ri ** 4)  # [kg*m2]
    Jx = Ix * 2  # [kg*m2]
    G = E / (2 * (1 + v))  # [GPa]
    r = np.sqrt(Iy/A)  # [kg]

    """
    if prop == 2:
        Jx *= 10**4
        Iy *= 10**4
        Iz *= 10**4
    """

    Kel = fct.create_Kel(E,A,Jx,Iy,Iz,G,l)
    Mel = fct.create_Mel(m,r,l)
    T = fct.create_T(coord1,coord2,l)


    Kes = np.dot(np.dot(np.transpose(T), Kel), T)
    Mes = np.dot(np.dot(np.transpose(T), Mel), T)

    if nodeConstraint.__contains__(node1+1):
        for j in range(6):
            for k in range(6):
                Kes[j][k] = 0
                Mes[j][k] = 0

    elif nodeConstraint.__contains__(node2+1):
        for j in range(6):
            for k in range(6):
                Kes[j+6][k+6] = 0
                Mes[j+6][k+6] = 0

    # Assemblage Matrice globale
    for j in range(len(locel[i])):
        for k in range(len(locel[i])):
            M[locel[i][j] - 1][locel[i][k] - 1] = M[locel[i][j] - 1][locel[i][k] - 1] + Mes[j][k]
            K[locel[i][j] - 1][locel[i][k] - 1] = K[locel[i][j] - 1][locel[i][k] - 1] + Kes[j][k]


fct.Add_lumped_mass(nodeLumped, dofList, M)


#fct.Add_const_emboit(nodeConstraint, dofList, M, K)


eigenvals, eigenvects = scipy.linalg.eigh(K, M)
print(sorted(eigenvals)[0:8])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy

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
            [5 - 25 * tan_3, 5 - 25 * tan_3, 25],  # node 20
            ]

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

def plot():
    # Créez une figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for elem in elemList:
        elem_1 = nodeList[elem[0] - 1]
        elem_2 = nodeList[elem[1] - 1]
        ax.plot([elem_1[0], elem_2[0]], [elem_1[1], elem_2[1]], [elem_1[2], elem_2[2]], c='b')

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)

    # Titres des axes
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')

    # Affichez le graphique
    plt.show()


numberElem = 3
numberBeam = len(elemList0)
elemList = []

for x in range(numberBeam):
    i = elemList0[x][0]
    j = elemList0[x][1]
    propriety = elemList0[x][2]

    current = i
    len_x = abs(nodeList[i - 1][0] - nodeList[j - 1][0]) / numberElem
    if nodeList[i - 1][0] > nodeList[j - 1][0]:
        len_x *= -1
    len_y = abs(nodeList[i - 1][1] - nodeList[j - 1][1]) / numberElem
    if nodeList[i - 1][1] > nodeList[j - 1][1]:
        len_y *= -1
    len_z = abs(nodeList[i - 1][2] - nodeList[j - 1][2]) / numberElem
    if nodeList[i - 1][2] > nodeList[j - 1][2]:
        len_z *= -1

    for m in range(numberElem):
        new = len(nodeList) + 1
        if m != (numberElem - 2):
            elemList.append([current, new, propriety])
            nodeList.append([nodeList[current - 1][0] + len_x, nodeList[current - 1][1] + len_y, nodeList[current - 1][2] + len_z,propriety])

            current = new
        else:
            elemList.append([new, j, propriety])


dofList = []
dof = 1
for i in range(len(nodeList * numberElem)):
    tmp = []
    for j in range(6):
        tmp.append(dof)
        dof += 1
    dofList.append(tmp)

locel = []
for i in range(len(elemList)):
    dofNode1 = dofList[elemList[i][0] - 1]
    dofNode2 = dofList[elemList[i][1] - 1]
    locel.append(dofNode1 + dofNode2)

# print(nodeList)
# print(elemList)
# print(dofList)
# print(locel)
# plot()


# Define parameter [densité [kg/m3], poisson [-], young [GPa], air section [m2], Rayon interne [m], Rayon externe[] en SI
mainBeam_d = 1  # m
othbeam_d = 0.6  # m
thickn = 0.02  # m
## Main Beam
main_beam_prop = [7800, 0.3, 210 * (10 ** 6), np.pi * ((mainBeam_d / 2) ** 2 - (mainBeam_d / 2 - thickn) ** 2), mainBeam_d/2, (mainBeam_d - 2*thickn)/2]

## Other Beam
other_beam_prop = [7800, 0.3, 210 * (10 ** 6), np.pi * ((othbeam_d / 2) ** 2 - (othbeam_d / 2 - thickn) ** 2), othbeam_d/2, (othbeam_d - 2*thickn)/2]

## Rigid Link  
rigid_link_prop = [main_beam_prop[0] * 10 ** 4, 0.3, main_beam_prop[2] * 10 ** 4, main_beam_prop[3] * 10 ** -2, mainBeam_d/2, (mainBeam_d - 2*thickn)/2]

proprieties = [main_beam_prop, other_beam_prop, rigid_link_prop]

M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))

for i in range(len(elemList)):
    node1 = elemList[i][0] - 1
    node2 = elemList[i][1] - 1
    propriety = elemList[i][2]

    coord1 = nodeList[node1]
    coord2 = nodeList[node2]

    l = np.sqrt(
        (coord1[0] - coord2[0]) * (coord1[0] - coord2[0]) + (coord1[1] - coord2[1]) * (coord1[1] - coord2[1]) + (
                coord1[2] - coord2[2]) * (coord1[2] - coord2[2]))
    
    prop = proprieties[propriety]
    rho = prop[0]
    v = prop[1] # [-]
    E = prop[2] # [GPa]
    A = prop[3] # [m2]
    Re = prop[4] # [m]
    Ri = prop[5] # [m
    
    #Valeur vérifié avec autre groupe chez qui ca focntionne 
    m = prop[0] * prop[3] * l
    Ix = (np.pi/64)*(Re**4 - Ri**4)
    Iy = (np.pi/64)*(Re**4 - Ri**4)
    Iz = (np.pi/64)*(Re**4 - Ri**4)
    Jx = Ix*2
    G = E/(2*(1+v))
    r = np.sqrt(Iy/A)

    # Elementary stiffness matrix
    Kel = np.array([
        [E * A / l, 0, 0, 0, 0, 0, -E * A / l, 0, 0, 0, 0, 0],
        [0, 12 * E * Iz / l ** 3, 0, 0, 0, 6 * E * Iz / l ** 2, 0, -12 * E * Iz / l ** 3, 0, 0, 0, 6 * E * Iz / l ** 2],
        [0, 0, 12 * E * Iy / l ** 3, 0, -6 * E * Iy / l ** 2, 0, 0, 0, -12 * E * Iy / l ** 3, 0, -6 * E * Iy / l ** 2, 0],
        [0, 0, 0, G * Jx / l, 0, 0, 0, 0, 0, -G * Jx / l, 0, 0],
        [0, 0, -6 * E * Iy / l ** 2, 0, 4 * E * Iy / l, 0, 0, 0, 6 * E * Iy / l ** 2, 0, 2 * E * Iy / l, 0],
        [0, 6 * E * Iz / l ** 2, 0, 0, 0, 4 * E * Iz / l, 0, -6 * E * Iz / l ** 2, 0, 0, 0, 2 * E * Iy / l],
        [-E * A / l, 0, 0, 0, 0, 0, E * A / l, 0, 0, 0, 0, 0],
        [0, -12 * E * Iz / l ** 3, 0, 0, 0, -6 * E * Iz / l ** 2, 0, 12 * E * Iz / l ** 3, 0, 0, 0, -6 * E * Iz / l ** 2],
        [0, 0, -12 * E * Iy / l ** 3, 0, 6 * E * Iy / l ** 2, 0, 0, 0, 12 * E * Iy / l ** 3, 0, 6 * E * Iy / l ** 2, 0],
        [0, 0, 0, -G * Jx / l, 0, 0, 0, 0, 0, G * Jx / l, 0, 0],
        [0, 0, -6 * E * Iy, 0, 2 * E * Iy / l, 0, 0, 0, 6 * E * Iy / l ** 2, 0, 4 * E * Iy / l, 0],
        [0, 6 * E * Iz / l ** 2, 0, 0, 0, 2 * E * Iz / l, 0, -6 * E * Iz / l ** 2, 0, 0, 0, 4 * E * Iz / l]
    ])
    
    # Elementary mass matrix
    Mel = m * np.array([
        [1 / 3, 0, 0, 0, 0, 0, 1 / 6, 0, 0, 0, 0, 0],
        [0, 13 / 35, 0, 0, 0, 11 * l / 210, 0, 9 / 70, 0, 0, 0, -13 * l / 420],
        [0, 0, 13 / 35, 0, -11 * l / 210, 0, 0, 0, 9 / 70, 0, 13 * l / 420, 0],
        [0, 0, 0, r ** 2 / 3, 0, 0, 0, 0, 0, r ** 2 / 6, 0, 0],
        [0, 0, -11 * l / 210, 0, l ** 2 / 105, 0, 0, 0, -13 * l / 420, 0, -l ** 2 / 140, 0],
        [0, 11 * l / 210, 0, 0, 0, l ** 2 / 105, 0, 13 * l / 420, 0, 0, 0, -l ** 2 / 140],
        [1 / 6, 0, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 0, 0],
        [0, 9 / 70, 0, 0, 0, 13 * l / 420, 0, 13 / 35, 0, 0, 0, -11 * l / 210],
        [0, 0, 9 / 70, 0, -13 * l / 420, 0, 0, 0, 13 / 35, 0, 11 * l / 210, 0],
        [0, 0, 0, r ** 2 / 6, 0, 0, 0, 0, 0, r ** 2 / 3, 0, 0],
        [0, 0, 13 * l / 420, 0, -l ** 2 / 140, 0, 0, 0, 11 * l / 210, 0, l ** 2 / 105, 0],
        [0, -13 * l / 420, 0, 0, 0, -l ** 2 / 140, 0, -11 * l / 210, 0, 0, 0, l ** 2 / 105]
    ])

    P1 = coord1
    P2 = coord2
    P3 = [0.5, 0.5, 0]

    d2 = [P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]]
    d3 = [P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]]


    ex = [(P2[0]-P1[0])/l, (P2[1]-P1[1])/l, (P2[2]-P1[2])/l]
    ey = np.cross(d2, d3)/np.linalg.norm(np.cross(d2, d3))
    ez = np.cross(ex, ey)
    localAxe = [ex, ey, ez]

    eX = [1, 0, 0]
    eY = [0, 1, 0]
    eZ = [0, 0, 1]
    globalAxe = [eX, eY, eZ]

    R = [[], [], []]

    for j in range(3):
        for k in range(3):
            # Je crois qu'il faut inverser k et j => a vérifier
            R[j].append(np.dot(globalAxe[k], localAxe[j]))


    T = np.zeros((12, 12))
    for j in range(3):
        for k in range(3):
            T[j][k] = R[j][k]
            T[j + 3][k + 3] = R[j][k]
            T[j + 6][k + 6] = R[j][k]
            T[j + 9][k + 9] = R[j][k]
            
    Kes = np.dot(np.dot(np.transpose(T), Kel), T)
    Mes = np.dot(np.dot(np.transpose(T), Mel), T)

    #Assemblage Matrice globale
    for j in range(len(locel[i])):
        for k in range(len(locel[i])):

            M[locel[i][j]-1][locel[i][k]-1] = M[locel[i][j]-1][locel[i][k]-1] + Mes[j][k]
            K[locel[i][j]-1][locel[i][k]-1] = K[locel[i][j]-1][locel[i][k]-1] + Kes[j][k]

nodeConstraint = [1, 2, 3, 4]
def Add_const(nodeConstraint, M, K) :
    for node in nodeConstraint:
        for dof in dofList[node-1]:
            for i in range(M.shape[0]):
                M[dof-1][i] = 0
                M[i][dof-1] = 0

                K[dof - 1][i] = 0
                K[i][dof - 1] = 0

Add_const(nodeConstraint,M,K)

eigenvals, eigenvects = scipy.linalg.eigh(K, M)
print(sorted(eigenvals)[0:8])

    
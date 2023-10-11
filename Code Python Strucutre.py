import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import fct

def ElementFini(numberElem):

    tan_3 = np.tan(np.radians(3))
    nodeList = [[0, 0, 0],  # node 0
                [5, 0, 0],  # node 1
                [0, 5, 0],  # node 2
                [5, 5, 0],  # node 3
                [tan_3, tan_3, 1],  # node 4
                [5 - tan_3, tan_3, 1],  # node 5
                [tan_3, 5 - tan_3, 1],  # node 6
                [5 - tan_3, 5 - tan_3, 1],  # node 7
                [9 * tan_3, 9 * tan_3, 9],  # node 8
                [5 - 9 * tan_3, 9 * tan_3, 9],  # node 9
                [9 * tan_3, 5 - 9 * tan_3, 9],  # node 10
                [5 - 9 * tan_3, 5 - 9 * tan_3, 9],  # node 11
                [17 * tan_3, 17 * tan_3, 17],  # node 12
                [5 - 17 * tan_3, 17 * tan_3, 17],  # node 13
                [17 * tan_3, 5 - 17 * tan_3, 17],  # node 14
                [5 - 17 * tan_3, 5 - 17 * tan_3, 17],  # node 15
                [25 * tan_3, 25 * tan_3, 25],  # node 16
                [5 - 25 * tan_3, 25 * tan_3, 25],  # node 17
                [25 * tan_3, 5 - 25 * tan_3, 25],  # node 18
                [5 - 25 * tan_3, 5 - 25 * tan_3, 25],  # node 19
                [2.5, 2.5, 25],  # node 20
                [2.5, 2.5, 80]]  # node 21"""

    elemList0 = [[0, 4, 0], [1, 5, 0], [2, 6, 0], [3, 7, 0],
                 [4, 8, 0], [5, 9, 0], [6, 10, 0], [7, 11, 0],
                 [8, 12, 0], [9, 13, 0], [10, 14, 0], [11, 15, 0],
                 [12, 16, 0], [13, 17, 0], [14, 18, 0], [15, 19, 0],
                 [4, 5, 1], [4, 6, 1], [5, 7, 1], [7, 6, 1],
                 [8, 9, 1], [8, 10, 1], [10, 11, 1], [11, 9, 1],
                 [12, 13, 1], [12, 14, 1], [14, 15, 1], [15, 13, 1],
                 [16, 17, 1], [16, 18, 1], [18, 19, 1], [19, 17, 1],
                 [8, 5, 1], [5, 11, 1], [11, 6, 1], [6, 8, 1],
                 [13, 8, 1], [8, 14, 1], [14, 11, 1], [11, 13, 1],
                 [16, 14, 1], [14, 19, 1], [19, 13, 1], [13, 16, 1],
                 [16, 20, 2], [17, 20, 2], [18, 20, 2], [19, 20, 2], [20, 21, 2]]

    elemList = fct.create_elemList(elemList0, nodeList, numberElem)
    dofList = fct.create_dofList(nodeList)
    locel = fct.create_locel(elemList, dofList)

    nodeConstraint = np.array([0, 1, 2, 3])
    nodeLumped = np.array([[22, 200000]])

    #fct.plot(elemList, nodeList)

    # Define properties
    #[densité [kg/m3], poisson [-], young [GPa], air section [m2], Rayon interne [m], Rayon externe[]
    # en SI
    mainBeam_d = 1  # [m]
    othbeam_d = 0.6  # [m]
    thickn = 0.02  # [m]
    proprieties = fct.create_properties(mainBeam_d, othbeam_d, thickn)

    M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))

    for i in range(len(elemList)):
        node1 = elemList[i][0]
        node2 = elemList[i][1]
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
        Ix = (np.pi / 64) * (Re ** 4 - Ri ** 4)  # [m4]
        Iy = (np.pi / 64) * (Re ** 4 - Ri ** 4)  # [m4]
        Iz = (np.pi / 64) * (Re ** 4 - Ri ** 4)  # [m4]
        Jx = Ix * 2  # [m4]
        G = E / (2 * (1 + v))  # [GPa]
        r = np.sqrt(Iy/A)  # [m]

        """
        if prop == 2:
            Jx *= 10**4
            Iy *= 10**4
            Iz *= 10**4
        """

        Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
        Mel = fct.create_Mel(m, r, l)
        T = fct.create_T(coord1, coord2, l)

        #Kes = np.dot(np.dot(np.transpose(T), Kel), T)
        Kes = T.T @ Kel @ T
        #Mes = np.dot(np.dot(np.transpose(T), Mel), T)
        Mes = T.T @ Mel @ T

        # Assemblage Matrice globale
        for j in range(len(locel[i])):
            for k in range(len(locel[i])):
                M[locel[i][j]][locel[i][k]] = M[locel[i][j]][locel[i][k]] + Mes[j][k]
                K[locel[i][j]][locel[i][k]] = K[locel[i][j]][locel[i][k]] + Kes[j][k]

    fct.Add_lumped_mass(nodeLumped, dofList, M)

    fct.Add_const_emboit(nodeConstraint, dofList, M, K)


    eigenvals, eigenvects = scipy.linalg.eigh(K, M)
    #print(sorted(np.sqrt(eigenvals)[:8]/(2*np.pi)))
    print(eigenvals[:8])
    #return np.sqrt(eigenvals[:8])/(2*np.pi)


    fig = plt.figure()
    newNodeList = []
    for i in range(8):
        for j in range(len(nodeList)):
            coord = nodeList[j]
            if j not in nodeConstraint:

                dx, dy, dz = eigenvects[i][6*j], eigenvects[i][6*j+1], eigenvects[i][6*j+2]

                #print(dx, dy, dz)

                new_coord = [coord[0]+dx, coord[1]+dy, coord[2]+dz]
                newNodeList.append(new_coord)
            else:
                newNodeList.append(coord)

        ax = fig.add_subplot(2, 4, i+1, projection='3d')

        for elem in elemList0:
            if elem[2] != 2:
                newnode1 = newNodeList[elem[0]]
                newnode2 = newNodeList[elem[1]]
                node1 = nodeList[elem[0]]
                node2 = nodeList[elem[1]]

                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]],'--', c='b')
                ax.plot([newnode1[0], newnode2[0]], [newnode1[1], newnode2[1]], [newnode1[2], newnode2[2]], c='r')


    plt.show()


#ElementFini(3)

def EtudeConvergence(precision):

    TestElem = np.arange(1, precision, 1)
    Result = np.zeros(len(TestElem))

    for i in range(len(Result)):
        Result[i] = ElementFini(TestElem[i])

    

EtudeConvergence(5)
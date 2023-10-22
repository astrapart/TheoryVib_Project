import matplotlib.pyplot as plt
import numpy as np
import scipy
import fct
import data


# proprieties = [rho [kg/m3], poisson [-], Young [Pa], Jx [m4], Iy [m4], Iz [m4]]
vertical_beams = [7800, 0.3, 2.1e11, 5.14e-3, 1.73e-7, 6.9e-6, 8.49e-5]
horizontal_beams = [7800, 0.3, 2.1e11, 5.68e-3, 1.76e-7, 1.2e-4, 7.3e-6]

proprieties = [vertical_beams, horizontal_beams]

def ElementFini(numberElem, verbose):

    nodeListsimple = data.nodeList_example
    elemList0simple = data.elemList0_example
    elemList = fct.create_elemList(elemList0simple, nodeListsimple, numberElem)
    dofList = fct.create_dofList(nodeListsimple)
    locel = fct.create_locel(elemList, dofList)

    if verbose:
        fct.plot_structure(elemList, nodeListsimple)

    nodeConstraint = [1, 2, 3, 4]

    numberNode = len(nodeListsimple)
    M = np.zeros([numberNode*6, numberNode*6])
    K = np.zeros([numberNode*6, numberNode*6])

    for i in range(len(elemList)):

        node1 = elemList[i][0]-1
        node2 = elemList[i][1]-1
        indexProp = elemList[i][2]

        coord1 = nodeListsimple[node1]
        coord2 = nodeListsimple[node2]

        l = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)

        propriety = proprieties[indexProp]

        rho = propriety[0]       # [kg/m3]
        v   = propriety[1]       # [-]
        E   = propriety[2]       # [Pa]
        A   = propriety[3]       # [m2]
        Jx  = propriety[4]       # [m4]
        Iy  = propriety[5]       # [m4]
        Iz  = propriety[6]       # [m4]
        G   = E / (2 * (1 + v))  # [Pa]
        r   = np.sqrt(Iy/A)      # [m]
        m   = rho * l * A        # [kg]

        Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
        Mel = fct.create_Mel(m, r, l)

        T = fct.create_T(coord1, coord2)

        Kes = np.dot(np.dot(np.transpose(T), Kel), T)
        Mes = np.dot(np.dot(np.transpose(T), Mel), T)

        for j in range(len(locel[i])):
            for k in range(len(locel[i])):

                M[locel[i][j]-1][locel[i][k]-1] = M[locel[i][j]-1][locel[i][k]-1] + Mes[j][k]

                K[locel[i][j]-1][locel[i][k]-1] = K[locel[i][j]-1][locel[i][k]-1] + Kes[j][k]

    print("m =", fct.calculate_mtot_rigid(M), "[kg] rigid")

    fct.Add_const_emboit(nodeConstraint, dofList, M, K)
    eigenvals, eigenvects = scipy.linalg.eig(K, M, right=True)
    val_prop = np.sort(eigenvals)

    fct.print_freq(val_prop[:4])

    if verbose:
        new_index = np.argsort(eigenvals)
        vect_prop = []
        for i in new_index:
            vect_prop.append(eigenvects[i])

        fct.plot_result(nodeListsimple, nodeConstraint, vect_prop[:4], elemList0simple)

    return val_prop[:4]

ElementFini(3, False)
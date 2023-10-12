"""
MECA0029-1 Theory of vibration
Analysis of the dynamic behaviour of an offshore wind turbine of jacket
"""

import numpy as np
import scipy
import fct
import data

def ElementFini(numberElem):
    nodeList = data.nodeList_eol
    elemList0 = data.elemList0_eol
    elemList = fct.create_elemList(elemList0, nodeList, numberElem)
    dofList = fct.create_dofList(nodeList)
    locel = fct.create_locel(elemList, dofList)

    nodeConstraint = np.array([0, 1, 2, 3])
    nodeLumped = np.array([[22, 200000]])

    mainBeam_d  = 1     # [m]
    othbeam_d   = 0.6   # [m]
    thickn      = 0.02  # [m]
    #proprieties = fct.create_properties(mainBeam_d, othbeam_d, thickn)

    M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))

    for i in range(len(elemList)):
        node1 = elemList[i][0]
        node2 = elemList[i][1]
        type_beam = elemList[i][2]

        coord1 = nodeList[node1]
        coord2 = nodeList[node2]

        l = fct.calculate_length(coord1, coord2)

        rho, v, E, A, Re, Ri, m, Jx, Iy, Iz, G, r = fct.properties(type_beam, l)

        Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
        Mel = fct.create_Mel(m, r, l)

        T = fct.create_T(coord1, coord2, l)

        Kes = T.T @ Kel @ T
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

    fct.plot_result(nodeList, nodeConstraint, eigenvects, elemList0)

def EtudeConvergence(precision):

    TestElem = np.arange(1, precision, 1)
    Result = np.zeros(len(TestElem))

    for i in range(len(Result)):
        Result[i] = ElementFini(TestElem[i])

    
ElementFini(3)
#EtudeConvergence(5)
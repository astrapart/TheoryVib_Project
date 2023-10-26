"""
MECA0029-1 Theory of vibration
Analysis of the dynamic behaviour of an offshore wind turbine of jacket
"""

import numpy as np
import scipy
import fct
import fct_verif
import data

def ElementFini(numberElem, verbose):

    nodeList = data.nodeList_eol
    elemList0 = data.elemList0_eol
    elemList = fct.create_elemList(elemList0, nodeList, numberElem)
    dofList = fct.create_dofList(nodeList)
    locel = fct.create_locel(elemList, dofList)

    #if verbose:
        #fct.plot_structure(elemList, nodeList)

    nodeConstraint = np.array([1, 2, 3, 4])
    nodeLumped = 23

    M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))

    for i in range(len(elemList)):
        node1 = elemList[i][0]-1
        node2 = elemList[i][1]-1
        type_beam = elemList[i][2]

        coord1 = nodeList[node1]
        coord2 = nodeList[node2]

        l = fct.calculate_length(coord1, coord2)

        rho, v, E, A, D, m, Jx, Iy, Iz, G, r = fct.properties(type_beam, l)


        print("--------Propriety {beam} for elem [{node1}, {node2}]--------".format(beam=type_beam, node1=node1+1, node2=node2+1))
        print("rho = {rho}, E = {E}  A = {A}  l = {l}".format(rho=rho, E=E, A=A, l=l))
        print("G = {G}  r = {r}  m = {m}".format(G=G, r=r, m=m))
        print("Jx = {Jx}  I = {Iy}".format(Jx=Jx, Iy=Iy))
        print()

        Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
        Mel = fct.create_Mel(m, r, l)

        T = fct.create_T(coord1, coord2, l)

        Kes = np.transpose(T) @ Kel @ T
        Mes = np.transpose(T) @ Mel @ T

        """
        if i == 0:
            fct.print_matrix(Kel)
            print()
            fct.print_matrix(Mel)
            print()
            fct.print_matrix(Kes)
            print(fct_verif.est_def_pos(Kes))
            print()
            fct.print_matrix(Mes)
            print(fct_verif.est_def_pos(Mes))
        """

        # Assemblage Matrice globale
        for j in range(len(locel[i])):
            for k in range(len(locel[i])):
                M[locel[i][j]-1][locel[i][k]-1] = M[locel[i][j]-1][locel[i][k]-1] + Mes[j][k]
                K[locel[i][j]-1][locel[i][k]-1] = K[locel[i][j]-1][locel[i][k]-1] + Kes[j][k]

    fct.Add_lumped_mass(nodeLumped, dofList, M)
    print("m =", fct.calculate_mtot_rigid(M), "[kg] rigid")

    #print("m =", fct.calculate_mtot(M, ltot), "[kg]")

    fct.Add_const_emboit(nodeConstraint, dofList, M, K)

    eigenvals, eigenvects = scipy.linalg.eig(K, M, right=True)
    val_prop = np.sort(eigenvals)

    fct.print_freq(val_prop[:8])

    if verbose:
        new_index = np.argsort(eigenvals)
        vect_prop = []
        for i in new_index:
            vect_prop.append(eigenvects[i])

        fct.plot_result(nodeList, nodeConstraint, vect_prop[:8], elemList0)

    return val_prop[:8]


def EtudeConvergence(precision):

    TestElem = np.arange(2, precision+1, 1)
    Result = []

    for i in range(len(TestElem)):
        tmp = ElementFini(TestElem[i], False)
        Result.append(tmp)

ElementFini(3, False)
#EtudeConvergence(5)
"""
MECA0029-1 Theory of vibration
Analysis of the dynamic behaviour of an offshore wind turbine of jacket
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import fct
import data
from scipy.linalg import block_diag, eigh, eigvals, eig

def ElementFini(numberElem, verbose):
    nodeList0 = data.nodeList_eol
    elemList0 = data.elemList0_eol
    elemList, nodeList = fct.create_elemList(elemList0, nodeList0, numberElem)
    dofList = fct.create_dofList(nodeList)
    locel = fct.create_locel(elemList, dofList)

    if verbose:
        fct.plot_structure(elemList, nodeList)

    nodeConstraint = np.array([1, 2, 3, 4])
    nodeLumped = 22

    M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))

    for i in range(len(elemList)):
        node1 = elemList[i][0]-1
        node2 = elemList[i][1]-1
        type_beam = elemList[i][2]

        coord1 = nodeList[node1]
        coord2 = nodeList[node2]

        l = fct.calculate_length(coord1, coord2)

        rho, v, E, A, m, Jx, Iy, Iz, G, r = fct.properties(type_beam, l)

        if verbose:
            fct.print_data_beam(node1, node2, type_beam, rho, v, E, A, m, Jx, Iy, Iz, G, r, l)

        Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
        Mel = fct.create_Mel(m, r, l)

        T = fct.create_T(coord1, coord2, l)

        Kes = np.transpose(T) @ Kel @ T
        Mes = np.transpose(T) @ Mel @ T

        for j in range(len(locel[i])):
            for k in range(len(locel[i])):
                M[locel[i][j]-1][locel[i][k]-1] += Mes[j][k]
                K[locel[i][j]-1][locel[i][k]-1] += Kes[j][k]

    M = fct.Add_lumped_mass(nodeLumped, dofList, M)

    mtot = fct.calculate_mtot_rigid(M)
    if verbose:
        print("m =", mtot, "[kg] rigid")

    M, K = fct.Add_const_emboit(nodeConstraint, dofList, M, K)

    eigenvals, eigenvects = scipy.linalg.eig(K, M)
    val_prop = np.sort(np.real(eigenvals))

    if verbose:
        new_index = np.argsort(eigenvals)
        vect_prop = []
        for i in new_index:
            vect_prop.append(eigenvects[i])

        fct.print_freq(val_prop[:8])
        fct.plot_result(nodeList, nodeConstraint, vect_prop[:8], elemList0)

    return val_prop[:8]


def EtudeConvergence(precision):
    TestElem = np.arange(2, precision + 1, 1)
    print(TestElem)
    Result = []

    for i in range(len(TestElem)):
        tmp = ElementFini(TestElem[i], False)
        Result.append(tmp)
        print('Les valeurs propres pour', TestElem[i] , 'élements sont : ', np.real(np.sqrt(tmp))/(2*np.pi))

    plt.figure()
    for i in range(len(TestElem)-1):
        plt.plot([TestElem[i], TestElem[i+1]], [Result[i][0], Result[i+1][0]])

    plt.grid()
    plt.title("Convergence of the fisrt valeur propre")
    plt.show()


#ElementFini(3, True)

EtudeConvergence(15)

"""
Result = [[0.4437535, 0.45433177,  0.97293389,  7.05536334,  7.40416045, 15.94143563, 20.54892234, 22.10797568],
          [0.44375284,  0.45433049,  0.97293385,  7.05444093,  7.40314732, 15.94072114, 20.52106343, 22.07593765],
          [ 0.44374422,  0.45432947,  0.97293014,  7.05437772,  7.40286752, 15.94055679, 20.51636776, 22.07033115],
          [ 0.44375049,  0.45432843 , 0.97293498,  7.05422351,  7.40294721, 15.94049676, 20.51505775, 22.06887598],
          [ 0.44375399 , 0.45432601,  0.97292409 , 7.05419012 , 7.4030251,  15.94046982, 20.51458456, 22.0682478 ],
          [0.44375396,  0.45433893,  0.97292579,  7.05431972,  7.40298813, 15.94045202, 20.51436938, 22.06804018],
          [ 0.44374995,  0.45432814,  0.97293293 , 7.05439332 , 7.40284139, 15.94043059, 20.51421719, 22.06789641],
          [ 0.44375102,  0.45433117,  0.97293204,  7.05442877,  7.40292553, 15.94040496, 20.51421218, 22.06789859],
          [ 0.44375631, 0.45432903,  0.97290094 , 7.05423684 , 7.40313393, 15.94042328, 20.51430098, 22.06779005]]

TestElem = [2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.figure()
for i in range(len(TestElem) - 1):
    plt.plot([TestElem[i], TestElem[i + 1]], [Result[i][0]*10, Result[i + 1][0]*10])

plt.grid()
plt.title("Convergence of the fisrt valeur propre")
plt.show()
"""

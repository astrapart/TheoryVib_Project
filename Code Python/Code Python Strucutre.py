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
import time


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
    Result = []

    for i in range(len(TestElem)):
        t1 = time.time()
        tmp = ElementFini(TestElem[i], False)
        t2 = time.time()
        Result.append(tmp)
        print(f'Les valeurs propres pour {TestElem[i]} élements sont : {np.real(np.sqrt(tmp))/(2*np.pi)} in {t2 - t1} sec' )

    plt.figure()
    for i in range(len(TestElem)-1):
        plt.plot([TestElem[i], TestElem[i+1]], [Result[i][0], Result[i+1][0]], c='b')

    plt.grid()
    plt.title("Convergence of the first natural frequencies")
    plt.show()


#ElementFini(3, True)

#EtudeConvergence(15)

Result = [[0.4437535, 0.45433177, 0.97293389, 7.05536334, 7.40416045, 15.94143563, 20.54892234, 22.10797568],
          [0.44375284, 0.45433049, 0.97293385, 7.05444093, 7.40314732, 15.94072114, 20.52106343, 22.07593765],
          [0.44374422, 0.45432947, 0.97293014, 7.05437772, 7.40286752, 15.94055679, 20.51636776, 22.07033115],
          [0.44375049, 0.45432843, 0.97293498, 7.05422351, 7.40294721, 15.94049676, 20.51505775, 22.06887598],
          [0.44375399, 0.45432601, 0.97292409, 7.05419012, 7.4030251, 15.94046982, 20.51458456, 22.0682478 ],
          [0.44375396, 0.45433893, 0.97292579, 7.05431972, 7.40298813, 15.94045202, 20.51436938, 22.06804018],
          [0.44374995, 0.45432814, 0.97293293, 7.05439332, 7.40284139, 15.94043059, 20.51421719, 22.06789641],
          [0.44375102, 0.45433117, 0.97293204, 7.05442877, 7.40292553, 15.94040496, 20.51421218, 22.06789859],
          [0.44375631, 0.45432903, 0.97290094, 7.05423684, 7.40313393, 15.94042328, 20.51430098, 22.06779005],
          [0.44375333, 0.45433295, 0.97293302, 7.0542291, 7.40297567, 15.9404245, 20.51415934, 22.06781938],
          [0.44375504, 0.45433308, 0.9729252, 7.05420672, 7.40312114, 15.94043722, 20.51424485, 22.06775965],
          [0.44375541, 0.45433097, 0.97293313, 7.05427859, 7.40300588, 15.94042955, 20.51412298, 22.06761888],
          [0.44374803, 0.4543304, 0.97293359, 7.05414533, 7.40263314, 15.94042454, 20.5138263, 22.06777685],
          [0.44375443, 0.45433027, 0.97293488, 7.05424238, 7.40299184, 15.94041767, 20.51413501, 22.06777646]]

Time = [0.3505735397338867, 1.1360270977020264, 2.9674744606018066, 8.783644199371338, 19.95775556564331,
        37.788264751434326, 62.78030729293823, 92.64252924919128, 138.71605777740479, 191.74096274375916,
        259.28460359573364, 329.9837477207184, 435.48192715644836, 613.6963446140289]

TestElem = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

plt.figure()
for i in range(len(TestElem) - 1):
    plt.plot([TestElem[i], TestElem[i + 1]], [Result[i][0], Result[i + 1][0]], c='b')

plt.grid()
plt.title("Convergence of the first natural frequencies")
plt.show()

plt.figure()
for i in range(len(TestElem) - 1):
    plt.plot([TestElem[i], TestElem[i + 1]], [Time[i], Time[i + 1]], c='b')

plt.grid()
plt.title("Evolution of time per number of beam")
plt.show()


def DampingMatrix(eigenVals, dampingratio, M, K):

    A = 0.5 * np.array([[1/eigenVals[0], eigenVals[0]],
                        [1/eigenVals[1], eigenVals[1]]])
    b = dampingratio

    alpha, beta = np.linalg.solve(A, b)

    return alpha * K + beta * M

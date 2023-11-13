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
from scipy.integrate import odeint
import time


def ElementFini_OffShoreStruct(numberElem, numberMode, verbose):
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
    val_prop = np.sqrt(val_prop) / (2 * np.pi)

    new_index = np.argsort(eigenvals)
    vect_prop = []
    for i in new_index:
        vect_prop.append(eigenvects[i])

    if verbose:
        fct.print_freq(val_prop[:numberMode])
        fct.plot_result(nodeList, nodeConstraint, vect_prop[:numberMode], elemList0, dofList)

    return val_prop[:numberMode], vect_prop[:numberMode], K, M, dofList

def EtudeConvergence(precision):
    TestElem = np.arange(2, precision + 1, 1)
    Result = []

    for i in range(len(TestElem)):
        t1 = time.time()
        tmp, _, _, _, _ = ElementFini_OffShoreStruct(TestElem[i], False)
        t2 = time.time()
        Result.append(tmp)
        print(f'Les valeurs propres pour {TestElem[i]} élements sont : {np.real(np.sqrt(tmp))/(2*np.pi)} in {t2 - t1} sec' )

    plt.figure()
    for i in range(len(TestElem)-1):
        plt.plot([TestElem[i], TestElem[i+1]], [Result[i][0], Result[i+1][0]], c='b')

    plt.grid()
    plt.title("Convergence of the first natural frequencies")
    plt.show()

#ElementFini_OffShoreStruct(3, 8, True)
#EtudeConvergence(15)

def ConvergencePlot():
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

    fig = plt.figure(figsize=(15.5, 7.5))
    for i in range(len(Result[0])):
        ax = fig.add_subplot(2, 4, i+1)
        for j in range(len(TestElem) - 1):
            ax.plot([TestElem[j], TestElem[j + 1]], [Result[j][i], Result[j + 1][i]], c='b')

        ax.grid()
        ax.set_title(f"{i+1} eigenvalues")

    plt.show()

    """  Plot du temps d'exécution, pas très intéressant 
    plt.figure()
    for i in range(len(TestElem) - 1):
        plt.plot([TestElem[i], TestElem[i + 1]], [Time[i], Time[i + 1]], c='b')
    
    plt.grid()
    plt.title("Evolution of time per number of beam")
    plt.show()
    """

#ConvergencePlot()


def F(t):
    return data.m * data.a * data.efficiency * np.sin(2*np.pi*data.f * t)


def P(n, applNode, dofList, t):
    labs = len(t)
    p = np.zeros((n, labs))
    x = dofList[applNode - 1][0]
    y = dofList[applNode - 1][1]
    for i in range(labs):
        if 0 <= t[i] <= data.timpact:
            p[x][i] = F(t[i]) * np.sqrt(2) / 2
            p[y][i] = F(t[i]) * np.sqrt(2) / 2
    return p


def Phi(x, mu, p):
    phi = []
    for i in range(len(x)):
        phi.append(x[i].T @ p / mu[i])

    return phi


def Wrd(wr,er) :
    return wr*np.sqrt(1-(er**2))


def H(er, wr, wrd,t):
    return np.exp(-er * wr * t) * np.sin(wrd * t) / wrd


def CoefficientAlphaBeta(eigenVals):
    A = 0.5 * np.array([[eigenVals[0], 1 / eigenVals[0]],
                        [eigenVals[1], 1 / eigenVals[1]]])
    b = data.dampingRatioInit

    return np.linalg.solve(A, b)


def DampingMatrix(alpha, beta, K, M):
    return alpha * K + beta * M


def Mu(eigenvectors, M):
    mu = []
    for eigenvect in eigenvectors:
        mu.append(np.transpose(eigenvect) @ M @ eigenvect)
    return np.array(mu)


def DampingRatios(alpha, beta, eigenValues):
    dampingRatios = np.zeros(len(eigenValues))
    dampingRatios[0] = data.dampingRatioInit[0]
    dampingRatios[1] = data.dampingRatioInit[1]

    for i in range(2, len(dampingRatios)):
        dampingRatios[i] = 0.5 * (alpha * eigenValues[i] + beta / eigenValues[i])

    return dampingRatios


def compute_eta(Eigenvectors,EigenValues, DampingRatio, phi, t):
    eta = []
    for i in range(len(Eigenvectors)):
        er = DampingRatio[i]
        wr = EigenValues[i]
        wrd = Wrd(wr, er)
        h = H(er, wr, wrd, t)
        eta.append(np.convolve(phi[i], h)[:len(t)])

    return eta


def compute_q(Eigenvectors, eta,t):
    nbreDof = len(Eigenvectors[0])
    Mode_nbr = len(Eigenvectors)

    q = np.zeros((nbreDof, len(t)))
    for i in range(Mode_nbr):
        for j in range(nbreDof):
            for k in range(len(t)):
                q[j][k] += eta[i][k] * Eigenvectors[i][j]

    return q


def ModeDisplacementMethod(eigenvectors, eta, t):
    nbreDof = len(eigenvectors[0])
    Mode_nbr = len(eigenvectors)

    q = np.zeros((nbreDof, len(t)))
    for i in range(Mode_nbr):
        for j in range(nbreDof):
            for k in range(len(t)):
                q[j][k] += eta[i][k] * eigenvectors[i][j]

    return q


def ModeAccelerationMethod(eigenvectors, eigenvalues, eta, K, phi, p, t):
    nbreDof = len(eigenvectors[0])
    Mode_nbr = len(eigenvectors)

    q = np.zeros((nbreDof, len(t)))
    for i in range(Mode_nbr):
        for j in range(nbreDof):
            for k in range(len(t)):
                q[j][k] += eta[i][k] * eigenvectors[i][j]

    for i in range(Mode_nbr):
        for j in range(nbreDof):
            for k in range(len(t)):
                q[j][k] -= phi[i][k] * eigenvectors[i][j] / eigenvalues[i] ** 2

    q += np.linalg.inv(K) @ p

    return q


numberElem = 3
numbermode = 8
EigenValues, EigenVectors, K, M, DofList = ElementFini_OffShoreStruct(numberElem, numbermode, False)

t_final = 40
t = np.linspace(0, t_final, 500)

mu = Mu(EigenVectors, M)
Alpha, Beta = CoefficientAlphaBeta(EigenValues)
C = DampingMatrix(Alpha, Beta, K, M)
DampingRatio = DampingRatios(Alpha, Beta, EigenValues)
p = P(len(EigenVectors[0]), data.ApplNode, DofList, t)
phi = Phi(EigenVectors, mu, p)
eta = compute_eta(EigenVectors, EigenValues, DampingRatio, phi, t)

q = compute_q(EigenVectors, eta, t)
qDisp = ModeDisplacementMethod(EigenVectors, eta, t)
qAcc = ModeAccelerationMethod(EigenVectors, EigenValues, eta, K, phi, p, t)


fig = plt.figure(figsize=(10, 6.5))

ax1 = fig.add_subplot(211)
DisplacementNodeX = qDisp[DofList[18][0]]
DisplacementNodeY = qDisp[DofList[18][1]]

ax1.plot(t, np.sqrt(DisplacementNodeX ** 2 + DisplacementNodeY ** 2))
ax1.set_title("Displacement of the Node")
ax1.set_ylim(0, 10e-4)

ax2 = fig.add_subplot(212)
DisplacementRotorX = qDisp[DofList[21][0]]
DisplacementRotorY = qDisp[DofList[21][1]]

ax2.plot(t, np.sqrt(DisplacementRotorX ** 2 + DisplacementRotorY ** 2), c='r')
ax2.set_title("Displacement of the Rotor")
ax2.set_ylim(0, 10e-4)

fig.suptitle("Mode Displacement Method")
plt.show()


fig = plt.figure(figsize=(10, 6.5))

ax1 = fig.add_subplot(211)
DisplacementNodeX = qAcc[DofList[18][0]]
DisplacementNodeY = qAcc[DofList[18][1]]

ax1.plot(t, np.sqrt(DisplacementNodeX ** 2 + DisplacementNodeY ** 2), label="displacement Node")
ax1.set_title("Displacement of the Node")
ax1.set_ylim(0, 10e-4)
ax1.legend()

ax2 = fig.add_subplot(212)
DisplacementRotorX = qAcc[DofList[21][0]]
DisplacementRotorY = qAcc[DofList[21][1]]

ax2.plot(t, np.sqrt(DisplacementRotorX ** 2 + DisplacementRotorY ** 2), c='r', label="displacement Rotor")
ax2.set_title("Displacement of the Rotor")
ax2.set_ylim(0, 10e-4)
ax2.legend()

fig.suptitle("Mode Acceleration Method")
plt.show()

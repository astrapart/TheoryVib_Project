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
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def ElementFini_OffShoreStruct(numberElem, numberMode, print_data_beam, print_mtot, plot_structure, plot_result):
    nodeList0 = data.nodeList_eol
    elemList0 = data.elemList0_eol
    elemList, nodeList = fct.create_elemList(elemList0, nodeList0, numberElem)
    dofList = fct.create_dofList(nodeList)
    locel = fct.create_locel(elemList, dofList)

    if plot_structure:
        fct.plot_structure(elemList, nodeList)

    nodeConstraint = np.array([1, 2, 3, 4])
    nodeLumped = 22

    M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    start = time.time()

    for i in range(len(elemList)):
        node1 = elemList[i][0]-1
        node2 = elemList[i][1]-1
        type_beam = elemList[i][2]

        coord1 = nodeList[node1]
        coord2 = nodeList[node2]

        l = fct.calculate_length(coord1, coord2)

        rho, v, E, A, m, Jx, Iy, Iz, G, r = fct.properties(type_beam, l)

        if print_data_beam:
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

        progress = i / (len(elemList) - 1) * 100
        print('\rProgress Element fini: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)

    end = time.time()
    execution = end - start
    print(f"\nTotal execution time: {execution:.2f} seconds")

    M = fct.Add_lumped_mass(nodeLumped, dofList, M)

    mtot = fct.calculate_mtot_rigid(M)
    if print_mtot:
        print("m =", mtot, "[kg] rigid")

    M, K = fct.Add_const_emboit(nodeConstraint, dofList, M, K)

    eigenvals, eigenvects = scipy.linalg.eig(K, M)
    new_index = np.argsort(np.real(eigenvals))

    eigenvects = eigenvects.T
    val_prop = []
    vect_prop = []
    for i in new_index:
        vect_prop.append(eigenvects[i])
        val_prop.append(np.sqrt(np.real(eigenvals[i]))/(2*np.pi))


    if plot_result:
        fct.print_freq(val_prop[:numberMode])
        fct.plot_result(nodeList, nodeConstraint, vect_prop[:numberMode], elemList0, dofList)

    return np.array(val_prop[:numberMode]), np.array(vect_prop[:numberMode]), K, M, dofList

#printDataBeam = False
#printMtot = False
#printStructure = False
#printResult = True
#tmp, _, _, _, _ = ElementFini_OffShoreStruct(12, 8, printDataBeam, printMtot, printStructure, printResult)

def EtudeConvergence(precision):
    TestElem = np.arange(2, precision + 1, 1)
    Result = []

    for i in range(len(TestElem)):
        t1 = time.time()
        tmp, _, _, _, _ = ElementFini_OffShoreStruct(TestElem[i], 8,False, False, False, False)
        t2 = time.time()
        Result.append(tmp)
        print(f'Les valeurs propres pour {TestElem[i]} élements sont : {np.real(np.sqrt(tmp))/(2*np.pi)} in {t2 - t1} sec' )

    plt.figure()
    for i in range(len(TestElem)-1):
        plt.plot([TestElem[i], TestElem[i+1]], [Result[i][0], Result[i+1][0]], c='b')

    plt.grid()
    plt.title("Convergence of the first natural frequencies")
    plt.show()

def ModeDisplacementMethod(eigenvectors, eta, t):
    start = time.time()

    nbreDof = len(eigenvectors[0])
    Mode_nbr = len(eigenvectors)

    q = np.zeros((nbreDof, len(t)))
    eta = np.array(eta)
    eigenvectors = np.array(eigenvectors)
    for i in range(Mode_nbr):
        q += np.dot(eigenvectors[i, :].reshape(nbreDof, 1), eta[i, :].reshape(1, len(t)))

        progress = i / (Mode_nbr - 1) * 100
        print('\rProgress Displacement Method: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)

    end = time.time()
    delta = end - start
    print(f"\nTotal execution time: {delta:.2f} seconds")

    return q.T


def ModeAccelerationMethod(eigenvectors, eigenvalues, eta, K, phi, p, t):
    start = time.time()

    nbreDof = len(eigenvectors[0])
    Mode_nbr = len(eigenvectors)

    q = np.zeros((nbreDof, len(t)))
    for i in range(Mode_nbr):
        q += np.dot(eigenvectors[i, :].reshape(nbreDof, 1), eta[i, :].reshape(1, len(t)))
        q -= np.dot(eigenvectors[i, :].reshape(nbreDof, 1), phi[i, :].reshape(1, len(t))) / eigenvalues[i] ** 2
        """
        for j in range(nbreDof):
            for k in range(len(t)):
                q[k][j] += eta[i][k] * eigenvectors[i][j]
                q[k][j] -= phi[i][k] * eigenvectors[i][j] / eigenvalues[i] ** 2
        """

        progress = i / (Mode_nbr - 1) * 100
        print('\rProgress Acceleration Method: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)

    q += (np.linalg.inv(K) @ p)

    end = time.time()
    delta = end - start
    print(f"\nTotal execution time: {delta:.2f} seconds")

    return q.T


def TransientResponse(numberMode, t, pas, verbose):
    numberElem = 3
    EigenValues, EigenVectors, K, M, DofList = ElementFini_OffShoreStruct(numberElem, numberMode, False, False, False, False)
    EigenValues = np.array(EigenValues) * np.pi * 2

    mu = fct.Mu(EigenVectors, M)
    Alpha, Beta = fct.CoefficientAlphaBeta(EigenValues)
    C = fct.DampingMatrix(Alpha, Beta, K, M)
    DampingRatio = fct.DampingRatios(Alpha, Beta, EigenValues)
    print(DampingRatio)
    p = fct.P(len(EigenVectors[0]), DofList, t, K)
    phi = fct.Phi(EigenVectors, mu, p)

    eta = fct.compute_eta(EigenVectors, EigenValues, DampingRatio, phi, t, pas)

    qDisp = ModeDisplacementMethod(EigenVectors, eta, t)
    qAcc = ModeAccelerationMethod(EigenVectors, EigenValues, eta, K, phi, p, t)

    if verbose:
        fct.print_TransientResponse(qAcc, qDisp, t, DofList)

    return qAcc, qDisp, C, p, K, M, DofList


def ConvergenceTransientResponse(numberMode, t, h):
    numberModeList = np.arange(2, numberMode + 1, 1)
    responseAcc = []
    responseDisp = []
    dofList = []

    for numberMode in numberModeList:
        qAcc, qDisp, _, _, _, _, DofList = TransientResponse(numberMode, t, h, False)

        dofList = DofList
        responseAcc.append(qAcc)
        responseDisp.append(qDisp)

    fct.print_ConvergenceTransientResponse(numberModeList, responseDisp, responseAcc, dofList, t)


def Newmark(M, C, K, p, h, t):
    gamma = data.gamma
    beta = data.beta
    qdisp = np.zeros((len(t), len(M)))
    qvel  = np.zeros((len(t), len(M)))
    qacc  = np.zeros((len(t), len(M)))

    S = fct.compute_S(M, h, gamma, C, beta, K)
    S_inv = np.linalg.inv(S)

    #qacc[0] = np.linalg.solve(M, p.T[0] - C @ qvel[0].T - K @ qdisp[0].T)

    start = time.time()
    for i in range(1, len(t)):
        qvel[i] = qvel[i - 1] + (1 - gamma) * h * qacc[i - 1]
        qdisp[i] = qdisp[i-1] + h * qvel[i-1] + (0.5 - beta) * (h**2) * qacc[i-1]

        #qacc[i] = np.linalg.solve(S, p.T[i] - C @ qvel[i].T - K @ qdisp[i].T)
        qacc[i] = S_inv @ (p.T[i] - C @ qvel[i].T - K @ qdisp[i].T)

        qvel[i] = qvel[i] + h * gamma * qacc[i]
        qdisp[i] = qdisp[i] + (h**2) * beta * qacc[i]

        progress = i / (len(t)-1) * 100
        print('\rProgress Newmark: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)
    end = time.time()
    execution_time = end - start
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

    return qdisp, qvel, qacc


# ElementFini_OffShoreStruct(3, 8, False)

NumberMode = 8
tfin = 10
pas = 0.001
t = np.arange(0, tfin, pas)

qAcc, qDisp, C, p, K, M, DofList = TransientResponse(NumberMode, t, pas, False)
qDispN, qVelN, qAccN = Newmark(M, C, K, p, pas, t)

#fct.printResult(qAcc, qDisp, qDispN, t, DofList)

# ConvergenceTransientResponse(NumberMode, t, pas)


def ElementFini_OffShoreStructReduced(numberElem, numberMode, print_data_beam, print_mtot, plot_structure, plot_result):
    nodeList0 = data.nodeList_eol
    elemList0 = data.elemList0_eol
    elemList, nodeList = fct.create_elemList(elemList0, nodeList0, numberElem)
    dofList = fct.create_dofList(nodeList)
    locel = fct.create_locel(elemList, dofList)

    if plot_structure:
        fct.plot_structure(elemList, nodeList)

    nodeConstraint = np.array([1, 2, 3, 4])
    nodeLumped = 22

    M = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    K = np.zeros((len(nodeList) * 6, len(nodeList) * 6))
    start = time.time()

    for i in range(len(elemList)):
        node1 = elemList[i][0]-1
        node2 = elemList[i][1]-1
        type_beam = elemList[i][2]

        coord1 = nodeList[node1]
        coord2 = nodeList[node2]

        l = fct.calculate_length(coord1, coord2)

        rho, v, E, A, m, Jx, Iy, Iz, G, r = fct.properties(type_beam, l)

        if print_data_beam:
            fct.print_data_beam(node1, node2, type_beam, rho, v, E, A, m, Jx, Iy, Iz, G, r, l)

        Kel = fct.create_Kel(E, A, Jx, Iy, Iz, G, l)
        Mel = fct.create_Mel(m, r, l)

        """ 
        def reduced(Matrix):
            for red in range(2):
                Matrix = np.delete(Matrix, 11, red)
                Matrix = np.delete(Matrix, 10, red)
                Matrix = np.delete(Matrix, 4, red)
                Matrix = np.delete(Matrix, 3, red)

            return Matrix
        """
        # KelRed = reduced(Kel.copy()) TODO à retirer
        # MelRed = reduced(Mel.copy())

        T = fct.create_T(coord1, coord2, l)

        # TRed = reduced(T.copy())

        Kes = np.transpose(T) @ Kel @ T
        Mes = np.transpose(T) @ Mel @ T

        # KesRed = np.transpose(TRed) @ KelRed @ TRed
        # MesRed = np.transpose(TRed) @ MelRed @ TRed

        for j in range(len(locel[i])):
            for k in range(len(locel[i])):
                M[locel[i][j]-1][locel[i][k]-1] += Mes[j][k]
                K[locel[i][j]-1][locel[i][k]-1] += Kes[j][k]

        progress = i / (len(elemList) - 1) * 100
        print('\rProgress Element fini: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)

    end = time.time()
    execution = end - start
    print(f"\nTotal execution time: {execution:.2f} seconds")

    M = fct.Add_lumped_mass(nodeLumped, dofList, M)

    mtot = fct.calculate_mtot_rigid(M)
    if print_mtot:
        print("m =", mtot, "[kg] rigid")

    M, K = fct.Add_const_emboit(nodeConstraint, dofList, M, K)

    MRR, MRC, MCR, MCC = M.copy(), M.copy(), M.copy(), M.copy()
    KRR, KRC, KCR, KCC = K.copy(), K.copy(), K.copy(), K.copy()
    for i in range(len(M) // 6, 0, -1):
        for j in range(1, 7):
            if j in [2, 3]:
                MRR = np.delete(MRR, 6 * i - j, 0)
                MRR = np.delete(MRR, 6 * i - j, 1)

                KRR = np.delete(KRR, 6 * i - j, 0)
                KRR = np.delete(KRR, 6 * i - j, 1)

                MRC = np.delete(MRC, 6 * i - j, 0)

                MCR = np.delete(MCR, 6 * i - j, 1)

                KRC = np.delete(KRC, 6 * i - j, 0)

                KCR = np.delete(KCR, 6 * i - j, 1)

            if j in [1, 4, 5, 6]:
                MCC = np.delete(MCC, 6 * i - j, 0)
                MCC = np.delete(MCC, 6 * i - j, 1)

                KCC = np.delete(KCC, 6 * i - j, 0)
                KCC = np.delete(KCC, 6 * i - j, 1)

                MRC = np.delete(MRC, 6 * i - j, 1)

                MCR = np.delete(MCR, 6 * i - j, 0)

                KRC = np.delete(KRC, 6 * i - j, 1)

                KCR = np.delete(KCR, 6 * i - j, 0)

        progress = (len(M) // 6 - i) / (len(M)//6 - 1) * 100
        print('\rProgress Guyan-Irons: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)
    print("\n")

    newK = np.concatenate((np.concatenate((KRR, KCR), axis=0), np.concatenate((KRC, KCC), axis=0)), axis=1)
    newM = np.concatenate((np.concatenate((MRR, MCR), axis=0), np.concatenate((MRC, MCC), axis=0)), axis=1)

    I = np.identity(len(KRR))
    tmp = - np.linalg.inv(KCC) @ KCR
    RGI = np.concatenate((I, tmp), axis=0)

    KGI = RGI.T @ newK @ RGI
    MGI = RGI.T @ newM @ RGI

    t1 = time.time()
    eigenvals, eigenvects = scipy.linalg.eig(K, M)
    t2 = time.time()
    new_index = np.argsort(np.real(eigenvals))

    t1GI = time.time()
    eigenvalsGI, eigenvectsGI = scipy.linalg.eig(KGI, MGI)
    t2GI = time.time()
    new_indexGI = np.argsort(np.real(eigenvalsGI))

    eigenvectsGI = eigenvectsGI.T
    #val_prop = np.sort(np.real(eigenvals))
    #val_prop = np.sqrt(val_prop) / (2 * np.pi)
    val_prop = []
    vect_prop = []
    for i in new_index:
        vect_prop.append(eigenvects[i])
        val_prop.append(np.sqrt(np.real(eigenvals[i]))/(2*np.pi))

    val_propGI = []
    vect_propGI = []
    for i in new_indexGI:
        val_propGI.append(np.sqrt(np.real(eigenvalsGI[i]))/(2*np.pi))
        vect_propGI.append(eigenvects[i])

    if plot_result:
        fct.print_freqComparaison(val_prop[:numberMode], val_propGI[:numberMode])

    return val_prop[:numberMode], vect_prop[:numberMode], val_propGI[:numberMode], vect_prop[:numberMode], KGI, MGI, dofList, t2 - t1, t2GI - t1GI


def ReducedMethod(numberElem, numberMode, numberModeIncluded, plot_result):

    eigenValues, eigenVectors, K, M, dofList = ElementFini_OffShoreStruct(numberElem, 8, False, False, False, False)

    MRR, MRC, MCR, MCC, KRR, KRC, KCR, KCC = fct.decompositionMatrix(K, M)

    newK = np.concatenate((np.concatenate((KRR, KCR), axis=0), np.concatenate((KRC, KCC), axis=0)), axis=1)
    newM = np.concatenate((np.concatenate((MRR, MCR), axis=0), np.concatenate((MRC, MCC), axis=0)), axis=1)

    _, X = scipy.linalg.eig(KCC, MCC)
    XI = X[:, :numberModeIncluded]
    I = np.identity(len(KRR))
    RGI = np.concatenate((I, - np.linalg.inv(KCC) @ KCR), axis=0)
    Zeros = np.zeros((len(KRR), len(XI[0])))
    RCB = np.concatenate((RGI, np.concatenate((Zeros, XI), axis=0)), axis=1)

    KGI = RGI.T @ newK @ RGI
    MGI = RGI.T @ newM @ RGI

    KCB = RCB.T @ newK @ RCB
    MCB = RCB.T @ newM @ RCB

    t1 = time.time()
    eigenvals, eigenvects = scipy.linalg.eig(K, M)
    t2 = time.time()
    new_index = np.argsort(np.real(eigenvals))

    t1GI = time.time()
    eigenvalsGI, eigenvectsGI = scipy.linalg.eig(KGI, MGI)
    t2GI = time.time()
    new_indexGI = np.argsort(np.real(eigenvalsGI))

    t1CB = time.time()
    eigenvalsCB, eigenvectsCB = scipy.linalg.eig(KCB, MCB)
    t2CB = time.time()
    new_indexCB = np.argsort(np.real(eigenvalsCB))

    val_prop = []
    vect_prop = []
    for i in new_index:
        vect_prop.append(eigenvects[i])
        val_prop.append(np.sqrt(np.real(eigenvals[i])) / (2 * np.pi))

    val_propGI = []
    vect_propGI = []
    for i in new_indexGI:
        val_propGI.append(np.sqrt(np.real(eigenvalsGI[i])) / (2 * np.pi))
        vect_propGI.append(eigenvectsGI[i])

    val_propCB = []
    vect_propCB = []
    for i in new_indexCB:
        val_propCB.append(np.sqrt(np.real(eigenvalsCB[i])) / (2 * np.pi))
        vect_propCB.append(eigenvectsCB[i])

    if plot_result:
        fct.print_freqComparaison(val_prop[:numberMode], val_propGI[:numberMode], val_propCB[:numberMode])

    return val_prop[:numberMode], vect_prop[:numberMode], K, M, val_propGI[:numberMode], vect_propGI[:numberMode], KGI, MGI, val_propCB[:numberMode], vect_propCB[:numberMode], KCB, MCB, t2 - t1, t2GI - t1GI, t2CB - t1CB


def NewmarkGI(KGI, MGI, eigenvalues, eigenvectors, h, t):

    p = np.zeros((len(MGI), len(t)))

    xAppl = (17 - 4) * 4
    yAppl = (17 - 4) * 4 + 1

    for i in range(len(t)):
        p[xAppl][i] = -fct.F(t[i]) * np.sqrt(2) / 2
        p[yAppl][i] = fct.F(t[i]) * np.sqrt(2) / 2

    alphaGI, betaGI = fct.CoefficientAlphaBeta(eigenvalues)
    CGI = fct.DampingMatrix(alphaGI, betaGI, KGI, MGI)
    gamma = data.gamma
    beta = data.beta
    qdisp = np.zeros((len(t), len(MGI)))
    qvel = np.zeros((len(t), len(MGI)))
    qacc = np.zeros((len(t), len(MGI)))

    S = fct.compute_S(MGI, h, gamma, CGI, beta, KGI)
    S_inv = np.linalg.inv(S)

    start = time.time()
    for i in range(1, len(t)):
        qvel[i] = qvel[i - 1] + (1 - gamma) * h * qacc[i - 1]
        qdisp[i] = qdisp[i - 1] + h * qvel[i - 1] + (0.5 - beta) * (h ** 2) * qacc[i - 1]

        qacc[i] = S_inv @ (p.T[i] - CGI @ qvel[i].T - KGI @ qdisp[i].T)

        qvel[i] = qvel[i] + h * gamma * qacc[i]
        qdisp[i] = qdisp[i] + (h ** 2) * beta * qacc[i]

        progress = i / (len(t) - 1) * 100
        print('\rProgress Newmark: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)
    end = time.time()
    execution_time = end - start
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

    return qdisp, qvel, qacc


printDataBeam = False
printMtot = False
printStructure = False
printResult = False
_, _, _, _, EigenValuesGI, EigenVectorsGI, KGI, MGI, _, _, _, _, _, _, _ = ReducedMethod(3, 8, 8, False)
qDispNGI, qVelNGI, qAccNGI = NewmarkGI(KGI, MGI, EigenValuesGI, EigenVectorsGI, pas, t)

fig = plt.figure(figsize=(10, 7))

ax1 = fig.add_subplot(211)
DisplacementNodeX = qDispNGI[:, (17 - 4) * 4]
DispNode = -np.sqrt(2) * DisplacementNodeX
DisplacementNodeXN = qDispN[:, (17 - 4) * 6]
DispNodeN = -np.sqrt(2) * DisplacementNodeXN
ax1.plot(t, DispNode*1000, label='Gyuan-Iron', c='blue')
ax1.plot(t, DispNodeN*1000, label='Full Matrix', c='red')
ax1.legend()
ax1.set_title("Displacement of the Node")

ax2 = fig.add_subplot(212)
DisplacementRotorX = qDispNGI[:, (21 - 4) * 4]
DispRotor = -np.sqrt(2) * DisplacementRotorX
DisplacementRotorXN = qDispN[:, (21 - 4) * 6]
DispRotorN = -np.sqrt(2) * DisplacementRotorXN
ax2.plot(t, DispRotor*1000, label='Gyuan-Iron', c='blue')
ax2.plot(t, DispRotorN*1000, label='Full Matrix', c='red')
ax2.legend()
ax2.set_title("Displacement of the Rotor")

plt.show()

#ReducedMethod(8, 20, True)


def CompareFE_GI():
    numberElemList = np.arange(1, 13, 1)
    Time = np.zeros(len(numberElemList))
    TimeGI = np.zeros(len(numberElemList))
    TimeCB = np.zeros(len(numberElemList))

    for i in range(len(numberElemList)):
        print(f"############# Number of Elem : {numberElemList[i]} #############")
        _, _, _, _, _, _, _, _, _, _, _, _, Time[i], TimeGI[i], TimeCB[i] = ReducedMethod(numberElemList[i], 8, 15, False)

    plt.plot(numberElemList, Time, label="FE")
    plt.plot(numberElemList, TimeGI, label="Guyan-Iron")
    plt.plot(numberElemList, TimeCB, label="Craig-Bampton")
    plt.legend()
    plt.show()


#CompareFE_GI()

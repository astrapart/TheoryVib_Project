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
    #val_prop = np.sort(np.real(eigenvals))
    #val_prop = np.sqrt(val_prop) / (2 * np.pi)
    val_prop = []
    vect_prop = []
    for i in new_index:
        vect_prop.append(eigenvects[i])
        val_prop.append(np.sqrt(np.real(eigenvals[i]))/(2*np.pi))


    if plot_result:
        fct.print_freq(val_prop[:numberMode])
        fct.plot_result(nodeList, nodeConstraint, vect_prop[:numberMode], elemList0, dofList)

    return val_prop[:numberMode], vect_prop[:numberMode], K, M, dofList

#printDataBeam = False
#printMtot = False
#printStructure = False
#printResult = True
#tmp, _, _, _, _ = ElementFini_OffShoreStruct(2, 8, printDataBeam, printMtot, printStructure, printResult)

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

    q = np.zeros((len(t), nbreDof))
    for i in range(Mode_nbr):
        for j in range(nbreDof):
            for k in range(len(t)):
                q[k][j] += eta[i][k] * eigenvectors[i][j]
                q[k][j] -= phi[i][k] * eigenvectors[i][j] / eigenvalues[i] ** 2

        progress = i / (Mode_nbr - 1) * 100
        print('\rProgress Acceleration Method: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)

    q += (np.linalg.inv(K) @ p).T

    end = time.time()
    delta = end - start
    print(f"\nTotal execution time: {delta:.2f} seconds")

    return q

def TransientResponse(numberMode, t, pas, verbose):
    numberElem = 3
    EigenValues, EigenVectors, K, M, DofList = ElementFini_OffShoreStruct(numberElem, numberMode, False, False, False, False)

    print("DOF 17 " ,DofList[17]-25)
    print(DofList[0])
    print("DOF 21 ", DofList[21] - 25)
    EigenValues = np.array(EigenValues) * np.pi * 2
    mu = fct.Mu(EigenVectors, M)
    Alpha, Beta = fct.CoefficientAlphaBeta(EigenValues)
    C = fct.DampingMatrix(Alpha, Beta, K, M)
    DampingRatio = fct.DampingRatios(Alpha, Beta, EigenValues)

    p = fct.P(len(EigenVectors[0]), data.ApplNode, DofList, t)
    phi = fct.Phi(EigenVectors, mu, p)

    eta = fct.compute_eta(EigenVectors, EigenValues, DampingRatio, phi, t, pas)
    print("eta", np.array(eta).shape)

    qDisp = ModeDisplacementMethod(EigenVectors, eta, t)
    qAcc = ModeAccelerationMethod(EigenVectors, EigenValues, eta, K, phi, p, t)

    if verbose:
        fct.print_TransientResponse(qAcc, qDisp, t, DofList)

    return qAcc, qDisp, C, p, K, M, DofList

def ConvergenceTransientResponse(numberMaxMode):
    numberMode_list = np.arange(2, numberMaxMode + 1, 1)
    responseAcc = []
    responseDisp = []
    t_final = 5
    t = np.linspace(0, t_final, 1001)
    dofList = []

    for numberMode in numberMode_list:
        qAcc, qDisp, _, _, _, _, DofList = TransientResponse(numberMode, t, False)

        dofList = DofList
        responseAcc.append(qAcc)
        responseDisp.append(qDisp)

    fct.print_ConvergenceTransientResponse(numberMaxMode, numberMode_list, responseDisp, responseAcc, dofList, t)

def Newmark(M, C, K, p, h,t):
    gamma = data.gamma
    beta = data.beta
    qdisp = np.zeros((len(t), len(M)))
    qvel  = np.zeros((len(t), len(M)))
    qacc  = np.zeros((len(t), len(M)))

    S = fct.compute_S(M, h, gamma, C, beta, K)
    S_inv = scipy.linalg.inv(S)

    #qacc[0] = np.linalg.solve(M, p.T[0] - C @ qvel[0].T - K @ qdisp[0].T)

    start = time.time()
    for i in range(1, len(t)):
        qvel[i] = qvel[i - 1] + (1 - gamma) * h * qacc[i - 1]
        qdisp[i] = qdisp[i-1] + h * qvel[i-1] + (0.5 - beta) * (h**2) * qacc[i-1]

        #qacc[i] = np.linalg.solve(S, p.T[i] - C @ qvel[i].T - K @ qdisp[i].T)
        qacc[i] = S_inv @ (p.T[i] - C @ qvel[i].T - K @ qdisp[i].T)

        qvel[i] =  qvel[i] + h * gamma * qacc[i]
        qdisp[i] = qdisp[i]  + (h**2) * beta * qacc[i]

        progress = i / (len(t)-1) * 100
        print('\rProgress Newmark: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)
    end = time.time()
    execution_time = end - start
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

    return qdisp, qvel, qacc

#ElementFini_OffShoreStruct(3, 8, False)
#EtudeConvergence(15)

NumberMode = 8
tfin = 10
pas = 0.001
t = np.arange(0, tfin, pas)
qAcc, qDisp, C, p, K, M, DofList = TransientResponse(NumberMode, t, pas,False)
# ConvergenceTransientResponse(5)
qDispN, qVelN, qAccN = Newmark(M, C, K, p, pas, t)
#fct.print_TransientResponse(qAcc, qDisp, t, DofList)
#fct.print_NewmarkResponse(qDispN, t, DofList)


fct.printResult(qAcc, qDisp, qDispN, t, DofList)



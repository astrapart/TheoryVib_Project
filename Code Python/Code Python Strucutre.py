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

    q = np.zeros((len(t), nbreDof))
    for i in range(Mode_nbr):
        for j in range(nbreDof):
            for k in range(len(t)):
                q[k][j] += eta[i][k] * eigenvectors[i][j]

        progress = i / (Mode_nbr - 1) * 100
        print('\rProgress Displacement Method: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)

    end = time.time()
    delta = end - start
    print(f"\nTotal execution time: {delta:.2f} seconds")

    return q

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
    EigenValues, EigenVectors, K, M, DofList = ElementFini_OffShoreStruct(numberElem, numberMode, False, False, False, True)

    mu = fct.Mu(EigenVectors, M)
    Alpha, Beta = fct.CoefficientAlphaBeta(EigenValues)
    C = fct.DampingMatrix(Alpha, Beta, K, M)
    DampingRatio = fct.DampingRatios(Alpha, Beta, EigenValues)
    p = fct.P(len(EigenVectors[0]), data.ApplNode, DofList, t)
    phi = fct.Phi(EigenVectors, mu, p)
    eta = fct.compute_eta(EigenVectors, EigenValues, DampingRatio, phi, t, pas)

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
    #S_inv = scipy.linalg.inv(S)

    #qacc[0] = np.linalg.solve(M, p.T[0] - C @ qvel[0].T - K @ qdisp[0].T)

    start = time.time()
    for i in range(1, len(t)):
        qvel[i] = qvel[i - 1] + (1 - gamma) * h * qacc[i - 1]
        qdisp[i] = qdisp[i-1] + h * qvel[i-1] + (0.5 - beta) * (h**2) * qacc[i-1]

        qacc[i] = np.linalg.solve(S, p.T[i] - C @ qvel[i].T - K @ qdisp[i].T)
        #qacc[i] = S_inv @ (p.T[i] - C @ qvel[i].T - K @ qdisp[i].T)

        qdisp[i] =  qdisp[i] + h * gamma * qacc[i]
        qvel[i] = qvel[i]  + (h**2) * beta * qacc[i]

        progress = i / (len(t)-1) * 100
        print('\rProgress Newmark: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)
    end = time.time()
    execution_time = end - start
    print(f"\nTotal execution time: {execution_time:.2f} seconds")

    return qdisp, qvel, qacc

#ElementFini_OffShoreStruct(3, 8, False)
#EtudeConvergence(15)

NumberMode = 8
tfin = 5
pas = 1/1000
t = np.arange(0, tfin, pas)
#qAcc, qDisp, C, p, K, M, DofList = TransientResponse(NumberMode, t, pas,False)
# ConvergenceTransientResponse(5)
#qDispN, qVelN, qAccN = Newmark(M, C, K, p, pas, t)
#fct.print_TransientResponse(qAcc, qDisp, t, DofList)
#fct.print_NewmarkResponse(qDispN, t, DofList)


def F(t):
    return data.A * np.sin(2*np.pi*data.f * t)


def TestTransientResponse(numberMode, t, verbose):

    numberElem = 3
    eigenValues, eigenVectors, K, M, dofList = ElementFini_OffShoreStruct(numberElem, numberMode, False, False, False, False)

    A = 0.5 * np.array([[eigenValues[0], 1 / eigenValues[0]],
                        [eigenValues[1], 1 / eigenValues[1]]])
    b = [0.5/100, 0.5/100]
    #alpha, beta = np.linalg.solve(A, b)
    alpha = 0.00177923
    beta = 0.0140498
    #print(alpha, beta)

    xAppl = dofList[17][0]
    yAppl = dofList[17][1]
    p = np.zeros((len(K), len(t)))
    for i in range(len(t)):
        p[xAppl][i] = F(t[i]) * np.sqrt(2) / 2
        p[yAppl][i] = F(t[i]) * np.sqrt(2) / 2

    phi = np.zeros((numberMode, len(t)))
    eta = np.zeros((numberMode, len(t)))

    qDisp = np.zeros((len(t), len(K)))
    qAcc = np.zeros((len(t), len(K)))
    for r in range(numberMode):
        mur = eigenVectors[r].T @ M @ eigenVectors[r]
        er = 0.5 * (alpha * eigenValues[r] + beta / eigenValues[r])
        wr = eigenValues[r]
        wrd = wr * np.sqrt(1 - er)

        phir = eigenVectors[r].T @ p / mur
        phi[r] = phir
        h = np.exp(-er * wr * t) * np.sin(wrd * t) / wrd

        etar = np.convolve(phir, h)[:len(t)]
        eta[r] = etar

        qDisp += np.dot(etar.reshape((len(etar), 1)), eigenVectors[r].reshape((1, len(eigenVectors[r]))))
        qAcc += qDisp - (phir.reshape((len(phir), 1)) / eigenValues[r] ** 2) @ eigenVectors[r].reshape((1, len(eigenVectors[r])))

    qAcc -= (np.linalg.inv(K) @ p).T

    C = alpha * K + beta * M

    if verbose:
        fct.print_TransientResponse(qAcc, qDisp, t, dofList)

    return qDisp, qAcc, K, M, C, dofList


#qDips, qAcc, K, M, C, DofList = TestTransientResponse(8, t, True)


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

        def reduced(Matrix):
            for red in range(2):
                Matrix = np.delete(Matrix, 11, red)
                Matrix = np.delete(Matrix, 10, red)
                Matrix = np.delete(Matrix, 4, red)
                Matrix = np.delete(Matrix, 3, red)

            return Matrix

        KelRed = reduced(Kel.copy())
        MelRed = reduced(Mel.copy())

        T = fct.create_T(coord1, coord2, l)

        TRed = reduced(T.copy())

        Kes = np.transpose(T) @ Kel @ T
        Mes = np.transpose(T) @ Mel @ T

        KesRed = np.transpose(TRed) @ KelRed @ TRed
        MesRed = np.transpose(TRed) @ MelRed @ TRed

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


printDataBeam = False
printMtot = False
printStructure = False
printResult = False
#tmp, _, _, _, _ = ElementFini_OffShoreStructReduced(12, 8, printDataBeam, printMtot, printStructure, printResult)

M = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
              [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 110, 111],
              [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 210, 211],
              [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311],
              [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 410, 411],
              [50, 51, 52, 53 ,54, 55, 56, 57, 58, 59, 510, 511],
              [60, 61, 62, 63, 64, 65, 66, 67 ,68 ,69, 610, 611],
              [70, 71, 72, 73, 74, 75, 76, 77, 78, 79 ,710, 711],
              [80, 81, 82 ,83, 84, 85, 86, 87, 88, 89, 810, 811],
              [90, 91, 92, 93, 94, 95 ,96 ,97 ,98, 99, 910, 911],
              [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 1010, 1011],
              [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 1110, 1111]])
K = np.array([
    [1,2,3,4,5,6,7,8,9,10,11,12],
    [1,2,3,4,5,6,7,8,9,10,11,12],
    [1,2,3,4,5,6,7,8,9,10,11,12],

    [1,2,3,4,5,6,7,8,9,10,11,12],
    [1,2,3,4,5,6,7,8,9,10,11,12],

    [1,2,3,4,5,6,7,8,9,10,11,12],


    [1,2,3,4,5,6,7,8,9,10,11,12],
    [1,2,3,4,5,6,7,8,9,10,11,12],
    [1,2,3,4,5,6,7,8,9,10,11,12],

    [1,2,3,4,5,6,7,8,9,10,11,12],
    [1,2,3,4,5,6,7,8,9,10,11,12],

    [1,2,3,4,5,6,7,8,9,10,11,12]
    ])

MRR, MRC, MCR, MCC = M.copy(), M.copy(), M.copy(), M.copy()
KRR, KRC, KCR, KCC = M.copy(), M.copy(), M.copy(), M.copy()

for i in range(len(M)//6, 0, -1):
    print(i)
    MRR = np.delete(MRR, 6*i - 2, 0)
    MRR = np.delete(MRR, 6*i - 2, 1)
    MRR = np.delete(MRR, 6*i - 3, 0)
    MRR = np.delete(MRR, 6*i - 3, 1)

    MCC = np.delete(MCC, 6*i - 1, 0)
    MCC = np.delete(MCC, 6*i - 1, 1)
    MCC = np.delete(MCC, 6*i - 4, 0)
    MCC = np.delete(MCC, 6*i - 4, 1)
    MCC = np.delete(MCC, 6*i - 5, 0)
    MCC = np.delete(MCC, 6*i - 5, 1)
    MCC = np.delete(MCC, 6*i - 6, 0)
    MCC = np.delete(MCC, 6*i - 6, 1)

    MRC = np.delete(MRC, 6*i - 2, 0)
    MRC = np.delete(MRC, 6*i - 3, 0)
    MRC = np.delete(MRC, 6*i - 1, 1)
    MRC = np.delete(MRC, 6*i - 4, 1)
    MRC = np.delete(MRC, 6*i - 5, 1)
    MRC = np.delete(MRC, 6*i - 6, 1)

    MCR = np.delete(MCR, 6*i - 2, 1)
    MCR = np.delete(MCR, 6*i - 3, 1)
    MCR = np.delete(MCR, 6*i - 1, 0)
    MCR = np.delete(MCR, 6*i - 4, 0)
    MCR = np.delete(MCR, 6*i - 5, 0)
    MCR = np.delete(MCR, 6*i - 6, 0)

print(MRR)
print(MCC)

"""
nbrelem = 2
MRR, KRR = np.zeros((nbrelem * 4, nbrelem * 4)), np.zeros((nbrelem * 4, nbrelem * 4))
MCC, KCC = np.zeros((nbrelem * 2, nbrelem * 2)), np.zeros((nbrelem * 2, nbrelem * 2))
MRC, KRC = np.zeros((nbrelem * 4, nbrelem * 2)), np.zeros((nbrelem * 4, nbrelem * 2))
MCR, KCR = np.zeros((nbrelem * 2, nbrelem * 4)), np.zeros((nbrelem * 2, nbrelem * 4))

listReduce = np.array([0, 1, 2, 6])
listAREt = np.array([3, 4])
for i in range(nbrelem):
    for j in range(nbrelem):
        MRR[4 * i, 4 * j] = M[6 * i, 6 * j]
        MRR[4 * i, 4 * j + 1] = M[6 * i, 6 * j + 1]
        MRR[4 * i, 4 * j + 2] = M[6 * i, 6 * j + 2]
        MRR[4 * i, 4 * j + 3] = M[6 * i, 6 * j + 5]
        MRR[4 * i + 1, 4 * j] = M[6 * i + 1, 6 * j]
        MRR[4 * i + 1, 4 * j + 1] = M[6 * i + 1, 6 * j + 1]
        MRR[4 * i + 1, 4 * j + 2] = M[6 * i + 1, 6 * j + 2]
        MRR[4 * i + 1, 4 * j + 3] = M[6 * i + 1, 6 * j + 5]
        MRR[4 * i + 2, 4 * j] = M[6 * i + 2, 6 * j]
        MRR[4 * i + 2, 4 * j + 1] = M[6 * i + 2, 6 * j + 1]
        MRR[4 * i + 2, 4 * j + 2] = M[6 * i + 2, 6 * j + 2]
        MRR[4 * i + 2, 4 * j + 3] = M[6 * i + 2, 6 * j + 5]
        MRR[4 * i + 3, 4 * j] = M[6 * i + 5, 6 * j]
        MRR[4 * i + 3, 4 * j + 1] = M[6 * i + 5, 6 * j + 1]
        MRR[4 * i + 3, 4 * j + 2] = M[6 * i + 5, 6 * j + 2]
        MRR[4 * i + 3, 4 * j + 3] = M[6 * i + 5, 6 * j + 5]

        MCC[2 * i, 2 * j] = M[6 * i + 3, 6 * j + 3]
        MCC[2 * i, 2 * j + 1] = M[6 * i + 3, 6 * j + 4]
        MCC[2 * i + 1, 2 * j] = M[6 * i + 4, 6 * j + 3]
        MCC[2 * i + 1, 2 * j + 1] = M[6 * i + 4, 6 * j + 4]

        MRC[4 * i, 2 * j] = M[6 * i, 6 * j + 3]
        MRC[4 * i, 2 * j + 1] = M[6 * i, 6 * j + 4]
        MRC[4 * i + 1, 2 * j] = M[6 * i + 1, 6 * j + 3]
        MRC[4 * i + 1, 2 * j + 1] = M[6 * i + 1, 6 * j + 4]
        MRC[4 * i + 2, 2 * j] = M[6 * i + 2, 6 * j + 3]
        MRC[4 * i + 2, 2 * j + 1] = M[6 * i + 2, 6 * j + 4]
        MRC[4 * i + 3, 2 * j] = M[6 * i + 5, 6 * j + 3]
        MRC[4 * i + 3, 2 * j + 1] = M[6 * i + 5, 6 * j + 4]

        MRC[2 * i, 4 * j] = M[6 * i + 3, 6 * j]
        MRC[2 * i, 4 * j + 1] = M[6 * i + 3, 6 * j + 1]
        MRC[2 * i, 4 * j + 2] = M[6 * i + 3, 6 * j + 2]
        MRC[2 * i, 4 * j + 3] = M[6 * i + 3, 6 * j + 5]
        MRC[2 * i + 1, 4 * j] = M[6 * i + 4, 6 * j]
        MRC[2 * i + 1, 4 * j + 1] = M[6 * i + 4, 6 * j + 1]
        MRC[2 * i + 1, 4 * j + 2] = M[6 * i + 4, 6 * j + 2]
        MRC[2 * i + 1, 4 * j + 3] = M[6 * i + 4, 6 * j + 5]

print(MRR)
print(KRR)
"""
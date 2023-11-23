"""
########################################################################################################################
IMPORT
########################################################################################################################
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import data

"""
########################################################################################################################
Fonction create
########################################################################################################################
"""


def create_elemList(elemList0, nodeList0, numberElem):
    elemList = []
    nodeList = nodeList0.copy()

    for elem in elemList0:
        i = elem[0]-1
        j = elem[1]-1
        propriety = elem[2]

        if propriety != 2:
            current = i + 1
            len_x = abs(nodeList[i][0] - nodeList[j][0]) / numberElem
            if nodeList[i][0] > nodeList[j][0]:
                len_x *= -1
            len_y = abs(nodeList[i][1] - nodeList[j][1]) / numberElem
            if nodeList[i][1] > nodeList[j][1]:
                len_y *= -1
            len_z = abs(nodeList[i][2] - nodeList[j][2]) / numberElem
            if nodeList[i][2] > nodeList[j][2]:
                len_z *= -1

            for m in range(numberElem):
                new = len(nodeList) + 1
                if m != (numberElem - 2):
                    elemList.append([current, new, propriety])
                    nodeList.append([nodeList[current-1][0] + len_x, nodeList[current-1][1] + len_y,
                                 nodeList[current-1][2] + len_z])

                    current = new
                else:
                    elemList.append([new, j+1, propriety])
        else:
            elemList.append(elem)

    return np.array(elemList), nodeList


def add_nodes(ElemList0, NodeList0, numberElem):
    ElemList = []
    NodeList = NodeList0.copy()

    for elem in ElemList0:
        node1 = elem[0]
        node2 = elem[1]
        propriety = elem[2]

        current = node1
        if propriety != 2:

            coord1 = np.array(NodeList[node1])
            coord2 = np.array(NodeList[node2])

            delta = (coord2 - coord1) / numberElem

            for i in range(numberElem - 1):
                newNode = len(NodeList)
                coord = NodeList[current]

                ElemList.append([current, newNode, propriety])
                NodeList.append([coord[0] + delta[0], coord[1] + delta[1], coord[2] + delta[2]])

                current = newNode

            ElemList.append([current, node2, propriety])

        else:
            ElemList.append(elem)

    return ElemList

    
def create_dofList(nodeList):
    dofList = []
    dof = 1
    for i in range(len(nodeList)):
        tmp = []
        for j in range(6):
            tmp.append(dof)
            dof += 1
        dofList.append(tmp)
    return np.array(dofList)


def create_locel(elemList, dofList):
    locel = []
    for i in range(len(elemList)):
        dofNode1 = dofList[elemList[i][0]-1]
        dofNode2 = dofList[elemList[i][1]-1]
        locel.append(np.concatenate((dofNode1, dofNode2), axis=0))
    
    return np.array(locel)

def create_T(coord1, coord2, l):
    P1 = np.array(coord1)
    P2 = np.array(coord2)
    P3 = np.array([1, 2, 20])

    d2 = P2 - P1
    d3 = P3 - P1

    ex = d2 / l
    ey = np.cross(d3, d2) / np.linalg.norm(np.cross(d3, d2))
    ez = np.cross(ex, ey)

    eX = [1, 0, 0]
    eY = [0, 1, 0]
    eZ = [0, 0, 1]

    R = [[np.dot(eX, ex), np.dot(eY, ex), np.dot(eZ, ex)],
         [np.dot(eX, ey), np.dot(eY, ey), np.dot(eZ, ey)],
         [np.dot(eX, ez), np.dot(eY, ez), np.dot(eZ, ez)]]

    T = scipy.linalg.block_diag(R, R, R, R)

    return np.array(T)


def create_Kel(E, A, Jx, Iy, Iz, G, l) :
    Kel = [[E*A/l],
           [  0, 12*E*Iz/(l*l*l)],
           [  0,       0,       12*E*Iy/(l*l*l)],
           [  0,       0,              0,       G*Jx/l],
           [  0,       0,        -6*E*Iy/(l*l),    0,   4*E*Iy/l],
           [  0,   6*E*Iz/(l*l),       0,          0,       0,     4*E*Iz/l],
           [-E*A/l,    0,              0,          0,       0,         0,         E*A/l],
           [  0, -12*E*Iz/(l*l*l),     0,          0,       0,   -6*E*Iz/(l*l),     0,     12*E*Iz/(l*l*l)],
           [  0,       0,       -12*E*Iy/(l*l*l),  0, 6*E*Iy/(l*l),    0,           0,            0,        12*E*Iy/(l*l*l)],
           [  0,       0,              0,       -G*Jx/l,    0,         0,           0,            0,               0,          G*Jx/l],
           [  0,       0,         -6*E*Iy/(l*l),   0,    2*E*Iy/l,     0,           0,            0,          6*E*Iy/(l*l),       0,   4*E*Iy/l],
           [  0,   6*E*Iz/(l*l),       0,          0,       0,     2*E*Iz/l,        0,      -6*E*Iz/(l*l),         0,             0,       0,    4*E*Iz/l]]

    for i in range(len(Kel)):
        for j in range(i+1, len(Kel)):

            Kel[i].append(Kel[j][i])

    return np.array(Kel)


def create_Mel(m, r, l) :

    Mel = [[1/3],
           [0, 13/35],
           [0, 0, 13/35],
           [0, 0, 0, r*r/3],
           [0, 0, -11*l/210, 0, l*l/105],
           [0, 11*l/210, 0, 0, 0, l*l/105],
           [1/6, 0, 0, 0, 0, 0, 1/3],
           [0, 9/70, 0, 0, 0, 13*l/420, 0, 13/35],
           [0, 0, 9/70, 0, -13*l/420, 0, 0, 0, 13/35],
           [0, 0, 0, r*r/6, 0, 0, 0, 0, 0, r*r/3],
           [0, 0, 13*l/420, 0, -l*l/140, 0, 0, 0, 11*l/210, 0, l*l/105],
           [0, -13*l/420, 0, 0, 0, -l*l/140, 0, -11*l/210, 0, 0, 0, l*l/105]]

    for i in range(len(Mel)):
        for j in range(i+1, len(Mel)):
            Mel[i].append(Mel[j][i])

    return m * np.array(Mel)


def Add_const_emboit(nodeConstraint, dofList, M, K):

    dofConstraint = []
    for node in nodeConstraint:
        dofConstraint = np.concatenate((dofConstraint, dofList[node]))

    dofConstraint = np.flip(np.sort(dofConstraint))
    dofConstraint = dofConstraint.astype(int)

    for dof in dofConstraint:
        M = np.delete(M, dof, 0)
        M = np.delete(M, dof, 1)

        K = np.delete(K, dof, 0)
        K = np.delete(K, dof, 1)

    return M, K


def Add_lumped_mass(nodeLumped, dofList, M):
    mass = data.mass_lumped
    J = data.node_lumped_J

    count = 0
    for tmp in dofList[nodeLumped-1]:
        i = tmp-1
        if count <= 2:
            M[i][i] += mass
        else:
            M[i][i] += J
        count += 1

    return M

"""
########################################################################################################################
Fonction Calculate
########################################################################################################################
"""


def calculate_length(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)


def properties(type_beam, l):
    rho = data.density_beam                                            # [kg/m3]
    v = data.poisson_ratio                                             # [-]
    E = data.young_mod                                                 # [Pa]
    D = data.diam_beam[type_beam]                                      # [m]
    A = np.pi * (D*D - (D - 2 * data.thickness_beam) ** 2)/4           # [m2]
    Ix = (np.pi / 64) * (D ** 4 - (D - 2 * data.thickness_beam) ** 4)  # [m4]
    Jx = Ix * 2                                                        # [m4]
    Iy = Ix                                                            # [m4]
    Iz = Iy                                                            # [m4]

    if type_beam == 2:
        rho = rho * 10 ** (-4)                                         # [kg/m3]
        A = A * 10 ** (-2)                                             # [m2]
        E = E * 10 ** 4                                                # [Pa]
        Jx = Jx*10**4                                                  # [m4]
        Iy = Iy*10**4                                                  # [m4]
        Iz = Iz*10**4                                                  # [m4]

    m = rho * A * l                                                    # [kg]
    G = E / (2 * (1 + v))                                              # [GPa]
    r = np.sqrt(Jx / A)                                                # [m]

    return rho, v, E, A, m, Jx, Iy, Iz, G, r


def calculate_mtot_rigid(M):

    ue = np.zeros(len(M))
    for i in range(len(M)//6):
        ue[i*6] = 1

    return np.transpose(ue) @ M @ ue


def F(t):
    return data.A * np.sin(2*np.pi*data.f * t)


def P(n, dofList, t, K):
    p = np.zeros((n, len(t)))

    xAppl = dofList[17][0] - 25
    yAppl = dofList[17][1] - 25

    for i in range(len(t)):
        p[xAppl][i] = -F(t[i]) * np.sqrt(2) / 2
        p[yAppl][i] = F(t[i]) * np.sqrt(2) / 2

    return p


def Phi(x, mu, p):
    phi = np.zeros((len(x), len(p[0])))
    for i in range(len(x)):
        phi[i] = x[i].T @ p / mu[i]

    return phi


def Wrd(wr, er):
    return wr*np.sqrt(1-(er**2))


def H(er, wr, wrd, t):
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
        mu.append(eigenvect.T @ M @ eigenvect)
    return np.array(mu)


def DampingRatios(alpha, beta, eigenValues):
    dampingRatios = np.zeros(len(eigenValues))

    for i in range(len(dampingRatios)):
        dampingRatios[i] = 0.5 * (alpha * eigenValues[i] + beta / eigenValues[i])

    return dampingRatios


def compute_eta(Eigenvectors, EigenValues, DampingRatio, phi, t, pas):
    eta = np.zeros((len(Eigenvectors), len(t)))
    for r in range(len(Eigenvectors)):
        er = DampingRatio[r]
        wr = EigenValues[r]
        wrd = Wrd(wr, er)
        h = H(er, wr, wrd, t)
        convolution = np.convolve(phi[r], h)[:len(t)]
        eta[r] = pas * convolution

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


def compute_S(M, h, gamma, C, beta, K):
    return M + h*gamma*C + (h**2) * beta * K


def decompositionMatrix(K, M):
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

        progress = (len(M) // 6 - i) / (len(M) // 6 - 1) * 100
        print('\rProgress Decomposition: [{:<50}] {:.2f}%'.format('=' * int(progress / 2), progress), end='', flush=True)
    print("\n")

    return MRR, MRC, MCR, MCC, KRR, KRC, KCR, KCC


"""
########################################################################################################################
Fonctions PLOT 
########################################################################################################################
"""


def plot_structure(elemList, nodeList):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for elem in elemList:
        elem_1 = nodeList[elem[0]-1]
        elem_2 = nodeList[elem[1]-1]
        ax.plot([elem_1[0], elem_2[0]], [elem_1[1], elem_2[1]], [elem_1[2], elem_2[2]], c='b')

    for node in nodeList:
        x = node[0]
        y = node[1]
        z = node[2]
        ax.scatter(x, y, z, c='g')

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)

    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')

    ax.set_box_aspect(aspect=(5, 5, 10))
    ax.grid(False)

    plt.show()

def plot_result(nodeList, nodeConstraint, eigenvects, elemList, dofList):
    factor = 5
    nbrConstraintNode = len(nodeConstraint)

    for i in range(len(eigenvects)):
        newNodeList = []
        for j in range(len(nodeList)):
            x, y, z = nodeList[j][0], nodeList[j][1], nodeList[j][2]

            if j + 1 in nodeConstraint:
                newNodeList.append([x, y, z])

            else:
                dx, dy, dz = eigenvects[i][6 * (j - nbrConstraintNode)], eigenvects[i][6 * (j - nbrConstraintNode) + 1], \
                             eigenvects[i][6 * (j - nbrConstraintNode) + 2]

                coord = [x + dx * factor, y + dy * factor, z + dz * factor]

                newNodeList.append(coord)

        ax = plt.axes(projection='3d')
        ax.set_box_aspect((10, 10, 30))
        #ax.set_title(f"Shape mode {i + 1}")
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        for elem in elemList:
            node1 = nodeList[elem[0] - 1]
            node2 = nodeList[elem[1] - 1]
            newNode1 = newNodeList[elem[0] - 1]
            newNode2 = newNodeList[elem[1] - 1]

            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], '--',  c='black')
            ax.plot([newNode1[0], newNode2[0]], [newNode1[1], newNode2[1]], [newNode1[2], newNode2[2]], c='r')

        name = f"Mode Shape {i + 1}"
        ax.legend(loc='upper left', labels=["Jacket", name ])
        #plt.savefig(f"ModeShape{i+1}.pdf")
        plt.show()


def print_freq(list_eign):

    for i in range(len(list_eign)):
        #print("La fréquence pour la valeur propre", i, "vaut :", f, "Hz")
        # 0.44 valeur propre 1
        # 0.45 valeur propre 2
        # 0.9  valeur propre 3
        print("La fréquence pour la valeur propre {index} vaut : {val:>8.5f} [Hz]".format(index=i+1, val=list_eign[i]))

    return


def print_freqComparaison(eigenvals, eigenvalsGI, eigenvalsCB):
    print("                                               FE [HZ]    GI [Hz]    [%]      CB [HZ]    [%]")
    for i in range(len(eigenvals)):
        #print("La fréquence pour la valeur propre", i, "vaut :", f, "Hz")
        # 0.44 valeur propre 1
        # 0.45 valeur propre 2
        # 0.9  valeur propre 3
        error = (eigenvals[i] - eigenvalsGI[i]) / eigenvals[i] * 100
        error1 = (eigenvals[i] - eigenvalsCB[i]) / eigenvals[i] * 100
        print("La fréquence pour la valeur propre {index} vaut : {val:>10f} {val1:>10f}  {error:>3f}  {val2:>10f}  {error1:>3f}".format(index=i+1, val=eigenvals[i], val1=eigenvalsGI[i], error=error, val2=eigenvalsCB[i], error1=error1))

    return


def print_data_beam(node1, node2, type_beam, rho, v, E, A, m, Jx, Iy, Iz, G, r,l) :
    print("--------Propriety {beam} for elem [{node1}, {node2}]--------".format(beam=type_beam, node1=node1 + 1,
                                                                                node2=node2 + 1))
    print("rho = {rho}, E = {E}  A = {A}  l = {l}".format(rho=rho, E=E, A=A, l=l))
    print("G = {G}  r = {r}  m = {m}".format(G=G, r=r, m=m))
    print("Jx = {Jx}  I = {Iy}".format(Jx=Jx, Iy=Iy))
    print()


def ConvergencePlot():
    Result = [[0.4437535, 0.45433177, 0.97293389, 7.05536334, 7.40416045, 15.94143563, 20.54892234, 22.10797568],
              [0.44375284, 0.45433049, 0.97293385, 7.05444093, 7.40314732, 15.94072114, 20.52106343, 22.07593765],
              [0.44374422, 0.45432947, 0.97293014, 7.05437772, 7.40286752, 15.94055679, 20.51636776, 22.07033115],
              [0.44375049, 0.45432843, 0.97293498, 7.05422351, 7.40294721, 15.94049676, 20.51505775, 22.06887598],
              [0.44375399, 0.45432601, 0.97292409, 7.05419012, 7.4030251, 15.94046982, 20.51458456, 22.0682478],
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
        ax = fig.add_subplot(2, 4, i + 1)
        for j in range(len(TestElem) - 1):
            ax.plot([TestElem[j], TestElem[j + 1]], [Result[j][i], Result[j + 1][i]], c='b')

        ax.grid()
        ax.set_title(f"{i + 1} eigenvalues")

    plt.show()

    """  Plot du temps d'exécution, pas très intéressant 
    plt.figure()
    for i in range(len(TestElem) - 1):
        plt.plot([TestElem[i], TestElem[i + 1]], [Time[i], Time[i + 1]], c='b')

    plt.grid()
    plt.title("Evolution of time per number of beam")
    plt.show()
    """


def print_ConvergenceTransientResponse(numberModeList, responseDisp, responseAcc, DofList, t):
    fig = plt.figure(figsize=(14, 6))
    for i in range(len(numberModeList)):
        DisplacementNodeX = responseDisp[i][:, DofList[17][0]-25]
        DispNode = -np.sqrt(2) * DisplacementNodeX

        AccelerationNodeX = responseAcc[i][:, DofList[17][0] - 25]
        AccNode = -np.sqrt(2) * AccelerationNodeX

        plt.plot(t, AccNode * 1000, label=f'Mode Acc {numberModeList[i]}')
        plt.plot(t, DispNode * 1000, label=f'Mode Dipl {numberModeList[i]}')

    plt.title("Displacement of the Node")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(14, 6))
    for i in range(len(numberModeList)):
        DisplacementRotorX = responseDisp[i][:, DofList[21][0] - 25]
        DispRotor = (-np.sqrt(2)) * DisplacementRotorX

        AccelerationRotorX = responseAcc[i][:, DofList[21][0] - 25]
        AccRotor = (-np.sqrt(2)) * AccelerationRotorX

        plt.plot(t, AccRotor * 1000, label=f'Mode Acc {numberModeList[i]}')
        plt.plot(t, DispRotor * 1000, label=f'Mode Dipl {numberModeList[i]}')

    plt.title("Displacement of the Rotor")
    plt.legend()
    plt.show()


def print_TransientResponse(qAcc, qDisp, t, DofList):
    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(211)
    DisplacementNodeX = qDisp[:, DofList[17][0]-25]
    DispNode = (-np.sqrt(2)) * DisplacementNodeX

    AccelerationNodeX = qAcc[:, DofList[17][0]-25]
    AccNode = (-np.sqrt(2)) * AccelerationNodeX

    ax1.plot(t, AccNode * 1000, label='Mode Acc', c='r')
    ax1.plot(t, DispNode * 1000, label='Mode Depl', c='blue')
    ax1.legend()
    ax1.set_title("Displacement of the Node")

    ax2 = fig.add_subplot(212)
    DisplacementRotorX = qDisp[:, DofList[21][0] - 25]
    DispRotor = (-np.sqrt(2)) * DisplacementRotorX

    AccelerationRotorX = qAcc[:, DofList[21][0] - 25]
    AccRotor = (-np.sqrt(2)) * AccelerationRotorX

    ax2.plot(t, AccRotor * 1000, label='Mode Acc', c='r')
    ax2.plot(t, DispRotor * 1000, label='Mode Dipl', c='blue')
    ax2.legend()
    ax2.set_title("Displacement of the Rotor")

    plt.show()


def printResult(qAcc, qDisp, qDispN, t, DofList):
    fig = plt.figure(figsize=(10, 7))

    ax1 = fig.add_subplot(211)
    DisplacementNodeX = qDisp[:, DofList[17][0]-25]
    DispNode = (-np.sqrt(2)) * DisplacementNodeX

    AccelerationNodeX = qAcc[:, DofList[17][0]-25]
    AccNode = (-np.sqrt(2)) * AccelerationNodeX

    DisplacementNodeXNM = qDispN[:, DofList[17][0] - 25]
    DispNodeNM = (-np.sqrt(2)) * DisplacementNodeXNM

    ax1.plot(t, AccNode*1000, label = 'Mode Acc', c = 'r')
    ax1.plot(t, DispNode*1000, label = 'Mode Depl', c = 'blue')
    ax1.plot(t, DispNodeNM*1000, label='NM', c='black')

    ax1.legend()
    ax1.set_title("Displacement of the Node")

    ax2 = fig.add_subplot(212)
    DisplacementRotorX = qDisp[:, DofList[21][0]-25]
    DispRotor = (-np.sqrt(2)) * DisplacementRotorX

    AccelerationRotorX = qAcc[:, DofList[21][0]-25]
    AccRotor = (-np.sqrt(2)) * AccelerationRotorX

    DisplacementRotorXNM = qDispN[:, DofList[21][0]-25]
    DispRotorNM = (-np.sqrt(2)) * DisplacementRotorXNM

    ax2.plot(t, AccRotor*1000, label = 'Mode Acc', c='r')
    ax2.plot(t, DispRotor*1000, label = 'Mode Dipl', c='blue')
    ax2.plot(t, DispRotorNM*1000, label='NM', c='black')
    ax2.legend()
    ax2.set_title("Displacement of the Rotor")

    plt.show()
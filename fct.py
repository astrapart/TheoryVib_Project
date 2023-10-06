import matplotlib.pyplot as plt
import numpy as np
import math


def est_symetrique(matrice):
    # Vérifiez si le nombre de lignes est égal au nombre de colonnes
    if len(matrice) != len(matrice[0]):
        return False
    
    # Comparez la matrice à sa transposée
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            if matrice[i][j] != matrice[j][i]:
                return False
    return True

def plot(elemList, nodeList):
    # Créez une figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for elem in elemList:
        elem_1 = nodeList[elem[0] - 1]
        elem_2 = nodeList[elem[1] - 1]
        ax.plot([elem_1[0], elem_2[0]], [elem_1[1], elem_2[1]], [elem_1[2], elem_2[2]], c='b')

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)

    # Titres des axes
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')

    # Affichez le graphique
    plt.show()


########################################################################################
    
########################################################################################
def create_elemList(elemList0, nodeList, numberElem) :
    elemList = []

    for elem in elemList0:
        i = elem[0]
        j = elem[1]
        propriety = elem[2]

        if propriety != 2:
            current = i
            len_x = abs(nodeList[i - 1][0] - nodeList[j - 1][0]) / numberElem
            if nodeList[i - 1][0] > nodeList[j - 1][0]:
                len_x *= -1
            len_y = abs(nodeList[i - 1][1] - nodeList[j - 1][1]) / numberElem
            if nodeList[i - 1][1] > nodeList[j - 1][1]:
                len_y *= -1
            len_z = abs(nodeList[i - 1][2] - nodeList[j - 1][2]) / numberElem
            if nodeList[i - 1][2] > nodeList[j - 1][2]:
                len_z *= -1
            
            for m in range(numberElem):
                new = len(nodeList) + 1
                if m != (numberElem - 2):
                    elemList.append([current, new, propriety])
                    nodeList.append([nodeList[current - 1][0] + len_x, nodeList[current - 1][1] + len_y,
                                 nodeList[current - 1][2] + len_z, propriety])

                    current = new
            else:
                elemList.append([new, j, propriety])
    else:
        elemList.append(elem)
            
    return elemList
    
    
def create_dofList(dof, nodeList, numberElem) :
    dofList = []
    for i in range(len(nodeList * numberElem)):
        tmp = []
        for j in range(6):
            tmp.append(dof)
            dof += 1
        dofList.append(tmp)
    return dofList


def create_locel(elemList,dofList) :
    locel = []
    for i in range(len(elemList)):
        dofNode1 = dofList[elemList[i][0] - 1]
        dofNode2 = dofList[elemList[i][1] - 1]
        locel.append(dofNode1 + dofNode2)
    
    return locel

def create_T(coord1,coord2,l) :
    P1 = coord1
    P2 = coord2
    P3 = [0.5, 0.5, 0]

    d2 = [P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]]
    d3 = [P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]]

    ex = [(P2[0] - P1[0]) / l, (P2[1] - P1[1]) / l, (P2[2] - P1[2]) / l]
    ey = np.cross(d2, d3) / np.linalg.norm(np.cross(d2, d3))
    ez = np.cross(ex, ey)
    localAxe = [ex, ey, ez]

    eX = [1, 0, 0]
    eY = [0, 1, 0]
    eZ = [0, 0, 1]
    globalAxe = [eX, eY, eZ]

    R = np.zeros((3,3))
    for j in range(3):
        for k in range(3):
            R[j][k] = np.dot(globalAxe[k], localAxe[j])

    T = np.zeros((12, 12))
    for j in range(3):
        for k in range(3):
            T[j][k] = R[j][k]
            T[j + 3][k + 3] = R[j][k]
            T[j + 6][k + 6] = R[j][k]
            T[j + 9][k + 9] = R[j][k]
    return T;

def create_Kel(E,A,Jx,Iy,Iz,G,l) :
    return np.array([
        [E*A/l, 0, 0, 0, 0, 0, -E*A/l, 0, 0, 0, 0, 0],
        [0, 12*E*Iz/(l**3), 0, 0, 0, 6*E*Iz/(l**2), 0, -12*E*Iz/(l**3), 0, 0, 0, 6*E*Iz/(l**2)],
        [0, 0, 12*E*Iy/(l**3), 0, -6*E*Iy/(l**2), 0, 0, 0, -12*E*Iy/(l**3), 0, -6*E*Iy/(l**2),0],
        [0, 0, 0, G*Jx/l, 0, 0, 0, 0, 0, -G*Jx/l, 0, 0],
        [0, 0, -6*E*Iy/(l**2), 0, 4*E*Iy/l, 0, 0, 0, 6*E*Iy/(l**2), 0, 2*E*Iy/l, 0],
        [0, 6*E*Iz/(l**2), 0, 0, 0, 4*E*Iz/l, 0, -6*E*Iz/(l**2), 0, 0, 0, 2*E*Iz/l],
        [-E*A/l, 0, 0, 0, 0, 0, E*A/l, 0, 0, 0, 0, 0],
        [0, -12*E*Iz/(l**3), 0, 0, 0, -6*E*Iz/(l**2), 0, 12*E*Iz/(l**3), 0, 0, 0,-6*E*Iz/(l**2)],
        [0, 0, -12*E*Iy/(l**3), 0, 6*E*Iy/(l**2), 0, 0, 0, 12*E*Iy/(l**3), 0, 6*E*Iy/(l**2), 0],
        [0, 0, 0, -G*Jx/l, 0, 0, 0, 0, 0, G*Jx/l, 0, 0],
        [0, 0, -6*E*Iy/(l**2), 0, 2*E*Iy/l, 0, 0, 0, 6*E*Iy/(l**2), 0, 4*E*Iy/l, 0],
        [0, 6 * E*Iz/(l**2), 0, 0, 0, 2*E*Iz/l, 0, -6*E*Iz/(l**2), 0, 0, 0, 4*E*Iz/l]])


def create_Mel(m,r,l) :
    return  m * np.array([
        [1 / 3, 0, 0, 0, 0, 0, 1 / 6, 0, 0, 0, 0, 0],
        [0, 13 / 35, 0, 0, 0, 11 * l / 210, 0, 9 / 70, 0, 0, 0, -13 * l / 420],
        [0, 0, 13 / 35, 0, -11 * l / 210, 0, 0, 0, 9 / 70, 0, 13 * l / 420, 0],
        [0, 0, 0, r ** 2 / 3, 0, 0, 0, 0, 0, r ** 2 / 6, 0, 0],
        [0, 0, -11 * l / 210, 0, l ** 2 / 105, 0, 0, 0, -13 * l / 420, 0, -l ** 2 / 140, 0],
        [0, 11 * l / 210, 0, 0, 0, l ** 2 / 105, 0, 13 * l / 420, 0, 0, 0, -l ** 2 / 140],
        [1 / 6, 0, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 0, 0],
        [0, 9 / 70, 0, 0, 0, 13 * l / 420, 0, 13 / 35, 0, 0, 0, -11 * l / 210],
        [0, 0, 9 / 70, 0, -13 * l / 420, 0, 0, 0, 13 / 35, 0, 11 * l / 210, 0],
        [0, 0, 0, r ** 2 / 6, 0, 0, 0, 0, 0, r ** 2 / 3, 0, 0],
        [0, 0, 13 * l / 420, 0, -l ** 2 / 140, 0, 0, 0, 11 * l / 210, 0, l ** 2 / 105, 0],
        [0, -13 * l / 420, 0, 0, 0, -l ** 2 / 140, 0, -11 * l / 210, 0, 0, 0, l ** 2 / 105]])
    

def create_properties(mainBeam_d, othbeam_d, thickn):
    ## Main Beam
    main_beam_prop = [7800, 0.3, 210 * (10 ** 6), np.pi * ((mainBeam_d / 2) ** 2 - (mainBeam_d / 2 - thickn) ** 2),
                  mainBeam_d / 2, (mainBeam_d - 2 * thickn) / 2]

    ## Other Beam
    other_beam_prop = [7800, 0.3, 210 * (10 ** 6), np.pi * ((othbeam_d / 2) ** 2 - (othbeam_d / 2 - thickn) ** 2),
                   othbeam_d / 2, (othbeam_d - 2 * thickn) / 2]

    ## Rigid Link  
    rigid_link_prop = [main_beam_prop[0] * 10 ** 4, 0.3, main_beam_prop[2] * 10 ** 4, main_beam_prop[3] * 10 ** -2,
                   mainBeam_d / 2, (mainBeam_d - 2 * thickn) / 2]

    return [main_beam_prop, other_beam_prop, rigid_link_prop]
    
    
def Add_const_emboit(nodeConstraint,dofList, M, K):
    for node in nodeConstraint:
        for dof in dofList[node - 1]:
            for i in range(M.shape[0]):
                M[dof - 1][i] = 0
                M[i][dof - 1] = 0

                K[dof - 1][i] = 0
                K[i][dof - 1] = 0

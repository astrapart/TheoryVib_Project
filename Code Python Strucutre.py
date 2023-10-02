import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


"""
## Structure
# Affichez les points sur le graphique 3D
ax.plot([0,0], [0,0], [0,1], c='r'), ax.plot([5,5], [0,0], [0,1], c='r'), ax.plot([0,0], [5,5], [0,1], c='r'), ax.plot([5,5], [5,5], [0,1], c='r')
ax.plot([0, 0.27], [0, 0.27], [1, 9], c='r'), ax.plot([0, 0.27], [5, 4.73], [1, 9], c='r'), ax.plot([5, 4.73], [0, 0.27], [1, 9], c='r'), ax.plot([5, 4.73], [5, 4.73], [1, 9], c='r')
ax.plot([0.27, 0.51], [0.27, 0.51], [9, 17], c='r'), ax.plot([0.27, 0.51], [4.73, 4.49], [9, 17], c='r'), ax.plot([4.73, 4.49], [0.27, 0.51], [9, 17], c='r'), ax.plot([4.73, 4.49], [4.73, 4.49], [9, 17], c='r')
ax.plot([0.51, 0.75], [0.51, 0.75], [17, 25], c='r'), ax.plot([0.51, 0.75], [4.49, 4.25], [17, 25], c='r'), ax.plot([4.49, 4.25], [0.51, 0.75], [17, 25], c='r'), ax.plot([4.49, 4.25], [4.49, 4.25], [17, 25], c='r')
ax.plot([0, 0], [0, 5], [1, 1], c="b"), ax.plot([0, 5], [0, 0], [1, 1], c="b"), ax.plot([5, 5], [5, 0], [1, 1], c="b"), ax.plot([5, 0], [5, 5], [1, 1], c="b")
ax.plot([0.27, 0.27], [0.27, 4.73], [9, 9], c="b"), ax.plot([0.27, 4.73], [0.27, 0.27], [9, 9], c="b"), ax.plot([4.73, 4.73], [4.73, 0.27], [9, 9], c="b"), ax.plot([4.73, 0.27], [4.73, 4.73], [9, 9], c="b")
ax.plot([0.51, 0.51], [0.51, 4.49], [17, 17], c="b"), ax.plot([0.51, 4.49], [0.51, 0.51], [17, 17], c="b"), ax.plot([4.49, 4.49], [4.49, 0.51], [17, 17], c="b"), ax.plot([4.49, 0.51], [4.49, 4.49], [17, 17], c="b")
ax.plot([0.75, 0.75], [0.75, 4.25], [25, 25], c="b"), ax.plot([0.75, 4.25], [0.75, 0.75], [25, 25], c="b"), ax.plot([4.25, 4.25], [4.25, 0.75], [25, 25], c="b"), ax.plot([4.25, 0.75], [4.25, 4.25], [25, 25], c="b")
ax.plot([0, 0.27], [5, 0.27], [1, 9], c="b"), ax.plot([5, 0.27], [0, 0.27], [1, 9], c="b"), ax.plot([0, 4.73], [5, 4.73], [1, 9], c="b"), ax.plot([5, 4.73], [0, 4.73], [1, 9], c="b")
ax.plot([0.27, 0.51], [0.27, 4.49], [9, 17], c="b"), ax.plot([0.51, 4.73], [4.49, 4.73], [17, 9], c="b"), ax.plot([4.73, 4.49], [4.73, 0.51], [9, 17], c="b"), ax.plot([4.49, 0.27], [0.51, 0.27], [17, 9], c="b")
ax.plot([0.51, 0.75], [0.51, 4.25], [17, 25], c="b"), ax.plot([0.75, 4.49], [4.25, 4.49], [25, 17], c="b"), ax.plot([4.49, 4.25], [4.49, 0.75], [17, 25], c="b"), ax.plot([4.25, 0.51], [0.75, 0.51], [25, 17], c="b")
ax.plot([0.75,4.25], [0.75, 4.25], [25, 25], c='g'), ax.plot([0.75,4.25], [4.25, 0.75], [25, 25], c='g'), ax.plot([2.5, 2.5], [2.5, 2.5], [25, 80], c='g')
"""



tan_3 = np.tan(np.radians(3))
nodeList = [[0, 0, 0],        				# node 1
            [5, 0, 0],       				# node 2
            [0, 5, 0],       				# node 3
            [5, 5, 0],         				# node 4
            [tan_3, tan_3, 1],         		# node 5
            [5-tan_3, tan_3, 1],        	# node 6
            [tan_3, 5-tan_3, 1],      		# node 7
            [5-tan_3, 5-tan_3, 1],         	# node 8
            [9*tan_3, 9*tan_3, 9],   		# node 9
            [5-9*tan_3, 9*tan_3, 9],  		# node 10
            [9*tan_3, 5-9*tan_3, 9],  		# node 11
            [5-9*tan_3, 5-9*tan_3, 9],  	# node 12
            [17*tan_3, 17*tan_3, 17],  		# node 13
            [5-17*tan_3, 17*tan_3, 17],  	# node 14
            [17*tan_3, 5-17*tan_3, 17],  	# node 15
            [5-17*tan_3, 5-17*tan_3, 17], 	# node 16
            [25*tan_3, 25*tan_3, 25],  		# node 17
            [5-25*tan_3, 25*tan_3, 25],  	# node 18
            [25*tan_3, 5-25*tan_3, 25],  	# node 19
            [5-25*tan_3, 5-25*tan_3, 25],  	# node 20
            ]

elemList = [[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0],
            [5, 9, 0], [6, 10, 0], [7, 11, 0], [8, 12, 0],
            [9, 13, 0], [10, 14, 0], [11, 15, 0], [12, 16, 0],
            [13, 17, 0], [14, 18, 0], [15, 19, 0], [16, 20, 0],
            [5, 6, 1], [5, 7, 1], [6, 8, 1], [8, 7, 1],
            [9, 10, 1], [9, 11, 1], [11, 12, 1], [12, 10, 1],
            [13, 14, 1], [13, 15, 1], [15, 16, 1], [16, 14, 1],
            [17, 18, 1], [17, 19, 1], [19, 20, 1], [20, 18, 1],
            [9, 6, 1], [6, 12, 1], [12, 7, 1], [7, 9, 1],
            [14, 9, 1], [9, 15, 1], [15, 12, 1], [12, 14, 1],
            [17, 15, 1], [15, 20, 1], [20, 14, 1], [14, 17, 1]
            ]
def plot ():
    # Créez une figure 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for elem in elemList :
        elem_1 = nodeList[elem[0]-1]
        elem_2 = nodeList[elem[1]-1]
        ax.plot([elem_1[0],elem_2[0]], [elem_1[1],elem_2[1]], [elem_1[2],elem_2[2]], c = 'b')

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    
    # Titres des axes
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.set_zlabel('Axe Z')

    # Affichez le graphique
    #plt.show()


numberElem = 3
numberBeam = len(elemList)
for x in range(numberBeam):
    i = elemList[x][0]
    j = elemList[x][1]
    propriety = elemList[x][2]

    current = i
    len_x = abs(nodeList[i-1][0] - nodeList[j-1][0])/numberElem
    if nodeList[i-1][0] > nodeList[j-1][0]:
        len_x *= -1
    len_y = abs(nodeList[i-1][1] - nodeList[j-1][1])/numberElem
    if nodeList[i-1][1] > nodeList[j-1][1]:
        len_y *= -1
    len_z = abs(nodeList[i-1][2] - nodeList[j-1][2])/numberElem
    if nodeList[i-1][2] > nodeList[j-1][2]:
        len_z *= -1

    for m in range(numberElem):
        new = len(nodeList) + 1
        if (m != numberElem-2):
            elemList.append([current, new, propriety])
            nodeList.append([nodeList[current-1][0] + len_x, nodeList[current-1][1] + len_y, nodeList[current-1][2] + len_z, propriety])

            current = new
        else:
            elemList.append([new, j, propriety])

dofList = []
dof = 1
##TODO doit contenir tous les degrés de liberté de chaque noeud, mais il faut aussi ajouter les noeuds intermédiaire nécessaire à la simulation
#il faut donc que le nombre de ligne soit égal au nombre de noeud principaux * nombre de noeud intérmédiaire

for i in range(len(nodeList*numberElem)):
    tmp = []
    for j in range(6):
        tmp.append(dof)
        dof += 1
    dofList.append(tmp)

#il faut créer les liens entre les noeuds cad toutes les bumes principales + les bumes horizontales (secondaire) en comptant les neouds intermédiaires

locel = []
for i in range(len(elemList)):
    dofNode1 = dofList[elemList[i][0]-1]
    dofNode2 = dofList[elemList[i][1]-1]
    locel.append(dofNode1 + dofNode2)

#print(nodeList)
#print(elemList)
#print(dofList)
#print(locel)
plot()

#Define parameter [densité [kg/m3], poisson [-], young [GPa], air section [m2]] en SI
mainBeam_d = 1 #m
othbeam_d = 0.6 #m
thickn = 0.02 #m
## Main Beam
main_beam_prop = [7800, 0.3, 210*10**6, np.pi * ((mainBeam_d /2)**2-(mainBeam_d/2 - thickn)**2)]

## Other Beam
other_beam_prop = [7800, 0.3, 210*10**6, np.pi * ((othbeam_d /2)**2 - (othbeam_d/2 - thickn)**2)]

## Rigid Link  
rigid_link_prop = [main_beam_prop[0] * 10**4, 0.3, main_beam_prop[2] * 10**4, main_beam_prop[3] * 10**-2]

proprieties = [main_beam_prop, other_beam_prop, rigid_link_prop]

# q = [u,v,w,phi1,phi2,phi3, ... (*nbre de point) ]*nombre d'elem
#q = np.matrix(numberElem * 3, len(nodeList))

M = np.zeros((len(nodeList), len(nodeList)))
K = np.zeros((len(nodeList), len(nodeList)))

for elem in elemList:
    node1 = elem[0]-1
    node2 = elem[1]-1
    propriety = elem[2]

    coord1 = nodeList[node1]
    coord2 = nodeList[node2]

    l = np.sqrt((coord1[0] - coord2[0])*(coord1[0] - coord2[0]) + (coord1[1] - coord2[1])*(coord1[1] - coord2[1]) + (coord1[2] - coord2[2])*(coord1[2] - coord2[2]))
    prop = proprieties[propriety]
    m = prop[0]*prop[3]*l
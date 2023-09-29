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

elemList = [[1, 5], [2, 6], [3, 7], [4, 8],
            [5, 9], [6, 10], [7, 11], [8, 12],
            [9, 13], [10, 14], [11, 15], [12, 16],
            [13, 17], [14, 18], [15, 19], [16, 20],
            [5, 6], [5, 7], [6, 8], [8, 7],
            [9, 10], [9, 11], [11, 12], [12, 10],
            [13, 14], [13, 15], [15, 16], [16, 14],
            [17, 18], [17, 19], [19, 20], [20, 18],
            [9, 6], [6, 12], [12, 7], [7, 9],
            [14, 9], [9, 15], [15, 12], [12, 14],
            [17, 15], [15, 20], [20, 14], [14, 17]
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
    plt.show()


numberElem = 3
numberBeam = len(elemList)
for x in range(numberBeam):
    i = elemList[x][0]
    j = elemList[x][1]

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
            elemList.append([current, new])
            nodeList.append([nodeList[current-1][0] + len_x, nodeList[current-1][1] + len_y, nodeList[current-1][2] + len_z])

            current = new
        else:
            elemList.append([new, j])

dofList = []
dof = 1
for i in range(len(nodeList)):
    tmp = []
    for j in range(6):
        tmp.append(dof)
        dof += 1
    dofList.append(tmp)

#print(nodeList)
#print(elemList)
#print(dofList)
#plot()

locel = []

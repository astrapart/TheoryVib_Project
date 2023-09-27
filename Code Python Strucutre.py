import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ceci est un test github

# Créez une figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

## Structure
# Coordonnées des points
x_s = [0, 0, 5, 5, 0, 0, 5, 5, 0.47, 0.47, 4.53, 4.53, 0.89, 0.89, 4.11, 4.11, 1.31, 1.31, 3.69, 3.69]
y_s = [0, 5, 0, 5, 0, 5, 0, 5, 0.47, 4.53, 0.47, 4.53, 0.89, 4.11, 0.89, 4.11, 1.31, 3.69, 1.31, 3.69]
z_s = [0, 0, 0, 0, 1, 1, 1, 1, 9, 9, 9, 9, 17, 17, 17, 17, 25, 25, 25, 25]

## Eolienne
# Coordonnées des points
x_e = [2.5, 2.5]
y_e = [2.5, 2.5]
z_e = [25, 80]

# Affichez les points sur le graphique 3D
ax.scatter(x_s, y_s, z_s, c='r', marker='o')
ax.scatter(x_e, y_e, z_e, c='g', marker='o')

"""
# Étiquetez les points si nécessaire
for i, txt in enumerate(['(0,0,0)', '(0,5,0)', '(5,0,0)', '(5,5,0)']):
    ax.text(x[i], y[i], z[i], txt, fontsize=12)
"""
# Définissez les limites des axes
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.set_zlim(0, 90)


# Titres des axes
ax.set_xlabel('Axe X')
ax.set_ylabel('Axe Y')
ax.set_zlabel('Axe Z')

# Affichez le graphique
plt.show()


nodeList = [[0, 0, 0],         # node 1
            [5, 0, 0],         # node 2
            [0, 5, 0],         # node 3
            [5, 5, 0],         # node 4
            [0, 0, 1],         # node 5
            [5, 0, 1],         # node 6
            [0, 5, 1],         # node 7
            [5, 5, 1],         # node 8
            [0.27, 0.27, 9],   # node 9
            [4.73, 0.27, 9],   # node 10
            [0.27, 4.73, 9],   # node 11
            [4.73, 4.73, 9],   # node 12
            [0.51, 0.51, 17],  # node 13
            [4.49, 0.51, 17],  # node 14
            [0.51, 4.49, 17],  # node 15
            [4.49, 4.49, 17],  # node 16
            [0.75, 0.75, 25],  # node 17
            [4.25, 0.75, 25],  # node 18
            [0.75, 4.25, 25],  # node 19
            [4.25, 4.25, 25],  # node 20
            ]


beamList = [[1, 5], [2, 6], [3, 7], [4, 8],
            [5, 9], [6, 10], [7, 11], [8, 12],
            [9, 13], [10, 14], [11, 15], [12, 16],
            [13, 17], [14, 18], [15, 19], [16, 20],
            [5, 6], [6, 7], [7, 8], [8, 5],
            [9, 10], [10, 11], [11, 12], [12, 9],
            [13, 14], [14, 15], [15, 16], [16, 13],
            [17, 18], [18, 19], [19, 20], [20, 17],
            [9, 6], [6, 12], [12, 7], [7, 9],
            [14, 9], [9, 15], [15, 12], [12, 14],
            [17, 15], [15, 20], [20, 14], [14, 17]
            ]

numberElem = 3
numberBeam = len(beamList)
for x in range(numberBeam):
    i = beamList[x][0]
    j = beamList[x][1]

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
            beamList.append([current, new])
            nodeList.append([nodeList[current-1][0] + len_x, nodeList[current-1][1] + len_y, nodeList[current-1][2] + len_z])

            current = new
        else:
            beamList.append([new, j])

print(nodeList)
print(beamList)
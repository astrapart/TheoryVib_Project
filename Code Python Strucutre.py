import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Créez une figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

## Structure
# Coordonnées des points
x_s = [0, 0, 5, 5, 0.05, 0.05, 4.95, 4.95, 0.47, 0.47, 4.53, 4.53, 0.89, 0.89, 4.11, 4.11, 1.31, 1.31, 3.69, 3.69]
y_s = [0, 5, 0, 5, 0.05, 4.95, 0.05, 4.95, 0.47, 4.53, 0.47, 4.53, 0.89, 4.11, 0.89, 4.11, 1.31, 3.69, 1.31, 3.69]
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

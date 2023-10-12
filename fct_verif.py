import numpy as np

def est_symetrique(matrix):
    if np.allclose(matrix, matrix.T) :
        return "La matrice est symétrique"
    else :
        return "La matrice n'est pas symétrique"

def est_def_pos(matrix) :
    eigenvalues, _ = np.linalg.eigh(matrix)
    if np.all(eigenvalues > 0):
        return "Toutes les valeurs propres sont strictement positives => définie positive."

    elif np.all(eigenvalues >= 0):
        return "Toutes les valeurs propres sont positives ou nulles => semi-définie positive."

    else:
        return "Il existe un ou des valeurs propres négatives => pas définie positive"


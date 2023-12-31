"""
MAJ 12/10/2023
Fichier contenant les différentes données du problème de vibration

"""
import numpy as np

"""
########################################################################################################################
DATA PROJECT
########################################################################################################################
"""
density_beam = 7800       # [kg/m3]
poisson_ratio = 0.3       # [-]
young_mod = 210e9         # [Pa]
diam_beam = [1, 0.6, 1]   # [m]
thickness_beam = 0.02     # [m]
mass_lumped = 200000      # [kg]
node_lumped_J = 24000000  # [kg*m2]


tan_3 = np.tan(np.radians(3))
nodeList_eol = [[0, 0, 0],  # node 0
                [5, 0, 0],  # node 1
                [0, 5, 0],  # node 2
                [5, 5, 0],  # node 3

                [tan_3, tan_3, 1],          # node 4
                [5 - tan_3, tan_3, 1],      # node 5
                [tan_3, 5 - tan_3, 1],      # node 6
                [5 - tan_3, 5 - tan_3, 1],  # node 7

                [9 * tan_3, 9 * tan_3, 9],          # node 8
                [5 - 9 * tan_3, 9 * tan_3, 9],      # node 9
                [9 * tan_3, 5 - 9 * tan_3, 9],      # node 10
                [5 - 9 * tan_3, 5 - 9 * tan_3, 9],  # node 11

                [17 * tan_3, 17 * tan_3, 17],          # node 12
                [5 - 17 * tan_3, 17 * tan_3, 17],      # node 13
                [17 * tan_3, 5 - 17 * tan_3, 17],      # node 14
                [5 - 17 * tan_3, 5 - 17 * tan_3, 17],  # node 15

                [25 * tan_3, 25 * tan_3, 25],          # node 16
                [5 - 25 * tan_3, 25 * tan_3, 25],      # node 17
                [25 * tan_3, 5 - 25 * tan_3, 25],      # node 18
                [5 - 25 * tan_3, 5 - 25 * tan_3, 25],  # node 19

                [2.5, 2.5, 25],  # node 20
                [2.5, 2.5, 80]]  # node 21

elemList0_eol = [[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0],          # main beam
                 [5, 9, 0], [6, 10, 0], [7, 11, 0], [8, 12, 0],       # main beam
                 [9, 13, 0], [10, 14, 0], [11, 15, 0], [12, 16, 0],   # main beam
                 [13, 17, 0], [14, 18, 0], [15, 19, 0], [16, 20, 0],  # main beam

                 [5, 6, 1], [5, 7, 1], [6, 8, 1], [8, 7, 1],              # secondary beam
                 [9, 10, 1], [9, 11, 1], [10, 12, 1], [12, 11, 1],        # secondary beam
                 [13, 14, 1], [13, 15, 1], [14, 16, 1], [16, 15, 1],      # secondary beam
                 [17, 18, 1], [17, 19, 1], [18, 20, 1], [20, 19, 1],      # secondary beam
                 [9, 6, 1], [6, 12, 1], [12, 7, 1], [7, 9, 1],            # secondary beam
                 [9, 14, 1], [14, 12, 1], [12, 15, 1], [15, 9, 1],        # secondary beam
                 [17, 15, 1], [15, 20, 1], [20, 14, 1], [14, 17, 1],      # secondary beam

                 [17, 21, 2], [18, 21, 2], [19, 21, 2], [20, 21, 2], [21, 22, 2]]  # rigid links

"""
########################################################################################################################
Transient response
########################################################################################################################
"""

dampingRatioInit = [0.5/100, 0.5/100]
gamma = 0.5
beta = 0.25

f = 1                 # [Hz] frequency of the sinus
m = 1000              # [kg] weight of the tail
v = 25 / 3.6          # [m/s] velocity of the impact
timpact = 0.05        # [s] impacts lasts
acc = v / timpact       # [m/s²] impacts acceleration
efficiency = 0.85   # [%] percentage of the tail momentum transferred

A = acc * efficiency * m

ApplNode = 17         # Node where F is applied

"""
########################################################################################################################
DATA EXAMPLE 3D
########################################################################################################################
"""
a = 5.49
b = 3.66
nodeList_example = [[0, 0, 0],    # node 0
                    [a, 0, 0],    # node 1
                    [0, a, 0],    # node 2
                    [a, a, 0],    # node 3
                    [0, 0, b],    # node 4
                    [a, 0, b],    # node 5
                    [0, a, b],    # node 6
                    [a, a, b],    # node 7
                    [0, 0, 2*b],  # node 8
                    [a, 0, 2*b],  # node 9
                    [0, a, 2*b],  # node 10
                    [a, a, 2*b]]  # node 11

elemList0_example = [[1, 5, 0], [2, 6, 0], [3, 7, 0], [4, 8, 0],
                     [5, 9, 0], [6, 10, 0], [7, 11, 0], [8, 12, 0],
                     [5, 6, 1], [6, 8, 1], [8, 7, 1], [7, 5, 1],
                     [9, 10, 1], [10, 12, 1], [12, 11, 1], [11, 9, 1]]

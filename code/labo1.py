import numpy as np
import matplotlib.pyplot as plt
# import helpers as hlp
from helpers import analysis

mat_cov = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 7]])

# les dimenrsions 1 et 2 sont corrélées (même valeur (2))
# peut-t-on enlever la dimenstion 2 ? Non pas encore

valeurs_propres, vecteurs_propres = np.linalg.eig(mat_cov)

print("Valeurs propres:\n", valeurs_propres)
print("Vecteurs propres:\n", vecteurs_propres)

# indice_max = np.argmax(valeurs_propres)
# direction_max = vecteurs_propres[:, indice_max]
# print("Direction de plus grande variance:", direction_max)


projected = analysis.project_onto_new_basis(mat_cov, vecteurs_propres)

print("Projected:\n", projected)

# plot projected data
# plt.scatter(projected[0, :], projected[1, :])
# plt.axis('equal')

# plt.show()

# plot eigenvectors and eigenvalues
# plt.plot([0, vecteurs_propres[0, 0]], [0, vecteurs_propres[1, 0]], 'r')
# plt.plot([0, vecteurs_propres[0, 1]], [0, vecteurs_propres[1, 1]], 'g')
# plt.plot([0, vecteurs_propres[0, 2]], [0, vecteurs_propres[1, 2]], 'b')
# plt.axis('equal')

# plt.show()


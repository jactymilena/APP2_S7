import numpy as np
import os
import helpers.analysis as an

def labo_E2():
    os.system('cls')
    mat_covariance = [[2,1,0],[1,2,0],[0,0,7]]
    val_propres, vecteurs_propres = np.linalg.eig(mat_covariance)
    print("-----------Valeurs propres-----------")
    print(val_propres)
    print("\n")
    print("-----------Vecteurs propres-----------")
    print(vecteurs_propres)
    print("\n")
   

    print("\n")

    rangee = [0,1,2]
    temp_vec = np.zeros(np.asarray(rangee).shape)
    #print(mat_covariance[1])
    for i in rangee:
        temp_vec[i] = vecteurs_propres[0][i]
    print(temp_vec)

    print("-----------NUMERO 5-----------")
    x = an.project_onto_new_basis(mat_covariance, temp_vec)
    print(x)





if __name__ == '__main__':
    labo_E2()
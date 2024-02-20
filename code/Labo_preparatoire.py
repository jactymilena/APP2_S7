import numpy as np

def labo_prep():
    n = 2
    N = 20
    m = [3,-1]
    E = [[1, 0], [0, 1]]

    print("================================== Exercice 1 ==================================")
    for i in range(n):
        print(np.random.multivariate_normal(m,E,N))


if __name__ == '__main__':
    labo_prep()
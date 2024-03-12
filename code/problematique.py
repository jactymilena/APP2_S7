"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt


from helpers.ImageCollection import ImageCollection
import helpers.classifiers as classifiers
from keras.optimizers import Adam
import keras as K


#######################################
def problematique_APP2():
    images = ImageCollection(True)
    data = images.generate_representation()

    if True:
        # 10, 9
        n_neurons = 10
        n_layers = 9
        nn1 = classifiers.NNClassify_APP2(data2train=data, data2test=data,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.0002), loss='categorical_crossentropy',
                                          metrics=['accuracy'],
                                          experiment_title='NN Simple',
                                          n_epochs=1500, savename='3classes',
                                          ndonnees_random=5000, gen_output=True, view=True)

    if True:
        ppv1km1 = classifiers.PPVClassify_APP2(data2train=data, data2test=data, n_neighbors=5,
                                               experiment_title='5-PPV sur le 17-moy',
                                               useKmean=True, n_representants=17,
                                               gen_output=True, view=True)

    if True:
        apriori = [1/3, 1/3, 1/3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
        bg1 = classifiers.BayesClassify_APP2(data2train=data, data2test=data,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)

    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()

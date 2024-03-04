"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

import os

from helpers.ImageCollection import ImageCollection
import helpers.classifiers as classifiers
from keras.optimizers import Adam
import keras as K


#######################################
def problematique_APP2():
    images = ImageCollection(True)
    data = images.generateRepresentation()
    os.system("cls")
    # N = 6
    # im_list_forest = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.coast)
    # images.generateHSVHistograms(im_list_forest)

    if False:
        n_neurons = 10
        n_layers = 7
        nn1 = classifiers.NNClassify_APP2(data2train=data, data2test=data,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
                                          callback_list=[K.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                                          metrics=['accuracy'],
                                          experiment_title='NN Simple',
                                          n_epochs=1500, savename='3classes',
                                          ndonnees_random=5000, gen_output=True, view=True)

    if False:
        ppv1 = classifiers.PPVClassify_APP2(data2train=data, data2test=data, n_neighbors=1,
                                            experiment_title='1-PPV avec données orig comme représentants',
                                            gen_output=True, view=True)

        ppv5 = classifiers.PPVClassify_APP2(data2train=data, data2test=data, n_neighbors=5,
                                            experiment_title='5-PPV avec données orig comme représentants',
                                            gen_output=True, view=True)

    if False:
        apriori = [1/3, 1/3, 1/3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
        bg1 = classifiers.BayesClassify_APP2(data2train=data, data2test=data,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)


    
    N = 979
    im_list = images.get_samples(N, random_samples=True)
    sorted_list = sorted(im_list)
    #print(sorted_list)
    images.get_straight_line(img_list=sorted_list, show_graphs=False, show_hist=True)

    #images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()

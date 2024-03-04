"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import os
from helpers.ImageCollection import ImageCollection


#######################################
def problematique_APP2():
    images = ImageCollection(True)
    data = images.generateRepresentation()

    if False:
        n_neurons = 8
        n_layers = 7
        nn1 = classifiers.NNClassify_APP2(data2train=data, data2test=data,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.0001), loss='mse',
                                          callback_list=[K.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
                                          metrics=['accuracy'],
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='3classes',
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

    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    # if True:
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        # N = 6
        # im_list = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.forest)
        # print(im_list)
        # images.images_display(im_list)
        # images.view_histogrammes(im_list)

    # TODO L1.E4.6 à L1.E4.8
    # images.generateLABHistograms()
    # images.generateRGBHistograms(im_list)
    # if True:
    #     N = 6

        # im_list_coast = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.coast)
        # images.generateHSVHistograms(im_list_coast)

        # im_list_forest = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.forest)
        # images.generateHSVHistograms(im_list_forest)
        #
        # im_list_street = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.street)
        # images.generateHSVHistograms(im_list_street)

    #images.edge_detection()
    os.system("cls")
    N = 979
    im_list = images.get_samples(N, random_samples=True)
    sorted_list = sorted(im_list)
    #print(sorted_list)
    images.get_straight_line(show_graphs=False, img_list=sorted_list, show_hist=True)
    
    #####images.hough_transform_circular_elliptical()

    #images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()

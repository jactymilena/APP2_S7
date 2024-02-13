"""
Départ des laboratoires
Visualisation, Classification, etc. de 3 classes avec toutes les méthodes couvertes par l'APP
APP2 S8 GIA
"""


import matplotlib.pyplot as plt

from helpers.ClassificationData import ClassificationData
import helpers.analysis as an
import helpers.classifiers as classifiers

from keras.optimizers import Adam
import keras as K


##########################################
def labo_APP2():
    data3classes = ClassificationData()
    # Changer le flag dans les sections pertinentes pour chaque partie de laboratoire
    if True:
        # TODO Labo L1.E1.3 et L3.E1
        print('\n\n=========================\nDonnées originales\n')
        # Affiche les stats de base
        data3classes.getStats(gen_print=True)
        # Figure avec les ellipses et les frontières
        data3classes.getBorders(view=True)
        # exemple d'une densité de probabilité arbitraire pour 1 classe
        an.creer_hist2D(data3classes.dataLists[0], 'C1', view=True)

    if True:
        # Décorrélation
        # TODO Labo L1.E3.5
        # data3classesDecorr = ClassificationData(il_manque_la_decorréleation_ici)
        data3classesDecorr = ClassificationData(an.project_onto_new_basis(data3classes.dataLists, data3classes.vectpr[0]))
        print('\n\n=========================\nDonnées décorrélées\n')
        data3classesDecorr.getStats(gen_print=True)
        data3classesDecorr.getBorders(view=True)

    if True: # TODO Labo L2.E4
        # Exemple de RN
        n_neurons = 2
        n_layers = 1
        nn1 = classifiers.NNClassify_APP2(data2train=data3classes, data2test=data3classes,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
                                          outputActivation='softmax', optimizer=Adam(), loss='binary_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[],     # TODO à compléter L2.E4
                                          experiment_title='NN Simple',
                                          n_epochs = 10, savename='3classes',
                                          ndonnees_random=5000, gen_output=True, view=True)

    if True:  # TODO L3.E2
        # Exemples de ppv avec ou sans k-moy
        # 1-PPV avec comme représentants de classes l'ensemble des points déjà classés
        ppv1 = classifiers.PPVClassify_APP2(data2train=data3classes, n_neighbors=1,
                                            experiment_title='1-PPV avec données orig comme représentants',
                                            gen_output=True, view=True)
        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        ppv1km1 = classifiers.PPVClassify_APP2(data2train=data3classes, data2test=data3classes, n_neighbors=1,
                                               experiment_title='1-PPV sur le 1-moy',
                                               useKmean=True, n_representants=1,
                                               gen_output=True, view=True)

    if True:  # TODO L3.E3
        # Exemple de classification bayésienne
        apriori = [1/3, 1/3, 1/3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
        bg1 = classifiers.BayesClassify_APP2(data2train=data3classes, data2test=data3classes,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)

    plt.show()


#####################################
if __name__ == '__main__':
    labo_APP2()
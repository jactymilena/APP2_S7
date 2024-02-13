"""
Fonctions utiles pour le traitement de données
APP2 S8 GIA
Classe :
    Extent: bornes ou plage utile de données

Fonctions :
    calc_erreur_classification: localise les différences entre 2 vecteurs pris comme les étiquettes prédites et
        anticipées, calcule le taux d'erreur et affiche la matrice de confusion

    splitDataNN: sépare des données et des étiquettes en 2 sous-ensembles en s'assurant que chaque classe est représentée

    viewEllipse: ajoute une ellipse à 1 sigma sur un graphique
    view_classes: affiche sur un graphique 2D les points de plusieurs classes
    view_classification_results: affichage générique de résultats de classification
    printModeleGaussien: affiche les stats de base sous forme un peu plus lisible
    plot_metrics: itère et affiche toutes les métriques d'entraînement d'un RN en regroupant 1 métrique entraînement
                + la même métrique de validation sur le même subplot
    creer_hist2D: crée la densité de probabilité d'une série de points 2D
    view3D: génère un graphique 3D de classes

    calcModeleGaussien: calcule les stats de base d'une série de données
    project_onto_new_basis: projette un espace sur une nouvelle base de vecteurs

    genDonneesTest: génère un échantillonnage aléatoire dans une plage 2D spécifiée

    scaleData: borne les min max e.g. des données d'entraînement pour les normaliser
    scaleDataKnownMinMax: normalise des données selon un min max déjà calculé
    descaleData: dénormalise des données selon un min max (utile pour dénormaliser une sortie prédite)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm
import itertools
import math
import random

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as ttsplit


class Extent:
    # TODO Problématique ou JB, generalize to N-D
    """
    classe pour contenir les min et max de données 2D
    membres: xmin, xmax, ymin, ymax
    Constructeur peut utiliser les 4 valeurs précédentes ou
        calculer directement les min et max d'une liste de points
    Accesseurs:
        get_array: retourne les min max formattés en array
        get_corners: retourne les coordonnées des points aux coins d'un range couvert par les min max
    """
    def __init__(self, xmin=0, xmax=10, ymin=0, ymax=10, ptList=None):
        """
        Constructeur
        2 options:
            passer 4 arguments min et max
            passer 1 array qui contient les des points sur lesquels sont calculées les min et max
        """
        if ptList is not None:
            self.xmin = np.floor(np.min(ptList[:,0]))-1
            self.xmax = np.ceil(np.max(ptList[:,0]))+1
            self.ymin = np.floor(np.min(ptList[:,1]))-1
            self.ymax = np.ceil(np.max(ptList[:,1]))+1
        else:
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax

    def get_array(self):
        """
        Accesseur qui retourne sous format matriciel
        """
        return [[self.xmin, self.xmax], [self.ymin, self.ymax]]

    def get_corners(self):
        """
        Accesseur qui retourne une liste points qui correspondent aux 4 coins d'un range 2D bornés par les min max
        """
        return np.array(list(itertools.product([self.xmin, self.xmax], [self.ymin, self.ymax])))


def calc_erreur_classification(original_data, classified_data, gen_output=False):
    """
    Retourne l'index des éléments différents entre deux vecteurs
    Affiche l'erreur moyenne et la matrice de confusion
    """
    # génère le vecteur d'erreurs de classification
    vect_err = np.absolute(original_data - classified_data).astype(bool)
    indexes = np.array(np.where(vect_err))[0]
    if gen_output:
        print(f'\n\n{len(indexes)} erreurs de classification sur {len(original_data)} données (= {len(indexes)/len(original_data)*100} %)')
        print('Confusion:\n')
        print(confusion_matrix(original_data, classified_data))
    return indexes


def calcModeleGaussien(data, message=''):
    """
    Calcule les stats de base de données
    :param data: les données à traiter, devrait contenir 1 point N-D par ligne
    :param message: si présent, génère un affichage des stats calculées
    :return: la moyenne, la matrice de covariance, les valeurs propres et les vecteurs propres de "data"
    """
    # TODO Labo L1.E2.2 Compléter le code avec les fonctions appropriées ici
    moyenne = np.ones(np.asarray(data).shape[-1]) # pas_la_bonne_moyenne
    matr_cov = np.identity(np.asarray(data).shape[-1]) # pas la bonne covariance
    val_propres, vect_propres = [[1, 2], [[1,1],[-1, -1]]] # pas la bonne affaire
    if message:
        printModeleGaussien(moyenne, matr_cov, val_propres, vect_propres, message)
    return moyenne, matr_cov, val_propres, vect_propres


def creer_hist2D(data, title='', nbinx=15, nbiny=15, view=False):
    """
    Crée une densité de probabilité pour une classe 2D au moyen d'un histogramme
    data: liste des points de la classe, 1 point par ligne (dimension 0)

    retourne un array 2D correspondant à l'histogramme et les "frontières" entre les bins
    """

    x = np.array(data[:, 0])
    y = np.array(data[:, 1])

    # TODO L2.E1.1 Faire du pseudocode et implémenter une segmentation en bins...
    # pas des bins de l'histogramme
    deltax = (np.max(x) - np.min(x)) / nbinx
    deltay = (np.max(y) - np.min(y)) / nbiny

    # TODO L3.S2.1: remplacer les valeurs bidons par la bonne logique ici
    hist, xedges, yedges = np.histogram2d([1, 1], [1, 1], bins=[1, 1]) # toutes ces valeurs sont n'importe quoi
    # normalise par la somme (somme de densité de prob = 1)
    histsum = np.sum(hist)
    hist = hist / histsum

    if view:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title(f'Densité de probabilité de {title}')

        # calcule des frontières des bins
        xpos, ypos = np.meshgrid(xedges[:-1] + deltax / 2, yedges[:-1] + deltay / 2, indexing="ij")
        dz = hist.ravel()

        # list of colors
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
        max_height = np.max(dz)  # get range of colorbars so we can normalize
        min_height = np.min(dz)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k - min_height) / max_height) for k in dz]

        ax.bar3d(xpos.ravel(), ypos.ravel(), 0, deltax * .9, deltay * .9, dz, color=rgba)

    return hist, xedges, yedges


def descaleData(x, minmax):
    # usage: OUT = descale_data(IN, MINMAX)
    #
    # Descale an input vector or matrix so that the values
    # are denormalized from the range [-1, 1].
    #
    # Input:
    # - IN, the input vector or matrix.
    # - MINMAX, the original range of IN.
    #
    # Output:
    # - OUT, the descaled input vector or matrix.
    #
    y = ((x + 1.0) / 2) * (minmax[1] - minmax[0]) + minmax[0]
    return y


def genDonneesTest(ndonnees, extent):
    # génération de n données aléatoires 2D sur une plage couverte par extent
    # TODO JB: generalize to N-D
    return np.transpose(np.array([(extent.xmax - extent.xmin) * np.random.random(ndonnees) + extent.xmin,
                                         (extent.ymax - extent.ymin) * np.random.random(ndonnees) + extent.ymin]))


def plot_metrics(NNmodel):
    """
    Helper function pour visualiser des métriques d'entraînement de RN
    :param NNmodel: réseau de neurones entraîné
    """
    assert NNmodel.history is not None
    # Détermine le nombre de subplots nécessaires
    # = combien de métriques uniques on a
    # pour afficher l'entraînement et la validation sur le même subplot
    n_subplots = 0
    for j, current_metric in enumerate(NNmodel.history.history):
        if current_metric.find('val_') != -1:
            continue
        else:
            n_subplots += 1
    [f, axs] = plt.subplots(1, n_subplots)

    # remplit les différents subplots
    currentSubplot = 0
    for _, current_metric in enumerate(NNmodel.history.history):
        # Skip les métriques de validation pour les afficher plus tard
        # sur le même subplot que la même métrique d'entraînement
        if current_metric.find('val_') != -1:
            continue
        else:
            # Workaround pour subplot() qui veut rien savoir de retourner un array 1D quand on lui demande 1x1
            if n_subplots > 1:
                ax = axs[currentSubplot]
            else:
                ax = axs

            ax.plot([x + 1 for x in NNmodel.history.epoch],
                    NNmodel.history.history[current_metric],
                    label=current_metric)
            if NNmodel.history.history.get('val_' + current_metric):
                ax.plot([x + 1 for x in NNmodel.history.epoch],
                        NNmodel.history.history['val_' + current_metric],
                        label='validation ' + current_metric)
            ax.legend()
            ax.grid()
            ax.set_title(current_metric)
            currentSubplot += 1
    f.tight_layout()


def printModeleGaussien(moyenne, matr_cov, val_propres, vect_propres, message=''):
    if message:
        print(message)
    print(f'Moy: {moyenne} \nCov: {matr_cov} \nVal prop: {val_propres} \nVect prop: {vect_propres}\n')


def project_onto_new_basis(data, basis):
    """
    Permet de projeter des données sur une base (pour les décorréler par exemple)
    :param data: classes à décorréler, la dimension 0 est le nombre de classes
    :param basis: les vecteurs (propres) sur lesquels projeter les données; doivent être déjà normalisés
    :return: les données projetées
    """
    dims = np.asarray(data).shape
    assert dims[-1] == len(basis)
    projected = np.zeros(np.asarray(data).shape)
    for i in range(dims[0]):  # dims[0] = n_classes
        tempdata = data[i]
        # TODO Labo L1.E2.5 Remplacer l'opération bidon par la bonne projection ici
        # projected[i] = ??
    return projected


def rescaleHistLab(LabImage, n_bins=256):
    """
    Helper function
    La représentation Lab requiert un rescaling avant d'histogrammer parce que ce sont des floats!
    """
    # Constantes de la représentation Lab
    class LabCte:  # TODO JB : utiliser an.Extent?
        min_L: int = 0
        max_L: int = 100
        min_ab: int = -110
        max_ab: int = 110

    # Création d'une image vide
    imageLabRescale = np.zeros(LabImage.shape)
    # Quantification de L en n_bins niveaux     # TODO JB : utiliser scaleData?
    imageLabRescale[:, :, 0] = np.round(
        (LabImage[:, :, 0] - LabCte.min_L) * (n_bins - 1) / (
                LabCte.max_L - LabCte.min_L))  # L has all values between 0 and 100
    # Quantification de a et b en n_bins niveaux
    imageLabRescale[:, :, 1:3] = np.round(
        (LabImage[:, :, 1:3] - LabCte.min_ab) * (n_bins - 1) / (
                LabCte.max_ab - LabCte.min_ab))  # a and b have all values between -110 and 110
    return imageLabRescale


def scaleData(x):
    # usage: OUT = scale_data(IN, MINMAX)
    #
    # Scale an input vector or matrix so that the values
    # are normalized in the range [-1, 1].
    #
    # Input:
    # - IN, the input vector or matrix.
    #
    # Output:
    # - OUT, the scaled input vector or matrix.
    # - MINMAX, the original range of IN, used later as scaling parameters.
    #
    minmax = (np.min(x), np.max(x))
    y = 2.0 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    return y, minmax


def scaleDataKnownMinMax(x, minmax):
    # todo JB assert dimensions
    y = 2.0 * (x - minmax[0]) / (minmax[1] - minmax[0]) - 1
    return y


def splitDataNN(n_classes, data, labels, train_fraction=0.8):
    # Split into train and validation subsets
    # This is overly complicated because in order to ensure that each class is represented in split sets,
    #   we have to split each class separately
    # The classes are shuffled individually first just in case all similar cases are regrouped in the original class data
    # The classes will be ordered in the resulting list, so we shuffle them even if shuffle=True is used
    # during training, this is more robust if eventually that option ends up not used

    traindataLists = []
    trainlabelsLists = []
    validdataLists = []
    validlabelsLists = []
    for i in range(n_classes):
        # The only datatype easy to shuffle starting with an array_like is a list, but we also have to ensure
        # that data and labels are shuffled together!!
        classData = list(zip(data[i], labels[i]))
        random.shuffle(classData)
        shuffledData, shuffledLabels = zip(*classData)
        shuffledData = list(shuffledData)  # why does zip not return a specifiable datatype directly
        shuffledLabels = list(shuffledLabels)
        # split into subsets
        training_data, validation_data, training_target, validation_target = \
            ttsplit(shuffledData, shuffledLabels, train_size=train_fraction)
        traindataLists.append(training_data)
        trainlabelsLists.append(training_target)
        validdataLists.append(validation_data)
        validlabelsLists.append(validation_target)

    # switch back to array_like
    traindataLists = np.array(traindataLists)
    trainlabelsLists = np.array(trainlabelsLists)
    validdataLists = np.array(validdataLists)
    validlabelsLists = np.array(validlabelsLists)

    # Merge all class splits into 1 contiguous array
    nclasses, nsamples, dimensions = traindataLists.shape
    traindata1array = traindataLists.reshape(nclasses * nsamples, dimensions)
    nclasses, nsamples, dimensions = trainlabelsLists.shape
    trainlabels1array = trainlabelsLists.reshape(nclasses * nsamples, dimensions)

    # Reshuffle the completed array just for good measure
    trainData = list(zip(traindata1array, trainlabels1array))
    random.shuffle(trainData)
    shuffledTrainData, shuffledTrainLabels = zip(*trainData)
    shuffledTrainData = np.array(list(shuffledTrainData))
    shuffledTrainLabels = np.array(list(shuffledTrainLabels))

    # Do the same for the other split
    nclasses, nsamples, dimensions = validdataLists.shape
    validdata1array = validdataLists.reshape(nclasses * nsamples, dimensions)
    nclasses, nsamples, dimensions = validlabelsLists.shape
    validlabels1array = validlabelsLists.reshape(nclasses * nsamples, dimensions)

    validData = list(zip(validdata1array, validlabels1array))
    random.shuffle(validData)
    shuffledValidData, shuffledValidLabels = zip(*validData)
    shuffledValidData = np.array(list(shuffledValidData))
    shuffledValidLabels = np.array(list(shuffledValidLabels))

    return shuffledTrainData, shuffledTrainLabels, shuffledValidData, shuffledValidLabels


def view3D(data3D, targets, title):
    """
    Génère un graphique 3D de classes
    :param data: tableau, les 3 colonnes sont les données x, y, z
    :param target: sert à distinguer les classes, expect un encodage one-hot
    """
    colors = np.array([[1.0, 0.0, 0.0],  # Red
                       [0.0, 1.0, 0.0],  # Green
                       [0.0, 0.0, 1.0]])  # Blue
    c = colors[targets]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data3D[:, 0], data3D[:, 1], data3D[:, 2], s=10.0, c=c, marker='x')
    ax.set_title(title)
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    ax.set_zlabel('Third component')
    fig.tight_layout()


def view_classes(data, extent, border_coeffs=None):
    """
    Affichage des classes dans data
    *** Fonctionne pour des classes 2D

    data: tableau des classes à afficher. La première dimension devrait être égale au nombre de classes.
    extent: bornes du graphique
    border_coeffs: coefficient des frontières, format des données voir helpers.classifiers.get_borders()
        coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
    """
    #  TODO JB: rendre général, seulement 2D pour l'instant
    dims = np.asarray(data).shape

    fig1, ax1 = plt.subplots(1, 1)
    ax1.set_title(r'Visualisation des classes, des ellipses à distance 1$\sigma$' +
                  (' et des frontières' if border_coeffs is not None else ''))

    #  TODO JB: rendre général, seulement 3 classes pour l'instant
    colorpoints = ['orange', 'purple', 'black']
    colorfeatures = ['red', 'green', 'blue']

    for i in range(dims[0]):
        tempdata = data[i]
        m, cov, valpr, vectprop = calcModeleGaussien(tempdata)
        ax1.scatter(tempdata[:, 0], tempdata[:, 1], s=5, c=colorpoints[i])
        ax1.scatter(m[0], m[1], c=colorfeatures[i])
        viewEllipse(tempdata, ax1, edgecolor=colorfeatures[i])

    # Ajout des frontières
    if border_coeffs is not None:
        x, y = np.meshgrid(np.linspace(extent.xmin, extent.xmax, 400),
                           np.linspace(extent.ymin, extent.ymax, 400))
        for i in range(math.comb(dims[0], 2)):
            # rappel: coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
            ax1.contour(x, y,
                        border_coeffs[i][0] * x ** 2 + border_coeffs[i][2] * y ** 2 +
                        border_coeffs[i][3] * x + border_coeffs[i][6] +
                        border_coeffs[i][1] * x * y + border_coeffs[i][4] * y, [border_coeffs[i][5]])

    ax1.set_xlim([extent.xmin, extent.xmax])
    ax1.set_ylim([extent.ymin, extent.ymax])

    ax1.axes.set_aspect('equal')


def view_classification_results(experiment_title, extent, original_data, colors_original, title_original,
                                test1data, colors_test1, title_test1, test1errors=None, test2data=None,
                                test2errors=None, colors_test2=None, title_test2=''):
    """
    Génère 1 graphique avec 3 subplots:
        1. Des données "d'origine" train_data avec leur étiquette encodée dans la couleur c1
        2. Un aperçu de frontière de décision au moyen d'un vecteur de données aléatoires test1 avec leur étiquette
            encodée dans la couleur c2
        3. D'autres données classées test2 (opt) avec affichage encodée dans la couleur c3
    :param original_data:
    :param test1data:
    :param test2data:
        données à afficher
    :param colors_original:
    :param colors_test1:
    :param colors_test2:
        couleurs
        c1, c2 et c3 sont traités comme des index dans un colormap
    :param experiment_title:
    :param title_original:
    :param title_test1:
    :param title_test2:
        titres de la figure et des subplots
    :param extent:
        range des données
    :return:
    """
    cmap = cm.get_cmap('seismic')
    if np.asarray(test2data).any():
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        if np.asarray(test2errors).any():
            colors_test2[test2errors] = error_class
        ax3.scatter(test2data[:, 0], test2data[:, 1], s=5, c=cmap(colors_test2))
        ax3.set_title(title_test2)
        ax3.set_xlim([extent.xmin, extent.xmax])
        ax3.set_ylim([extent.ymin, extent.ymax])
        ax3.axes.set_aspect('equal')
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(experiment_title)
    ax1.scatter(original_data[:, 0], original_data[:, 1], s=5, c=colors_original, cmap='viridis')
    if np.asarray(test1errors).any():
        colors_test1[test1errors] = error_class
    ax2.scatter(test1data[:, 0], test1data[:, 1], s=5, c=colors_test1, cmap='viridis')
    ax1.set_title(title_original)
    ax2.set_title(title_test1)
    ax1.set_xlim([extent.xmin, extent.xmax])
    ax1.set_ylim([extent.ymin, extent.ymax])
    ax2.set_xlim([extent.xmin, extent.xmax])
    ax2.set_ylim([extent.ymin, extent.ymax])
    ax1.axes.set_aspect('equal')
    ax2.axes.set_aspect('equal')


def viewEllipse(data, ax, scale=1, facecolor='none', edgecolor='red', **kwargs):
    """
    ***Testé seulement sur les données du labo
    Ajoute une ellipse à distance 1 sigma du centre d'une classe
    Inspiration de la documentation de matplotlib 'Plot a confidence ellipse'

    data: données de la classe, les lignes sont des données 2D
    ax: axe des figures matplotlib où ajouter l'ellipse
    scale: Facteur d'échelle de l'ellipse, peut être utilisé comme paramètre pour tracer des ellipses à une
        équiprobabilité différente, 1 = 1 sigma
    facecolor, edgecolor, and kwargs: Arguments pour la fonction plot de matplotlib

    retourne l'objet Ellipse créé
    """
    moy, cov, lambdas, vectors = calcModeleGaussien(data)
    # TODO L3.E1.1 Remplacer les valeurs bidons par les bons paramètres à partir des stats ici
    # tous les 1 sont suspects
    ellipse = Ellipse((1, 1), width=2 * np.sqrt(1) * scale, height=2 * np.sqrt(1) * scale,
                      angle=-np.degrees(1), facecolor=facecolor,
                      edgecolor=edgecolor, linewidth=2, **kwargs)
    return ax.add_patch(ellipse)


error_class = 6
# numéro de classe arbitraire à assigner aux points en erreur pour l'affichage, permet de les mettre d'une autre couleur

"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

from helpers.ImageCollection import ImageCollection


#######################################
def problematique_APP2():
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
    if False:  
        N = 6

        im_list_forest = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.coast)
        images.generateHSVHistograms(im_list_forest)

        im_list_coast = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.forest)
        images.generateHSVHistograms(im_list_coast)

        im_list_street = images.get_samples(N, random_samples=True, labels=ImageCollection.imageLabels.street)
        images.generateHSVHistograms(im_list_street)

    # images.edge_detection()
    # images.hough_transform_straight_line()

    images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()

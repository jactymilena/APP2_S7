"""
Classe "ImageCollection" pour charger et visualiser les images de la problématique
Membres :
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, changer le flag load_all du constructeur à True)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes pour la problématique :
    generateRGBHistograms : calcul l'histogramme RGB de chaque image, à compléter
    generateRepresentation : vide, à compléter pour la problématique
Méthodes génériques : TODO JB move to helpers
    generateHistogram : histogramme une image à 3 canaux de couleurs arbitraires
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from enum import IntEnum, auto

from skimage import color as skic
from skimage import io as skiio
from skimage import img_as_ubyte

from skimage.filters import sobel
# Import for Hough
from skimage.transform import probabilistic_hough_line, hough_ellipse
from skimage.feature import canny
from skimage.draw import ellipse_perimeter

import helpers.classifiers as classifiers

from keras.optimizers import Adam
import keras as K

import helpers.analysis as an
from helpers.ClassificationData import ClassificationData


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """
    class imageLabels(IntEnum):
        coast = auto()
        forest = auto()
        street = auto()


    def __init__(self, load_all=False):
        # liste de toutes les images
        self.image_folder = r"data" + os.sep + "baseDeDonneesImages"
        self._path = glob.glob(self.image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in image_list if '.jpg' in i]

        self.all_images_loaded = False
        self.images = []

        # Crée un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        #            [Nombre image, hauteur, largeur, RGB]
        if load_all:
            # self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            # self.all_images_loaded = True
            self.load_all_images()

        self.labels = []
        for i in image_list:
            if 'coast' in i:
                # print(f"coast {j}")
                self.labels.append(ImageCollection.imageLabels.coast)
            elif 'forest' in i:
                # print(f"forest {j}")
                self.labels.append(ImageCollection.imageLabels.forest)
            elif 'street' in i:
                # print(f"street {j}")
                self.labels.append(ImageCollection.imageLabels.street)
            else:
                raise ValueError(i)


    def load_all_images(self):
        """
        Charge toutes les images dans la liste
        """
        self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
        self.all_images_loaded = True

    
    def get_images(self, idx0, idx1):
        """
        Charge images dans la liste de idx0 à idx1
        """
        return np.array([np.array(skiio.imread(image)) for image in self._path[idx0:idx1]])

    
    def load_images(self, N):
        coast_images = self.get_images(0, 0 + N)
        forest_images = self.get_images(360, 360 + N)
        street_images = self.get_images(688, 688 + N)

        # for images in [coast_images, forest_images, street_images]:
        #     self.images.append(images)

        self.images = np.concatenate((coast_images, forest_images, street_images))


    def get_samples(self, N, random_samples=False, labels=None):
        idx = 0
        idx2 = 979
        if labels is not None:
            if labels == ImageCollection.imageLabels.coast:
                idx = 0
                idx2 = 359
            if labels == ImageCollection.imageLabels.forest:
                idx = 360
                idx2 = 687
            elif labels == ImageCollection.imageLabels.street:
                idx = 688
                idx2 = 979

        if random_samples:
            return random.sample([n for n in range(idx, idx2)], N)
        
        return [n for n in range(idx, idx + N)]


    def generateHistogram(self, image, n_bins=256):
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = 3
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            for j in range(n_channels):
                pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)
        return pixel_values
    

    def get_color_quantity(self, image, color_index):
        """
        Retourne la quantité de couleur pour chaque canal
        """
        green_col = image[:, :, color_index]
        g = 0
        for j in range(256):
            for k in range(256):
                g += green_col[j, k]
        return g


    def generateHSVHistograms(self, im_list):
        """
        Calcule les histogrammes HSV de toutes les images
        """
        n_bins = 256

        fig = plt.figure()
        ax = fig.subplots(len(im_list), 2)

        # for i in range(len(self.image_list)):
        for j, i in enumerate(im_list):
            imageRGB = self.get_RGB_from_indx(i)
            imageHSV = skic.rgb2hsv(imageRGB)
            imageHSV = np.round(imageHSV * (n_bins - 1))

            histvaluesHSV = self.generateHistogram(imageHSV)

            # plot imae
            ax[j, 0].imshow(imageRGB)
            # scatter hue values
            ax[j, 1].scatter(range(n_bins), histvaluesHSV[0], s=3, c='magenta')

            m1 = self.getMeanMaxValues(imageHSV)

            print(f"Mean Max Values: {m1} for image {i} : {self.image_list[i]}")

        fig.show()


    def plot_histogram(self, ax, title, qty_array, plt_index, color):
        """
        Affiche l'histogramme
        """
        ax[plt_index].scatter(range(len(qty_array)), qty_array, s=3, c=color)
        ax[plt_index].set(xlabel='# image', ylabel='intensity')
        ax[plt_index].set_title(title)
    

    def get_variance(self, image, index):
        """
        Retourne la variance de l'image
        """
        return np.var(image[:, :, index])


    def generateRGBHistograms(self,im_list):
        """
        Calcule les histogrammes RGB de toutes les images
        """
        # TODO L1.E4.6 S'inspirer de view_histogrammes et déménager le code pertinent ici
        red_qty = []
        green_qty = []
        blue_qty = []
        # for i in im_list:
        for i in range(len(self.image_list)):
            # charge une image si nécessaire
            imageRGB = self.get_RGB_from_indx(i)
            
            print(f"Started Image {i} : {self.image_list[i]}") 
            # red_qty.append(self.get_color_quantity(imageRGB, 0))
            # green_qty.append(self.get_color_quantity(imageRGB, 1))
            # blue_qty.append(self.get_color_quantity(imageRGB, 2))

            red_qty.append(self.get_variance(imageRGB, 0))
            green_qty.append(self.get_variance(imageRGB, 1))
            blue_qty.append(self.get_variance(imageRGB, 2))

            # print(f"Variance Red: {self.get_variance(imageRGB, 0)}")
            # print(f"Variance Green: {self.get_variance(imageRGB, 1)}")
            # print(f"Variance Blue: {self.get_variance(imageRGB, 2)}")

            print(f"Finished Image {i} : {self.image_list[i]}") 

        fig = plt.figure()
        ax = fig.subplots(3, 3)

        # Red Histogram
        self.plot_histogram(ax, 'COAST - RED', red_qty[0:359], (0, 0), 'red')
        self.plot_histogram(ax, 'FOREST - RED', red_qty[360:687], (1, 0), 'red')
        self.plot_histogram(ax, 'STREET - RED', red_qty[688:979], (2, 0), 'red')

        # Green Histogram
        self.plot_histogram(ax, 'COAST - GREEN', green_qty[0:359], (0, 1), 'green')
        self.plot_histogram(ax, 'FOREST - GREEN', green_qty[360:687], (1, 1), 'green')
        self.plot_histogram(ax, 'STREET - GREEN', green_qty[688:979], (2, 1), 'green')

        # Blue Histogram
        self.plot_histogram(ax, 'COAST - BLUE', blue_qty[0:359], (0, 2), 'blue')
        self.plot_histogram(ax, 'FOREST - BLUE', blue_qty[360:687], (1, 2), 'blue')
        self.plot_histogram(ax, 'STREET - BLUE', blue_qty[688:979], (2, 2), 'blue')

        fig.show()
    

    def getHuePeak(self, imageHSV):
        """
        Retourne le pic de du Hue
        """
        histvaluesHSV = self.generateHistogram(imageHSV)

        return histvaluesHSV[0].max()


    def getMeanMaxValues(self, imageHSV):
        """
        Retourne la moyenne des pics du Hue
        """
        imageHue = imageHSV[0]
        # sort by y value
        imageHue = np.sort(imageHue, axis=1)
        hueVertical = imageHue[1]

        return np.mean(hueVertical[-10:])


    def getHSVData(self):
        """
        Retourne les données HSV
        """
        n_bins = 256

        self.load_images(6)

        data_coast = []
        data_forest = []
        data_street = []

        for img in self.images[0:6]:
            imgHSV = skic.rgb2hsv(img)
            data_coast.append([0, self.getMeanMaxValues(imgHSV)])

        for img in self.images[6:12]:
            imgHSV = skic.rgb2hsv(img)
            data_forest.append([0, self.getMeanMaxValues(imgHSV)])

        for img in self.images[12:18]:
            imgHSV = skic.rgb2hsv(img)
            imgHSV = np.round(imgHSV * (n_bins - 1))
            data_street.append([0 ,self.getMeanMaxValues(imgHSV)])

        data = [data_coast, data_forest, data_street]

        # print(f"Data Coast shape: {data_coast.shape}")

        # for img in self.images:
        #     imgHSV = skic.rgb2hsv(img)
        #     imgHSV = np.round(imgHSV * (n_bins - 1))
        #     hue_peak = self.getHuePeak(imgHSV)
        #     data.append(hue_peak)

        return np.array(data)


    def generateRepresentation(self):
        # produce a ClassificationData object usable by the classifiers
        # TODO L1.E4.8: commencer l'analyse de la représentation choisie
        # hsv_data = self.getHSVData()
        # print(hsv_data.shape)

        hsv_data = self.getHSVData()
        data = ClassificationData(hsv_data)

        data.getStats(gen_print=True)
        # data.getBorders(view=True)

        # d1 = hsv_data[0:2]
        # d2 = hsv_data[3:5]
        # d3 = hsv_data[6:8]
        #
        # data_train = ClassificationData(np.concatenate(d1, d2, d3))

        # n_neurons = 20
        # n_layers = 10
        #
        # nn1 = classifiers.NNClassify_APP2(data2train=data, data2test=data_train,
        #                                   n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
        #                                   outputActivation='softmax', optimizer=Adam(), loss='binary_crossentropy',
        #                                   metrics=['accuracy'],
        #                                   callback_list=[K.callbacks.EarlyStopping(monitor='val_loss', patience=10)],     # TODO à compléter L2.E4
        #                                   experiment_title='NN Simple',
        #                                   n_epochs=1000, savename='3classes',
        #                                   ndonnees_random=5000, gen_output=True, view=True)


    def images_display(self, indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[indexes[i]]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax2[i].imshow(im)


    def edge_detection(self):
        # images = [ 'coast_cdmc838.jpg', 'coast_natu804.jpg', 'coast_sun34.jpg' ]
        # images = ['coast_cdmc838.jpg' ,'coast_natu804.jpg', 'coast_sun34.jpg', 'forest_bost190.jpg', 'forest_for82.jpg', 'forest_land81.jpg', 'forest_land765.jpg', 'forest_land107.jpg', 'forest_nat717.jpg' , 'street_a232022.jpg', 'street_bost77.jpg', 'street_city42.jpg', 'street_par21.jpg', 'street_urb562.jpg', 'street_urban997.jpg']
        images = ['coast_art487.jpg','coast_bea9.jpg','coast_cdmc891.jpg','coast_land253.jpg','coast_land261.jpg','coast_n199065.jpg','coast_n708024.jpg','coast_nat167.jpg']

        for img_name in images:
            img = skiio.imread(self.image_folder + os.sep + img_name)

            # Turn image to grayscale.
            gray_img = skic.rgb2gray(img)
            sobel_filtered_img = sobel(gray_img)

            fig, ax = plt.subplots(ncols=2,nrows=1, figsize=(10,8), sharex=True, sharey=True)
            ax = ax.ravel()
            fig.tight_layout()
            #Plot the original image
            ax[0].imshow(gray_img, cmap=plt.cm.gray)
            ax[0].set_title('Image en noir et blanc')
            #Plot the Sobel filter applied image
            ax[1].imshow(sobel_filtered_img, cmap=plt.cm.gray)
            ax[1].set_title('Image avec filtre de Sobel')
            for a in ax:
                a.axis('off')

    def hough_transform_straight_line(self, gray_img, ax=None):
        """
        gray_img : grayscale image
        ax : optional - if visuals representation are desired
        """
        # Edge filter an image using the Canny algorithm.
        edges = canny(gray_img, sigma=0.75, low_threshold=0.1, high_threshold=0.3)
        # Return lines from a progressive probabilistic line Hough transform.
        lines = probabilistic_hough_line(edges, threshold=5, line_length=35, line_gap=5)
        # Show pictures if specified earlier
        if type(ax) != type(None):
            ax[1].imshow(edges, cmap=plt.cm.gray)
            ax[1].set_title('Canny edges')
            # Plot des lignes détectées par la transformée de Hough probabiliste.
            ax[2].imshow(edges * 0)
            for line in lines:
                p0, p1 = line
                #print(f"Ligne p0 = {p0} and p1 = {p1}")
                ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), 'r')  # 'r' pour rouge
            ax[2].set_xlim((0, gray_img.shape[1]))
            ax[2].set_ylim((gray_img.shape[0], 0))
            ax[2].set_title('Transformée de Hough probabiliste')
            for a in ax:
                a.set_axis_off()
            plt.tight_layout()
        return lines
        
    def get_straight_line(self, img_list=None, show_graphs=False, show_hist=False):
        my_images = []
        counter = 0
        if img_list == None:
            default_images = ['coast_art487.jpg','coast_bea9.jpg','coast_cdmc891.jpg','coast_land253.jpg','coast_land261.jpg','coast_n199065.jpg','coast_n708024.jpg','coast_nat167.jpg']
            for img_name in default_images:
                print(f"{img_name}")
                my_images.append(skiio.imread(self.image_folder + os.sep + img_name))
                counter = counter + 1
        else:
            for i in img_list:
                my_images.append(self.get_RGB_from_indx(i))
                counter = counter + 1

        counted_lines = []
        for i in range(counter):
            # Turn image to grayscale.
            gray_img = skic.rgb2gray(my_images[i])
            if show_graphs == True:
                # Generating figure
                fig2, ax = plt.subplots(ncols=3,nrows=1, figsize=(10,5), sharex=True, sharey=True)
                ax = ax.ravel()
                fig2.tight_layout()
                # Plot the original image
                ax[0].imshow(gray_img, cmap=plt.cm.gray)
                ax[0].set_title(f'{img_name} en noir et blanc')
            else:
                ax=None
            raw_lines = self.hough_transform_straight_line(gray_img, ax)
            counted_lines.append(self.categorize_hough_lines(raw_lines))

        if show_hist:
            self.show_cat_lines_hist(counter, counted_lines)
            
        
    def show_cat_lines_hist(self, counter, counted_lines):
        # Extract the first value (horizontal lines count) from each row in counted_lines
        horizontal_counts = [row[0] for row in counted_lines]
        vertical_counts = [row[1] for row in counted_lines]
        other_counts = [row[2] for row in counted_lines]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
        axes[0].hist(range(len(horizontal_counts)), weights=horizontal_counts, bins=counter, color='blue', alpha=0.7)
        axes[0].set_title('Lignes Horizontales')
        axes[0].set_xlabel('Index de l\'image')
        axes[0].set_ylabel('Fréquence')
        
        axes[1].hist(range(len(vertical_counts)), weights=vertical_counts, bins=counter, color='green', alpha=0.7)
        axes[1].set_title('Lignes Verticales')
        axes[1].set_xlabel('Index de l\'image')
        axes[1].set_ylabel('Fréquence')
        
        axes[2].hist(range(len(other_counts)), weights=other_counts, bins=counter, color='red', alpha=0.7)
        axes[2].set_title('Autres Lignes')
        axes[2].set_xlabel('Index de l\'image')
        axes[2].set_ylabel('Fréquence')

        title = 'Histogramme des types de lignes dans les images'
        plt.suptitle(title)
        plt.tight_layout()



    def categorize_hough_lines(self, lines):
        """
        Returns the number of horizontal lines, vertical lines and other lines
        Return format [No Horz, No Vert, No Other]
        """
        cat_lines = [0,0,0]
        tolerance = 3

        for line in lines:
            p0, p1 = line
            if abs(p0[1] - p1[1]) <= tolerance:
            #p0[0] == p1[0]:
                cat_lines[0] = cat_lines[0] + 1
            elif abs(p0[0] - p1[0]) <= tolerance:
                cat_lines[1] = cat_lines[1] + 1
            else:
                cat_lines[2] = cat_lines[2] + 1
        
        return cat_lines

    def hough_transform_circular_elliptical(self):
        """
        This function should NOT be used and is there only for possible future fixes.
        """
        #images = ['coast_art487.jpg','coast_bea9.jpg','coast_cdmc891.jpg','coast_land253.jpg','coast_land261.jpg','coast_n199065.jpg','coast_n708024.jpg','coast_nat167.jpg']
        images = ['coast_art487.jpg']
        print("A")
        for img_name in images:
            print("B")
            img = skiio.imread(self.image_folder + os.sep + img_name)

            # Turn image to grayscale.
            gray_img = skic.rgb2gray(img)
            # Edge filter an image using the Canny algorithm.
            edges = canny(gray_img, sigma=0.75)
            #edges = canny(gray_img, sigma=1, low_threshold=0.1, high_threshold=0.3)
            print("C")
            # Perform a Hough Transform
            # The accuracy corresponds to the bin size of a major axis.
            # The value is chosen in order to get a single high accumulator.
            # The threshold eliminates low accumulators
            result = hough_ellipse(edges, min_size=4, max_size=50)
            result.sort(order='accumulator')

            # Estimated parameters for the ellipse
            best = list(result[-1])
            yc, xc, a, b = (int(round(x)) for x in best[1:5])
            orientation = best[5]

            print("D")
            # Draw the ellipse on the original image
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            img[cy, cx] = (0, 0, 255)
            # Draw the edge (white) and the resulting ellipse (red)
            edges = skic.gray2rgb(img_as_ubyte(edges))
            edges[cy, cx] = (250, 0, 0)
            
            fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)
            ax1.set_title('Original picture')
            ax1.imshow(img)
            
            ax2.set_title('Edge (white) and result (red)')
            ax2.imshow(edges)
            print("E")

    def view_histogrammes(self, indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for image_counter in range(len(indexes)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[indexes[image_counter]]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')
            ax[image_counter, 0].legend(['R', 'G', 'B'])

            #Test de vue en 3D
            an.view3D(imageRGB, [1], f'RGB de {image_name}', ["Red", "Green", "Blue"])
            
            

            # 2e histogramme
            # TODO L1.E4 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
            
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c='Gray')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c='green', alpha=0.7)
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c='blue', alpha=0.7)
            ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 1].set_title(f'histogramme LAB de {image_name}')
            ax[image_counter, 1].legend(['l (Lightness)', 'a (g-r)', 'b (b-y)'])
        
            # Ici on affiche en HSV
            # En HSV, on a des valeurs de Hue, Saturation et Value (brightness of a color).
            # https://programmingdesignsystems.com/color/color-models-and-color-spaces/index.html
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='Brown')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='Purple')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c='Yellow')
            #ax[image_counter, 1].scatter(range(start, end), histvaluesHSV[3, start:end], s=3, c='Black')
            ax[image_counter, 2].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 2].set_title(f'histogramme HSV de {image_name}')
            ax[image_counter, 2].legend(['Hue', 'Sat', 'Bright'])

             
    def get_RGB_from_indx(self, indx):
        if self.all_images_loaded:
            imageRGB = self.image_list[indx]
        else:
            imageRGB = skiio.imread(self.image_folder + os.sep + self.image_list[indx])
        return imageRGB
# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

"""
Standalone example
Classificateur de fleurs basé sur des caractéristiques mesurées (représentation mesurée)
S8 GIA APP2
TODO voir L2.E3
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as ttsplit

import helpers.analysis as an
import helpers.classifiers as classifiers


def main():
    # Load iris data set from file
    # Attributes are: petal length, petal width, sepal length, sepal width
    S = scipy.io.loadmat('data'+os.sep+'iris.mat')
    data = np.array(S['data'], dtype=np.float32)
    target = np.array(S['target'], dtype=np.float32)

    # TODO: Analyze the input data
    target_decode = np.argmax(target, axis=-1) # targets are 1hot encoded
    # sépare les classes pour en afficher les propriétés
    C1 = data[np.where(target_decode == 0)]
    C2 = data[np.where(target_decode == 1)]
    C3 = data[np.where(target_decode == 2)]
    an.calcModeleGaussien(C1, '\nClasse versicolor')
    an.calcModeleGaussien(C2, '\nClasse virginica')
    an.calcModeleGaussien(C3, '\nClasse setose')

    # Show the 3D projection of the data
    # TODO L2.E3.1 Observez si différentes combinaisons de dimensions sont discriminantes
    data3D = data[:, 1:4]
    an.view3D(data3D, target_decode, 'dims 1 2 3')

    # TODO Problématique Ici on prend un raccourci avec PCA, mais dans la problématique on demande d'utiliser
    # les techniques vues au labo1
    pca3 = PCA(n_components=3)
    pca3.fit(data)
    data3D = pca3.transform(data)
    C1p3 = pca3.transform(C1)
    C2p3 = pca3.transform(C2)
    C3p3 = pca3.transform(C3)
    an.view3D(data3D, target_decode, 'IRIS dataset (3D projection)')
    an.calcModeleGaussien(data3D, '\nPCA 3d')
    an.calcModeleGaussien(C1p3, '\nC1p 3d')
    an.calcModeleGaussien(C2p3, '\nC2p 3d')
    an.calcModeleGaussien(C3p3, '\nC3p 3d')

    pca2 = PCA(n_components=2)
    pca2.fit(data)
    data2D = pca3.transform(data)
    C1p2 = pca2.transform(C1)
    C2p2 = pca2.transform(C2)
    C3p2 = pca2.transform(C3)
    an.view_classes([C1p2, C2p2, C3p2], an.Extent(ptList=data2D))
    an.calcModeleGaussien(data2D, '\nPCA')
    an.calcModeleGaussien(C1p2, '\nC1p 2D')
    an.calcModeleGaussien(C2p2, '\nC2p 2D')
    an.calcModeleGaussien(C3p2, '\nC3p 2D')

    # TODO : Apply any relevant transformation to the data
    # TODO L2.E3.1 Conservez les dimensions qui vous semblent appropriées et décorrélées-les au besoin
    # (e.g. filtering, normalization, dimensionality reduction)
    data, minmax = an.scaleData(data)

    # TODO L2.E3.4
    training_data = data
    validation_data = []
    training_target = target
    validation_target = []

    # Create neural network
    # TODO L2.E3.3  Tune the number and size of hidden layers
    model = Sequential()
    model.add(Dense(units=3, activation='linear',
                    input_shape=(data.shape[-1],)))
    model.add(Dense(units=target.shape[-1], activation='linear'))
    print(model.summary())

    # Define training parameters
    # TODO L2.E3.3 Tune the training parameters
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.01), loss='mse')

    # Perform training
    callback_list = []  # TODO Labo: callbacks
    # TODO L2.E3.3  Tune the training hyperparameters
    model.fit(training_data, training_target, batch_size=len(data), verbose=0,
              epochs=10, shuffle=True, callbacks=callback_list)  # TODO Labo: ajouter les arguments pour le validation set

    # Save trained model to disk
    model.save('saves'+os.sep+'iris.keras')

    an.plot_metrics(model)

    # Test model (loading from disk)
    model = load_model('saves'+os.sep+'iris.keras')
    targetPred = model.predict(data)

    # Print the number of classification errors from the training data
    error_indexes = an.calc_erreur_classification(np.argmax(targetPred, axis=-1), target_decode, gen_output=True)

    plt.show()

if __name__ == "__main__":
    main()

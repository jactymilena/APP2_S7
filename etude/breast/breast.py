# Copyright (c) 2019, Simon Brodeur
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
# Université de Sherbrooke, APP3 S8GIA, A2019

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
from mpl_toolkits.mplot3d import Axes3D

# ajoute par moi
from sklearn.model_selection import train_test_split as ttsplit

###############################################
# Define helper functions here
###############################################

def scaleData(x, minmax=None):
    if minmax is None:
        minmax = (np.min(x), np.max(x))
    y = 2.0 * (x - np.min(x))/(np.max(x) - np.min(x)) - 1

    return y, minmax


###############################################
# Define code logic here
###############################################

def main():

    # Load breast cancer data set from file
    # Attributes:
    # mean_radius: mean of distances from center to points on the perimeter
    # mean_area: mean area of the core tumor
    # mean_texture: standard deviation of gray-scale values
    # mean_perimeter: mean size of the core tumor
    # mean_smoothness: mean of local variation in radius lengths

    # TODO: Analyze the input data
    # Input attributes: mean_texture, mean_area, mean_smoothness
    S = np.genfromtxt('Breast_cancer_data.csv', delimiter=',', skip_header=1)
    data = np.array(S[:, [2, 3, 4]], dtype=np.float32)

    # Output:
    # The diagnosis of breast tissues (benign, malignant) where malignant denotes that the disease is harmful
    target = np.eye(2, 2)[np.array(S[:, -1], dtype=np.int64)]

    # Show the data
    colors = np.array([[1.0, 0.0, 0.0],   # Red
                       [0.0, 0.0, 1.0]])  # Blue
    c = colors[np.argmax(target, axis=-1)]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10.0, c=c, marker='x')
    ax.set_title('Breast cancer dataset')
    ax.set_xlabel('mean texture')
    ax.set_ylabel('mean area')
    ax.set_zlabel('mean smoothness')
    plt.show()

    # TODO: Create training and validation datasets
    ratio = 0.75    # 25% of the data goes for validation
    idx = np.arange(len(data))
    train_idx = idx[:int(len(idx) * ratio)]
    test_idx = idx[len(train_idx):]

    train_data = data[train_idx]
    train_target = target[train_idx]
    test_data = data[test_idx]
    test_target = target[test_idx]

    # train_data, test_data, train_target, test_target = ttsplit(data, target, test_size=0.25)

    # TODO : Apply any relevant transformation to the data
    # (e.g. filtering, normalization, dimensionality reduction)
    _, minmax = scaleData(data)
    train_data, minmax = scaleData(train_data, minmax)
    test_data, minmax = scaleData(test_data, minmax)

    # Create neural network
    # TODO : Tune the number and size of hidden layers
    if True:
        model = Sequential()
        model.add(Dense(units=16, activation='tanh',
                        input_shape=(data.shape[-1],)))
        model.add(Dense(units=target.shape[-1], activation='linear'))
        print(model.summary())

    
        # Define training parameters
        # TODO : Tune the training parameters
        model.compile(optimizer=SGD(lr=0.1, momentum=0.9),
                    loss='mse')

        # Perform training
        # TODO : Tune the maximum number of iterations and desired error
        model.fit(train_data, train_target, batch_size=len(data),
                epochs=100, shuffle=True, verbose=1)
        
        model.save('model.h5')
        model = load_model('model.h5')

        # Print the number of classification errors from the training data
        targetPred = model.predict(train_data)
        nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(train_target, axis=-1))
        accuracy = (len(train_data) - nbErrors) / len(train_data)
        print('Classification accuracy (training set): %0.3f' % (accuracy))

        # Print the number of classification errors from the test data
        targetPred = model.predict(test_data)
        nbErrors = np.sum(np.argmax(targetPred, axis=-1) != np.argmax(test_target, axis=-1))
        accuracy = (len(test_data) - nbErrors) / len(test_data)
        print('Classification accuracy (test set): %0.3f' % (accuracy))


if __name__ == "__main__":
    main()

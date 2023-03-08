from tensorflow import keras
import numpy as np
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, concatenate


def loadmodel(name):
    net = keras.models.load_model(r'name')
    return net


def init_myCNN(shapes):

    inputlayer = Input(shape=shapes)    # x_train.shape[1:]

    pipe1_layer1 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(inputlayer)
    pipe1_layer2 = MaxPool2D(pool_size=(2, 2))(pipe1_layer1)
    pipe1_layer3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(pipe1_layer2)
    pipe1_layer4 = MaxPool2D(pool_size=(2, 2))(pipe1_layer3)
    pipe1_layerFlat = Flatten()(pipe1_layer4)

    pipe2_layer1 = Conv2D(filters=10, kernel_size=(19, 19), activation='sigmoid')(inputlayer)
    pipe2_layer2 = MaxPool2D(pool_size=(2, 2))(pipe2_layer1)
    pipe2_layerFlat = Flatten()(pipe2_layer2)

    pipe3_layerFlat = Flatten()(inputlayer)
    pipe3_layer1 = Dropout(rate=0.01)(pipe3_layerFlat)
    pipe3_layer2 = Dense(800, activation='sigmoid')(pipe3_layer1)
    pipe3_layer3 = Dense(256, activation='sigmoid')(pipe3_layer2)
    pipe3_layer4 = Dropout(rate=0.01)(pipe3_layer3)

    layerconcat = concatenate([pipe1_layerFlat, pipe2_layerFlat, pipe3_layer4])

    blend_layer1 = Dense(256, activation='relu')(layerconcat)
    blend_layer2 = Dense(128, activation='relu')(blend_layer1)

    outputlayer = Dense(10, activation='softmax')(blend_layer2)

    net = Model(inputs=inputlayer, outputs=outputlayer)

    net.summary()
    net.compile(loss='categorical_crossentropy', optimizer=opt)

    return net
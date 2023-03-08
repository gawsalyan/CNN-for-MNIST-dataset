from tensorflow import keras
import numpy as np
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, concatenate
from sklearn.model_selection import KFold


def loadmodel(name):
    net = keras.models.load_model(r'name')
    return net


def plotHistory(history):
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def kFoldTrain(net, inputs, targets, n_splits=7, batch_size=256, epochs=1, shuffle=True):
    # Merge inputs and targets
    # inputs = np.concatenate((x_train, x_test), axis=0)
    # targets = np.concatenate((y_train, y_test), axis=0)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(inputs, targets):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = net.fit(inputs[train], targets[train], batch_size=batch_size, epochs=epochs)

        # Generate generalization metrics
        scores = net.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {net.metrics_names[0]} of {scores[0]}; {net.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    return net


def init_myCNN(shapes):
    inputlayer = Input(shape=shapes)  # x_train.shape[1:]

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

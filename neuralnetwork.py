from tensorflow import keras
import numpy as np
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, concatenate
from sklearn.model_selection import KFold


def load_model(name):
    net = keras.models.load_model(r'name')
    return net


def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def session_train(net, x_train, y_train, labels_train, x_test, y_test, labels_test, **kwargs):

    history = []
    for session in range(0, kwargs['session_max']):
        if kwargs['typeSession'] == "kFold":
            # Merge inputs and targets
            inputs = np.concatenate((x_train, x_test), axis=0)
            targets = np.concatenate((y_train, y_test), axis=0)

            net, historyT = kfold_train(net, inputs, targets, **kwargs)
            history.append(historyT)

            outputTrains = net.predict(x_train)
            labels_Trains = np.argmax(outputTrains, axis=1)
            misclassified = sum(labels_Trains != labels_train)
            accuracyTrains = 100 * (1 - misclassified / labels_train.size)
            outputs = net.predict(x_test)
            labels_predicted = np.argmax(outputs, axis=1)
            misclassified = sum(labels_predicted != labels_test)
            print('Percentage misclassified = ', 100 * misclassified / labels_test.size)
            accuracy = 100 * (1 - misclassified / labels_test.size)
            print('Percentage accuracy = ', accuracy)
            name = 'CNN_LG3_' + str(session) + '_TsAcc' + str(round(accuracy, 2)) \
                   + '_TrAcc' + str(round(accuracyTrains, 2)) + '_CV_ADAM_for_minist.h5'
        else:

            history = net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=kwargs['epochs'], batch_size=kwargs['batch_size'])
            outputs = net.predict(x_test)
            labels_predicted = np.argmax(outputs, axis=1)
            misclassified = sum(labels_predicted != labels_test)
            print('Percentage misclassified = ', 100 * misclassified / labels_test.size)
            accuracy = 100 * (1 - misclassified / labels_test.size)
            print('Percentage accuracy = ', 100 * (1 - misclassified / labels_test.size))
            name = 'CNN_LG_' + str(session) + '_' + str(accuracy) + '_for_minist.h5'

        if kwargs['saveEvery']:
            net.save(name)

    return net, history


def kfold_train(net, inputs, targets, **kwargs):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=kwargs['n_splits'], shuffle=kwargs['shuffle'])

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    history = []
    for train, test in kfold.split(inputs, targets):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history.append(net.fit(inputs[train], targets[train], batch_size=kwargs['batch_size'], epochs=kwargs['epochs']))

        # Generate generalization metrics
        scores = net.evaluate(inputs[test], targets[test], verbose=0)
        print(f'Score for fold {fold_no}: {net.metrics_names[0]} of {scores[0]}; {net.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    return net, history


def init_mycnn(shapes):
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
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    net.compile(loss='categorical_crossentropy', optimizer=opt)

    return net

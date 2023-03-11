import numpy as np
from tensorflow import keras
from loaddata import get_MNIST
from neuralnetwork import load_model, init_mycnn, session_train, plot_history


if __name__ == "__main__":

    print('Loading MNIST dataset ...')
    (x_train, y_train, labels_train), (x_test, y_test, labels_test) = get_MNIST()

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    initType = "new"
    if initType == "load":
        print('Loading existing model ...')
        net = load_model("model_name.h5")
    elif initType == "new":
        print('Initialising CNN mmodel ...')
        net = init_mycnn(x_train.shape[1:])

    options = {"session_max": 10,
               "typeSession": "kFold",
               "saveEvery": False,
               "n_splits": 7,
               "batch_size": 256,
               "epochs": 1,
               "shuffle": True
               }


    history = []

    print(' ')
    print('Training Initiated ...')
    net, history = session_train(net, x_train, y_train, labels_train, x_test, y_test, labels_test, **options)

    history.append(history)

    outputs = net.predict(x_test)
    labels_predicted = np.argmax(outputs, axis=1)

    misclassified = sum(labels_predicted != labels_test)
    print('Percentage misclassified = ', 100 * misclassified / labels_test.size)

    accuracy = 100 * (1 - misclassified / labels_test.size)
    print('Percentage accuracy = ', 100 * (1 - misclassified / labels_test.size))

    name = 'CNN_final_' + str(accuracy) + '_for_mnist.h5'
    net.save(name)

    plot_history(history)

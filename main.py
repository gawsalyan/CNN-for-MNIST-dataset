import numpy as np
from tensorflow import keras
from loaddata import get_MNIST
from neuralnetwork import load_model, init_mycnn, plot_history


if __name__ == "__main__":

    (x_train, y_train, labels_train), (x_test, y_test, labels_test) = get_MNIST()

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    initType = "new"
    if initType == "load":
        net = load_model("model_name.h5")
    elif initType == "new":
        net = init_mycnn(x_train.shape[1:])

    history = []
    for session in range(1, 100):

        history.append(net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=1024))

        outputs = net.predict(x_test)
        labels_predicted = np.argmax(outputs, axis=1)

        misclassified = sum(labels_predicted != labels_test)
        print('Percentage misclassified = ', 100 * misclassified / labels_test.size)

        accuracy = 100 * (1 - misclassified / labels_test.size)
        print('Percentage accuracy = ', 100 * (1 - misclassified / labels_test.size))

        name = 'CNN_' + str(session) + '_' + str(accuracy) + '_for_mnist.h5'
        net.save(name)

    plot_history(history)

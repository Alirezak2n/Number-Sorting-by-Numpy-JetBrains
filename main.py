import numpy as np

# global parameters.
batch_size = 32
numbers_in_input_set = 10
maximum_number_in_set = 100
multiply = batch_size * numbers_in_input_set * maximum_number_in_set
data_split = int(0.5 * multiply)


class Layer:
    """
    It is a based layer class which is used by any other layer
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class Network:
    """
    It is the main Network class for our layers
    """

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):

        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))


class FCLayer(Layer):
    """
    This is the main layer of our Model
    """

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    """
    This is the activation layer which is called by activation function
    """

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


def softmax(x):
    """
    The activation function
    """
    return np.exp(x) / np.sum(np.exp(x))  # np.tanh(x)


def tanh_bw(x):
    return 1 - np.power(np.tanh(x), 2)


def mse(y_true, y_pred):
    """
    The loss function
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_bw(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def encoding_loop(inputed_vector, matrix):
    """
    This Function is used for converting input to matrix by changing the corresponding element of matrix to 1 from 0
    """
    for i, j in enumerate(inputed_vector):
        for k, l in enumerate(j):
            matrix[i, k, l] = 1
    return matrix


def matrix_zero_maker(no_matrix, rows=numbers_in_input_set, columns=maximum_number_in_set):
    """
    This Function is used for making matrix of zeros for base encoding process with numpy
    """
    matrix_zeros = np.zeros((no_matrix, rows, columns), dtype=np.float32)
    return matrix_zeros


def hot_encoder(input_set):
    """
    This function is hot encoder that convert set to matrix so it can be used in machine learning
    """
    input_set_matrix = matrix_zero_maker(len(input_set))
    encoding_loop(input_set, input_set_matrix)
    return input_set_matrix


def batch_generator(batches_size):
    """
    This Function is used for generating and encoding training batch
    """
    input_set_encoded = matrix_zero_maker(batches_size)
    sorted_input_set_encoded = matrix_zero_maker(batches_size)

    input_set_vectors = np.random.randint(maximum_number_in_set, size=(batches_size, numbers_in_input_set))

    sorted_input_set_vectors = np.sort(input_set_vectors, axis=1)

    encoding_loop(input_set_vectors, input_set_encoded)

    encoding_loop(sorted_input_set_vectors, sorted_input_set_encoded)

    inputed_set_encoded_1d = input_set_encoded.reshape((1, 1, multiply))
    sorted_input_set_encoded_1d = sorted_input_set_encoded.reshape((1, 1, multiply))

    yield inputed_set_encoded_1d, sorted_input_set_encoded_1d


# Preparing Data

for inputed, sorted_inputed in batch_generator(batch_size):
    x_train, x_test = inputed[:, :, :data_split], inputed[:, :, data_split:]
    y_train, y_test = sorted_inputed[:, :, :data_split], sorted_inputed[:, :, data_split:]

# Model and Layers

autoEncoder = Network()

autoEncoder.add(FCLayer(data_split, int(data_split / 10)))
autoEncoder.add(ActivationLayer(softmax, tanh_bw))
autoEncoder.add(FCLayer(int(data_split / 10), 1000))
autoEncoder.add(ActivationLayer(softmax, tanh_bw))
autoEncoder.add(FCLayer(1000, maximum_number_in_set))
autoEncoder.add(ActivationLayer(softmax, tanh_bw))
autoEncoder.add(FCLayer(maximum_number_in_set, 1000))
autoEncoder.add(ActivationLayer(softmax, tanh_bw))
autoEncoder.add(FCLayer(1000, data_split))

autoEncoder.use(mse, mse_bw)

autoEncoder.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

out = autoEncoder.predict(x_test[0:3])

print("Sorted data by my model is : ")
print(np.argmax(out, axis=1))
print("Exact data is : ")
print(np.argmax(y_test[0:3], axis=1))

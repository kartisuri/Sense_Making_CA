"""Human activity recognition using inertial dataset and an LSTM RNN."""
import tensorflow as tf
import numpy as np
import mat_to_txt

# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.split(',') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)
def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 72 training series
        self.test_data_count = len(X_test)  # 24 testing series
        self.n_steps = len(X_train[0])  # 209 time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 15

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 6: 2 * 3D sensors features over time
        self.n_hidden = 25  # nb of neurons inside the neural network
        self.n_classes = 3  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells
    Two LSTM cells are stacked which adds deepness to the neural network.
    """

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def inertial_LSTM_RNN():
    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised inertial input features for the neural network
    INPUT_INERTIAL_SIGNAL_TYPES = [
        "acc_x_",
        "acc_y_",
        "acc_z_",
        "gyro_x_",
        "gyro_y_",
        "gyro_z_",
    ]

    INERTIAL = "ProcessedData/InertialData/"

    X_train_inertial_signals_paths = [
        INERTIAL + signal_file + "train.txt" for signal_file in INPUT_INERTIAL_SIGNAL_TYPES
    ]
    X_test_inertial_signals_paths = [
        INERTIAL + signal_file + "test.txt" for signal_file in INPUT_INERTIAL_SIGNAL_TYPES
    ]

    X_inertial_train = load_X(X_train_inertial_signals_paths)
    X_inertial_test = load_X(X_test_inertial_signals_paths)

    y_inertial_train_path = INERTIAL + "y_train.txt"
    y_inertial_test_path = INERTIAL + "y_test.txt"
    y_inertial_train = one_hot(load_y(y_inertial_train_path))
    y_inertial_test = one_hot(load_y(y_inertial_test_path))

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config_inertial = Config(X_inertial_train, X_inertial_test)
    print("################################ INERTIAL #############################")
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_inertial_test.shape, y_inertial_test.shape,
          np.mean(X_inertial_test), np.std(X_inertial_test))
    print("the dataset is therefore properly normalised, as expected.")
    
    # ------------------------------------------------------
    # Step 3: Build the neural network
    # ------------------------------------------------------

    XI = tf.placeholder(tf.float32, [None, config_inertial.n_steps, config_inertial.n_inputs])
    YI = tf.placeholder(tf.float32, [None, config_inertial.n_classes])
    pred_YI = LSTM_Network(XI, config_inertial)
    
    # Loss,optimizer,evaluation
    l2_I = config_inertial.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    # Softmax loss and L2
    cost_I = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=YI, logits=pred_YI)) + l2_I
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=config_inertial.learning_rate).minimize(cost_I)
    correct_pred_I = tf.equal(tf.argmax(pred_YI, 1), tf.argmax(YI, 1))
    accuracy_I = tf.reduce_mean(tf.cast(correct_pred_I, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Train the neural network
    # --------------------------------------------
    sess_I = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init_I = tf.global_variables_initializer()
    sess_I.run(init_I)
    best_accuracy_I = 0.0

    # Start training for each batch and loop epochs for inertial data
    for i in range(config_inertial.training_epochs):
        for start, end in zip(range(0, config_inertial.train_count, config_inertial.batch_size),
                              range(config_inertial.batch_size, config_inertial.train_count + 1, 
                              	    config_inertial.batch_size)):
            sess_I.run(optimizer, feed_dict={XI: X_inertial_train[start:end],
                                             YI: y_inertial_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out_I, accuracy_out_I, loss_out_I = sess_I.run(
            [pred_YI, accuracy_I, cost_I],
            feed_dict={
                XI: X_inertial_test,
                YI: y_inertial_test
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out_I) +
              " loss : {}".format(loss_out_I))
        best_accuracy_I = max(best_accuracy_I, accuracy_out_I)

    print("")
    print("final test accuracy: {}".format(accuracy_out_I))
    print("best epoch's test accuracy: {}".format(best_accuracy_I))
    print("")
    return accuracy_out_I, best_accuracy_I

if __name__ == "__main__":
    inertial_LSTM_RNN()
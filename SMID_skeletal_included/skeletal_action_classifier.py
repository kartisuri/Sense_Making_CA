"""Human activity recognition using skeletal dataset and an LSTM RNN."""
import tensorflow as tf
import numpy as np

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
        self.train_count = len(X_train)  # training series
        self.test_data_count = len(X_test)  # testing series
        self.n_steps = len(X_train[0])  # time_steps per series

        # Training
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 8

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count
        self.n_hidden = 52  # nb of neurons inside the neural network
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
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
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

def skeletal_LSTM_RNN():
    # -----------------------------
    # Step 1: load and prepare data
    # -----------------------------

    # Those are separate normalised skeletal input features for the neural network
    INPUT_SKELETAL_JOINT_TYPES = [
        "joint_1_x_", "joint_1_y_", "joint_1_z_",
        "joint_2_x_", "joint_2_y_", "joint_2_z_",
        "joint_3_x_", "joint_3_y_", "joint_3_z_",
        "joint_4_x_", "joint_4_y_", "joint_4_z_",
        "joint_5_x_", "joint_5_y_", "joint_5_z_",
        "joint_6_x_", "joint_6_y_", "joint_6_z_",
        "joint_7_x_", "joint_7_y_", "joint_7_z_",
        "joint_8_x_", "joint_8_y_", "joint_8_z_",
        "joint_9_x_", "joint_9_y_", "joint_9_z_",
        "joint_10_x_", "joint_10_y_", "joint_10_z_",
        "joint_11_x_", "joint_11_y_", "joint_11_z_",
        "joint_12_x_", "joint_12_y_", "joint_12_z_",
        "joint_13_x_", "joint_13_y_", "joint_13_z_",
        "joint_14_x_", "joint_14_y_", "joint_14_z_",
        "joint_15_x_", "joint_15_y_", "joint_15_z_",
        "joint_16_x_", "joint_16_y_", "joint_16_z_",
        "joint_17_x_", "joint_17_y_", "joint_17_z_",
        "joint_18_x_", "joint_18_y_", "joint_18_z_",
        "joint_19_x_", "joint_19_y_", "joint_19_z_",
        "joint_20_x_", "joint_20_y_", "joint_20_z_",
    ]

    SKELETAL = "ProcessedData/SkeletalData/"

    X_train_skeletal_joints_paths = [
        SKELETAL + joint_file + "train.txt" for joint_file in INPUT_SKELETAL_JOINT_TYPES
    ]
    X_test_skeletal_joints_paths = [
        SKELETAL + joint_file + "test.txt" for joint_file in INPUT_SKELETAL_JOINT_TYPES
    ]

    X_skeletal_train = load_X(X_train_skeletal_joints_paths)
    X_skeletal_test = load_X(X_test_skeletal_joints_paths)

    y_skeletal_train_path = SKELETAL + "y_train.txt"
    y_skeletal_test_path = SKELETAL + "y_test.txt"
    y_skeletal_train = one_hot(load_y(y_skeletal_train_path))
    y_skeletal_test = one_hot(load_y(y_skeletal_test_path))

    # -----------------------------------
    # Step 2: define parameters for model
    # -----------------------------------

    config_skeletal = Config(X_skeletal_train, X_skeletal_test)
    print("################################ SKELETAL #############################")
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_skeletal_test.shape, y_skeletal_test.shape,
          np.mean(X_skeletal_test), np.std(X_skeletal_test))
    print("the dataset is therefore properly normalised, as expected.")

    # ------------------------------------------------------
    # Step 3: Build the neural network
    # ------------------------------------------------------

    XS = tf.placeholder(tf.float32, [None, config_skeletal.n_steps, config_skeletal.n_inputs])
    YS = tf.placeholder(tf.float32, [None, config_skeletal.n_classes])
    pred_YS = LSTM_Network(XS, config_skeletal)

    l2_S = config_skeletal.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    cost_S = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=YS, logits=pred_YS)) + l2_S
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=config_skeletal.learning_rate).minimize(cost_S)
    correct_pred_S = tf.equal(tf.argmax(pred_YS, 1), tf.argmax(YS, 1))
    accuracy_S = tf.reduce_mean(tf.cast(correct_pred_S, dtype=tf.float32))

    # --------------------------------------------
    # Step 4: Train the neural network
    # --------------------------------------------
    
    sess_S = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init_S = tf.global_variables_initializer()
    sess_S.run(init_S)
    best_accuracy_S = 0.0

    # Start training for each batch and loop epochs for skeletal data
    for i in range(config_skeletal.training_epochs):
        for start, end in zip(range(0, config_skeletal.train_count, config_skeletal.batch_size),
                              range(config_skeletal.batch_size, config_skeletal.train_count + 1, 
                              	    config_skeletal.batch_size)):
            sess_S.run(optimizer, feed_dict={XS: X_skeletal_train[start:end],
                                             YS: y_skeletal_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out_S, accuracy_out_S, loss_out_S = sess_S.run(
            [pred_YS, accuracy_S, cost_S],
            feed_dict={
                XS: X_skeletal_test,
                YS: y_skeletal_test
            }
        )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out_S) +
              " loss : {}".format(loss_out_S))
        best_accuracy_S = max(best_accuracy_S, accuracy_out_S)

    print("")
    print("final test accuracy: {}".format(accuracy_out_S))
    print("best epoch's test accuracy: {}".format(best_accuracy_S))
    print("")
    return (accuracy_out_S, best_accuracy_S)

if __name__ == "__main__":
    skeletal_LSTM_RNN()
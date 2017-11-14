import mat_to_txt
from skeletal_action_classifier import skeletal_LSTM_RNN
from inertial_action_classifier import inertial_LSTM_RNN

acc_i, best_acc_i = inertial_LSTM_RNN()
acc_s, best_acc_s = skeletal_LSTM_RNN()

print("FUSION........................")
inertial_weight = 0.6
skeletal_weight = 0.4

overall_accuracy_out = acc_s*skeletal_weight + acc_i*inertial_weight
overall_accuracy_best = best_acc_s*skeletal_weight + best_acc_i*inertial_weight

print("")
print("Final overall test accuracy: {}".format(overall_accuracy_out))
print("best epoch's overall test accuracy: {}".format(overall_accuracy_best))
print("")

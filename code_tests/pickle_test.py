import pickle

f = open("../objects/test.pckl", "wb")
pickle.dump([mean_model_train_acc,
             mean_model_test_acc,
             train_acc_dict,
             test_acc_dict,
             weights_dict,
             y_pred_dict], f)
f.close()

f = open("../objects/test.pckl", "rb")
mean_model_train_acc, mean_model_test_acc, train_acc_dict, test_acc_dict, weights_dict, y_pred_dict = pickle.load(f)
f.close()

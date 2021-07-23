import generate_parties
import Server_Parties_Interface
import server_class
import Prediction_and_Classification_Performance
import Import_Data
import numpy as np
import random
import timeit
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

Initial_seed = random.randint(1, 10 ** 5)
random.seed(Initial_seed)


##IMPORT DATA
Data_Set = "depression_new_approach"
data_set, attribute_information, number_target_classes = Import_Data.import_data(Data_Set)


patients_indices = np.unique(data_set[:, 0])
indices_all = np.arange(0, len(data_set[:, 0]))

groundTruth = []
prediction_list_ppdert = []


def find_idx(vec1, vec2):
    list = []
    for value in vec2:
        list.append(np.argwhere(vec1 == value))
    result = np.concatenate(np.asarray(list)).flatten()
    return result

round_num = 0
kf = KFold(n_splits=55, random_state=0, shuffle=False)
for patient_train_index, patient_test_index in kf.split(patients_indices):
    round_num += 1
    print("round ", round_num)
    print(patients_indices[patient_train_index], patients_indices[patient_test_index])
    train_index, test_index = \
        find_idx(data_set[:, 0], patients_indices[patient_train_index]),\
        find_idx(data_set[:, 0], patients_indices[patient_test_index])
    train_set, test_set = data_set[train_index, 1:], data_set[test_index, 1:]

    attributes_range = Import_Data.find_range_of_data(train_set, attribute_information)


    Perform_PPDERT = True
    if Perform_PPDERT:
        # PPD-ERT

        # Settings
        Data_split_train_test_seed = random.randint(1, 10 ** 5)
        global_seed = random.randint(1, 10 ** 5)
        number_of_parties = 2 #80
        number_of_trees = 200
        # generate personal random seeds
        personal_random_seeds = [random.randint(0, 10000 * number_of_parties) for p in range(0, number_of_parties)]


        attribute_percentage = np.around(np.sqrt(len(attribute_information)) / len(attribute_information), decimals=3)

        num_parties = number_of_parties

        parties_all = generate_parties.generate(global_seed=global_seed, number_of_parties=number_of_parties,
                                                train_set=train_set, \
                                                attribute_information=attribute_information, \
                                                number_target_classes=number_target_classes, \
                                                attributes_range=attributes_range, \
                                                personal_random_seeds=personal_random_seeds, \
                                                attribute_percentage=attribute_percentage, \
                                                Data_split_train_test_seed=Data_split_train_test_seed)

        included_parties_indices = np.array(range(0,num_parties))
        parties = parties_all[0:num_parties]

        # instantiate Interface class for communications between server and parties
        Interface = Server_Parties_Interface.interface(parties)

        # instantiate server
        server = server_class.server(global_seed=global_seed, attribute_range=attributes_range, \
                                     attribute_info=attribute_information, \
                                     num_target_classes=number_target_classes, \
                                     aggregator_func=Interface.aggregator, \
                                     parties_update_func=Interface.parties_update, \
                                     personal_random_seeds=personal_random_seeds, \
                                     attribute_percentage=attribute_percentage,\
                                     included_parties_indices=included_parties_indices ,\
                                     parties_reset_func=Interface.parties_reset)

        print("========================================")
        print("LEARNING...")
        start = timeit.default_timer()
        list_of_trees = server.make_tree_group(impurity_measure='entropy', num_of_trees=number_of_trees)
        stop = timeit.default_timer()
        print("A Group of ", number_of_trees, "Trees are Learned!")
        print("Elapsed Time: ", stop-start, " Sec")



        print("========================================")
        print("CLASSIFICATION PERFORMANCE...")

        prediction, true_labels = \
            Prediction_and_Classification_Performance.get_result_vectors(list_of_trees, test_set)

        groundTruth.append(true_labels)
        prediction_list_ppdert.append(prediction)



def print_results(labels_vec, predictions_vec):
    tn, fp, fn, tp = confusion_matrix(labels_vec, predictions_vec).ravel()
    f1_performance = f1_score(labels_vec, predictions_vec, average='weighted')
    acc_performance = accuracy_score(labels_vec, predictions_vec)
    mcc_performance = matthews_corrcoef(labels_vec, predictions_vec)
    print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    print("f1_performance: ", f1_performance)
    print("acc_performance", acc_performance)
    print("mcc_performance: ", mcc_performance)


print("CLASSIFICATION PERFORMANCE...")
groundTruth_vec = np.concatenate(np.asarray(groundTruth))
prediction_ppdert_vec = np.concatenate(np.asarray(prediction_list_ppdert))
print("groundTruth_vec and prediction_ert_vec:")
print_results(groundTruth_vec, prediction_ppdert_vec)


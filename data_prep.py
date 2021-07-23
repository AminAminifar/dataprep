import numpy as np
import os
import pandas as pd
from pathlib import Path
from natsort import natsorted
from random import sample


def load_files(file_names, the_path, days):
    activity_list = []
    timestamp_list = []
    for i, filename in enumerate(file_names):
        data_pd = pd.read_csv(the_path/filename)
        dates = data_pd['date'].unique()
        num_days = days[i]
        for day in range(0, num_days):
            indices = np.where(data_pd['date'] == dates[day])[0]
            activity_vector = data_pd['activity'][indices].to_numpy()
            time_vector = data_pd['timestamp'][indices]
            new_time = pd.to_datetime(time_vector).dt.hour*60+pd.to_datetime(time_vector).dt.minute
            new_time_numpy = new_time.to_numpy()
            activity_list.append(activity_vector)
            timestamp_list.append(new_time_numpy)
    return activity_list, timestamp_list


#  import scores.csv
scores = pd.read_csv('data/scores.csv')
days_all = scores['days'].to_numpy()
days_condition = days_all[0:23]
days_control = days_all[23:55]

#  import condition
path = Path('data/condition')
data_files = os.listdir(path)
data_files = natsorted(data_files)
data_condition, timestamp_condition = load_files(file_names=data_files, the_path=path, days=days_condition)


#  import control
path = Path('data/control')
data_files = os.listdir(path)
data_files = natsorted(data_files)

data_control, timestamp_control = load_files(file_names=data_files, the_path=path, days=days_control)

num_samples = 100
len_sample = 60*24


def create_new_sample(collected_data, collected_timestamp, len_samples):
    new_sample = []
    for i in range(0,len_samples):
        diff = abs(collected_timestamp-i)
        ind = np.where(abs(diff - diff.min()) <= 10)
        arr = np.asarray(ind).tolist()
        final_ind = sample(arr[0], 1)
        new_sample.extend(collected_data[final_ind])
    return new_sample


def feature_extraction(data_array, timestamp_array, index_array, num_samples, len_samples):
    engineered_data = []
    collected_data = []
    collected_timestamp = []
    for record in range(0, len(index_array)):
        collected_data = np.concatenate([collected_data, data_array[record]])
        collected_timestamp = np.concatenate([collected_timestamp, timestamp_array[record]])
        if (record == len(index_array) - 1  or index_array[record] < index_array[record + 1]):
            for new_record in range(0, num_samples):
                new_sample = create_new_sample(collected_data, collected_timestamp, len_samples)
                engineered_data.append(new_sample)
            collected_data = []
            collected_timestamp = []
    engineered_data = np.array(engineered_data)
    return engineered_data


index_list_condition = []
index_list_control = []

#  condition
for patient in range(len(days_condition)):
    index_list_condition.append(np.ones((days_condition[patient], 1), dtype=int)*(1+patient))
#  control
for patient in range(len(days_control)):
    index_list_control.append(np.ones((days_control[patient], 1), dtype=int)*(24+patient))

index_array_condition = np.vstack(index_list_condition)
index_array_control = np.vstack(index_list_control)

#  condition
data_condition_FE = feature_extraction(data_condition, timestamp_condition, index_array_condition, num_samples, len_sample)
#  control
data_control_FE = feature_extraction(data_control, timestamp_control, index_array_control, num_samples, len_sample)

#  add labels
#  condition
labels_condition = np.ones((len(data_condition_FE[:, 0]), 1), dtype=int)
data_condition_labeled = np.append(data_condition_FE, labels_condition, axis=1)
#  control
labels_control = np.zeros((len(data_control_FE[:, 0]), 1), dtype=int)
data_control_labeled = np.append(data_control_FE, labels_control, axis=1)


#  add patient index
index_list = []
#  condition
for patient in range(23):
    index_list.append(np.ones((num_samples, 1), dtype=int)*(1+patient))
#  control
for patient in range((55-23)):
    index_list.append(np.ones((num_samples, 1), dtype=int)*(24+patient))
index_array = np.vstack(index_list)


#  merge all
data_all = np.append(data_condition_labeled, data_control_labeled, axis=0)
data_all_with_patient_index = np.append(index_array, data_all, axis=1)
np.random.shuffle(data_all_with_patient_index)

#  save the result
pd.DataFrame(data_all_with_patient_index).to_csv("data_all_with_patient_index.csv", header=None, index=None)


# Please find the sourcecode for PPD-ERT at https://github.com/paperepo/PPD-ERT



# Copyright 2022 Simon Smith. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Authors: Simon Smith <simonsmith95@hotmail.com>
#
# Purpose: Part of a motion detection project in the course
# "Deep Learning - methods and applications 5TF078"
#
# Content: Functions and main script for loading data and a base analysis + reshaping.
# ==============================================================================
# Change history:
#    2022-05-01 (SiSm) Created
#    2022-05-26 (SiSm) Submitted to course
#    2022-05-27 (SiSm) Adding open license
# ==============================================================================


import collections
import keras
import keras.utils.np_utils
import pandas as pd
import glob
import os
from os.path import exists
import fileinput as fi
import numpy as np
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
import matplotlib.pyplot as plt


# Changing max size for printing pandas df.


def fetch_metric(fp, set_target):
    """
    A function that takes a list of filepaths, creates a pd.Dataframe containing the info from each file. If the
    file contain more than one row of samples all the rows below the first data row is removed before appending it to
    the final df that gets returned.

    :param fp: A list of the filepaths we want to read in.
    :param set_target: The target we want for the files in the folder. 0 for acl, 1 for control, 2 for athletes.
    :return: A dataframe containing the data from the files we just read in.
    """
    # Iterating over all filepaths given.
    dict_with_values = collections.defaultdict(list)
    prev_entries = 0
    current_max = 0
    for filepath in fp:
        print(f'IMPORTING FILE: {filepath}')
        # Loading data as df and extracting the rows needed.

        temp_df = pd.read_csv(filepath, delimiter="\t")
        #print(temp_df)
        temp_df.drop(temp_df.columns[0], axis=1, inplace=True)
        patient_name = [i for i in temp_df.columns]

        # Combining the sensors location with the axis the datapoint is regarding.
        # skipping inclution of node since they still have unique names. No Metric data with the same name.
        categories_of_data = temp_df.iloc[0] + temp_df.iloc[3]


        # Getting a list of all the measured values.
        actual_datapoints = [i for i in temp_df.iloc[4]]
        list_of_columns_to_create = [str(i) for i in categories_of_data]
        # Checking for missing data.
        d = dict()
        for value in list_of_columns_to_create:
            if value not in d:
                d[value] = 1
            else:
                d[value] += 1
        number_reps = max(d.values())
        unique_cols = list(d.keys())
        # Making a new list of all correct columns. Meaning the missing columns have been added.
        correct_cols = unique_cols * number_reps
        index_to_add_nan = []
        # Finding which categories actually are missing.
        missing_categories = [k for k, v in d.items() if float(v) < number_reps]
        fixed = 0
        for i in range(len(list_of_columns_to_create)):
            if correct_cols[i + fixed] != list_of_columns_to_create[i]:
                index_to_add_nan.append(i)
                fixed += 1

        for index in index_to_add_nan:
            actual_datapoints.insert(index, np.nan)


        # Making sure each datapoint have a corresponding description of location for the sensor.
        if len(actual_datapoints) != len(correct_cols):
            print('Something wrong with the import. Number of datapoints do not match categories')
            break

        # Setting up list containing the pairs of sensor location and the actual value.
        setup_for_dict = []
        for i in range(len(correct_cols)):
            setup_for_dict.append([correct_cols[i], actual_datapoints[i]])

        # Setting up dict to create dataframe of based on the setup list above.
        for sublist in setup_for_dict:
            if sublist[0] in dict_with_values.keys():
                dict_with_values[sublist[0]].append(sublist[1])
            else:
                dict_with_values[sublist[0]] = [sublist[1]]


        for k, v in dict_with_values.items():
            current_max = max(current_max, len(list(filter(None, v))))
        number_of_repetitions_added = current_max - prev_entries
        prev_entries = current_max
        for i in range(number_of_repetitions_added):
            if 'Patient_name' in dict_with_values.keys():
                dict_with_values['Target'].append(set_target)
                dict_with_values['Patient_name'].append(patient_name[1] + str(i+1))
            else:
                dict_with_values['Target'] = [set_target]
                dict_with_values['Patient_name'] = [patient_name[1] + str(i+1)]

        # Creating df for the patient.
    folder_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_with_values.items()]))

    #print(folder_df.head(500))
    #print(folder_df[folder_df.isnull().any(axis=1)])
    folder_df.dropna(inplace=True)
    return folder_df


def fetch_p2d(fp, set_target):
    """
        A function that takes a list of filepaths, creates a pd.Dataframe containing the info from each file. If the
        file contain more than one row of samples all the rows below the first data row is removed before appending it to
        the final df that gets returned.

        :param fp: A list of the filepaths we want to read in.
        :param set_target: The target we want for the files in the folder. 0 for acl, 1 for controll, 2 for athletes.
        :return: A dataframe containing the data from the files we just read in.
        """
    # Iterating over all filepaths given.
    dict_with_values = collections.defaultdict(list)
    dict_list_of_list = collections.defaultdict(list)
    prev_entries = 0
    breaking = 0
    current_max = 0
    for filepath in fp:
        print(f'IMPORTING FILE: {filepath}')
        # Loading data as df and extracting the rows needed.
        temp_df = pd.read_csv(filepath, delimiter="\t")
        temp_df.drop(temp_df.columns[0], axis=1, inplace=True)
        patient_name = [i for i in temp_df.columns]


        # Combining the sensors location with the axis the datapoint is regarding.
        # skipping inclution of node since they still have unique names. No Metric data with the same name.
        categories_of_data = temp_df.iloc[0] + temp_df.iloc[3]

        # Getting a list of all the measured values.
        actual_datapoints = [temp_df[i].iloc[4:].tolist() for i in temp_df.columns]
        list_of_columns_to_create = [i for i in categories_of_data]

        # Making sure each datapoint have a corresponding description of location for the sensor.
        if len(actual_datapoints) != len(list_of_columns_to_create):
            print('Something wrong with the import. Number of datapoints do not match categories')
            breaking += 1
            break

        # Setting up list containing the pairs of sensor location and the actual value.
        setup_for_dict = []
        for i in range(len(list_of_columns_to_create)):
            setup_for_dict.append([list_of_columns_to_create[i], actual_datapoints[i]])

        # Setting up dict to create dataframe of based on the setup list above.
        for sublist in setup_for_dict:
            if sublist[0] in dict_with_values.keys():
                dict_with_values[sublist[0]].append(sublist[1])
            else:
                dict_with_values[sublist[0]] = [sublist[1]]


        number_of_cols_in_temp_df = len(list_of_columns_to_create)

        # Getting the amount of unique cols on first iteration before target and name is added.
        if prev_entries == 0:
            number_of_unique_cols = len(dict_with_values.keys())
        number_of_repetitions = number_of_cols_in_temp_df / number_of_unique_cols
        temp_df_cols = [i for i in temp_df.columns]
        # Slicing df width so that that repetitions don't get added as columns.
        for i in range(1, int(number_of_repetitions) + 1):
            # Creating df with the slice of the columns containing one repetitions data.
            new_df = temp_df.loc[:, (temp_df_cols[(i - 1) * int(number_of_unique_cols):(i * int(number_of_unique_cols))])]
            columns_of_data = new_df.iloc[0] + new_df.iloc[3]

            # Dropping all values but the actual datapoints.
            new_df.drop([0, 1, 2, 3], inplace=True)
            # Checking for nan values and only adding if the set is intact.
            if not new_df.isna().values.any():
                # Making the datapoints into a list of lists.
                actual_data = new_df.values.tolist()

                # Filling dictionary to save each repetitions data.
                if 'patient_name' in dict_list_of_list:
                    dict_list_of_list['data'].append(actual_data)
                    dict_list_of_list['patient_name'].append(patient_name[1] + str(i))
                    dict_list_of_list['Target'].append(set_target)
                else:
                    dict_list_of_list['data'] = [actual_data]
                    dict_list_of_list['patient_name'] = [patient_name[1] + str(i)]
                    dict_list_of_list['Target'] = [set_target]



        # Keeping this segment because we use prev_entries to build dict above.
        for k, v in dict_with_values.items():
            current_max = max(current_max, len(list(filter(None, v))))
        number_of_repetitions_added = current_max - prev_entries
        prev_entries = current_max
        for i in range(number_of_repetitions_added):
            if 'Patient_name' in dict_with_values.keys():
                dict_with_values['Target'].append(set_target)
                dict_with_values['Patient_name'].append(patient_name[1])
            else:
                dict_with_values['Target'] = [set_target]
                dict_with_values['Patient_name'] = [patient_name[1]]

        # Creating df for the patient.
        # saknas värden för vissa sensorer så df blir olika lång -.- fix it.
    #print('BUILDING DF')
    folder_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_list_of_list.items()]))
    #fo_df = pd.DataFrame(dict_list_of_list)
    #print(fo_df.head(20))
    #print(dict_list_of_list.keys())
    folder_df.dropna(inplace=True)
    #print(folder_df)
    # Changing df structure from each col containing values to one col containing a list of lists
    #print(breaking)
    #print(folder_df.head(200))
    return folder_df

def build_sets(folder_path):
    """
    A function that gets all patient files on the affected or non-dominant leg from each folderpath specified below.
    In order for function to work you will have to change the paths to be the location where you store the data locally.
    The dataframes created contain the data points with set targets (0 for acl, 1 for control and 2 for athlete) and
    the patient name in the form of one sample filename.

    :param folder_path: Is the path to the folder containing all the datasets.
    :return: Two dataframes containing metric and p2d data from all different tests except olvj.

    Comments: Making individual models for each test ( OLHD, SD etc.) and then combine outputs from each test
    into an input for a new model that predicts category. Might need more data and patient name must be taken into
    consideration in order for the model to get accurate inputs. Meaning every patient must have corresponding data for
    each test.
    """
    # Setting up variables containing the paths to the 3 groups.
    ########## ACL GROUP #############

    # OLHD data paths
    olhd_acl_p2d_fp = glob.glob(f"{folder_path}/OLHD AI export/ACL Injured/anonymized/*_A_p2d.txt")
    olhd_acl_metric_fp = glob.glob(f"{folder_path}/OLHD AI export/ACL Injured/anonymized/*_A_metric.txt")

    # OLVJ data paths
    olvj_acl_p2d_fp = glob.glob(f"{folder_path}/OLVJ AI export/ACL Injured/anonymized/*_A_p2d.txt")
    # No point using metric data, only one sample avalible. Will only be an outliner decreasing performance.
    # olvj_acl_metric_fp = glob.glob("/home/wubba/ai_umu/raw_data/OLVJ AI export/ACL Injured/anonymized/*_A_metric.txt")

    # SD data paths
    sd_acl_p2d_fp = glob.glob(f"{folder_path}/SD AI export/ACL Injured/anonymized/*_A_p2d.txt")
    sd_acl_metric_fp = glob.glob(f"{folder_path}/SD AI export/ACL Injured/anonymized/*_A_metric.txt")

    # SDPS data paths
    sdps_acl_p2d_fp = glob.glob(f"{folder_path}/SDPS AI export/ACL Injured/anonymized/*_A2_p2d.txt")
    sdps_acl_metric_fp = glob.glob(f"{folder_path}/SDPS AI export/ACL Injured/anonymized/*_A2_metric.txt")

    # SH data paths
    sh_acl_p2d_fp = glob.glob(f"{folder_path}/SH AI export/ACL Injured/anonymized/*_A_p2d.txt")
    sh_acl_metric_fp = glob.glob(f"{folder_path}/SH AI export/ACL Injured/anonymized/*_A_metric.txt")


    ######### ATHLETE GROUP ##############

    # OLHD data paths
    olhd_athletes_p2d_fp = glob.glob(f"{folder_path}/OLHD AI export/Athletes/anonymized/*_ND_p2d.txt")
    olhd_athletes_metric_fp = glob.glob(f"{folder_path}/OLHD AI export/Athletes/anonymized/*_ND_metric.txt")

    # OLVJ data paths
    olvj_athletes_p2d_fp = glob.glob(f"{folder_path}/OLVJ AI export/Athletes/anonymized/*_ND_p2d.txt")
    # No point using metric data, only one sample avalible. Will only be an outliner decreasing performance.
    # olvj_acl_metric_fp = glob.glob("/home/wubba/ai_umu/raw_data/OLVJ AI export/ACL Injured/anonymized/*_A_metric.txt")

    # SD data paths
    sd_athletes_p2d_fp = glob.glob(f"{folder_path}/SD AI export/Athletes/anonymized/*_ND_p2d.txt")
    sd_athletes_metric_fp = glob.glob(f"{folder_path}/SD AI export/Athletes/anonymized/*_ND_metric.txt")

    # SDPS data paths
    sdps_athletes_p2d_fp = glob.glob(f"{folder_path}/SDPS AI export/Athletes/anonymized/*_ND2_p2d.txt")
    sdps_athletes_metric_fp = glob.glob(f"{folder_path}/SDPS AI export/Athletes/anonymized/*_ND2_metric.txt")

    # SH data paths
    sh_athletes_p2d_fp = glob.glob(f"{folder_path}/SH AI export/Athletes/anonymized/*_ND_p2d.txt")
    sh_athletes_metric_fp = glob.glob(f"{folder_path}/SH AI export/Athletes/anonymized/*_ND_metric.txt")

    ######### Controls group #############

    # OLHD data paths
    olhd_controls_p2d_fp = glob.glob(f"{folder_path}/OLHD AI export/Controls/anonymized/*_ND_p2d.txt")
    olhd_controls_metric_fp = glob.glob(f"{folder_path}/OLHD AI export/Controls/anonymized/*_ND_metric.txt")

    # OLVJ data paths
    olvj_controls_p2d_fp = glob.glob(f"{folder_path}/OLVJ AI export/Controls/anonymized/*_ND_p2d.txt")
    # No point using metric data, only one sample avalible. Will only be an outliner decreasing performance.


    # SD data paths
    sd_controls_p2d_fp = glob.glob(f"{folder_path}/SD AI export/Controls/anonymized/*_ND_p2d.txt")
    sd_controls_metric_fp = glob.glob(f"{folder_path}/SD AI export/Controls/anonymized/*_ND_metric.txt")

    # SDPS data paths
    sdps_controls_p2d_fp = glob.glob(f"{folder_path}/SDPS AI export/Controls/anonymized/*_ND2_p2d.txt")
    sdps_controls_metric_fp = glob.glob(f"{folder_path}/SDPS AI export/Controls/anonymized/*_ND2_metric.txt")

    # SH data paths
    sh_controls_p2d_fp = glob.glob(f"{folder_path}/SH AI export/Controls/anonymized/*_ND_p2d.txt")
    sh_controls_metric_fp = glob.glob(f"{folder_path}/SH AI export/Controls/anonymized/*_ND_metric.txt")

    ########## IMPORTING ############

    # OLHD
    olhd_acl_metric_df = fetch_metric(olhd_acl_metric_fp, 0)
    olhd_controls_metric_df = fetch_metric(olhd_controls_metric_fp, 1)
    olhd_athletes_metric_df = fetch_metric(olhd_athletes_metric_fp, 2)
    olhd_metric_df = pd.concat([olhd_acl_metric_df, olhd_controls_metric_df, olhd_athletes_metric_df], ignore_index=True)


    olhd_acl_p2d_df = fetch_p2d(olhd_acl_p2d_fp, 0)
    olhd_controls_p2d_df = fetch_p2d(olhd_controls_p2d_fp, 1)
    olhd_athletes_p2d_df = fetch_p2d(olhd_athletes_p2d_fp, 2)
    olhd_p2d_df = pd.concat([olhd_acl_p2d_df, olhd_controls_p2d_df, olhd_athletes_p2d_df], ignore_index=True)

    # OLVJ
    olvj_acl_p2d_df = fetch_p2d(olvj_acl_p2d_fp, 0)
    olvj_controls_p2d_df = fetch_p2d(olvj_controls_p2d_fp, 1)
    olvj_athletes_p2d_df = fetch_p2d(olvj_athletes_p2d_fp, 2)
    olvj_p2d_df = pd.concat([olvj_acl_p2d_df, olvj_controls_p2d_df, olvj_athletes_p2d_df], ignore_index=True)

    # SD
    sd_acl_metric_df = fetch_metric(sd_acl_metric_fp, 0)
    sd_controls_metric_df = fetch_metric(sd_controls_metric_fp, 1)
    sd_athletes_metric_df = fetch_metric(sd_athletes_metric_fp, 2)
    sd_metric_df = pd.concat([sd_acl_metric_df, sd_controls_metric_df, sd_athletes_metric_df], ignore_index=True)

    sd_acl_p2d_df = fetch_p2d(sd_acl_p2d_fp, 0)
    sd_controls_p2d_df = fetch_p2d(sd_controls_p2d_fp, 1)
    sd_athletes_p2d_df = fetch_p2d(sd_athletes_p2d_fp, 2)
    sd_p2d_df = pd.concat([sd_acl_p2d_df, sd_controls_p2d_df, sd_athletes_p2d_df], ignore_index=True)

    # SDPS
    sdps_acl_metric_df = fetch_metric(sdps_acl_metric_fp, 0)
    sdps_controls_metric_df = fetch_metric(sdps_controls_metric_fp, 1)
    sdps_athletes_metric_df = fetch_metric(sdps_athletes_metric_fp, 2)
    sdps_metric_df = pd.concat([sdps_acl_metric_df, sdps_controls_metric_df, sdps_athletes_metric_df], ignore_index=True)

    sdps_acl_p2d_df = fetch_p2d(sdps_acl_p2d_fp, 0)
    sdps_controls_p2d_df = fetch_p2d(sdps_controls_p2d_fp, 1)
    sdps_athletes_p2d_df = fetch_p2d(sdps_athletes_p2d_fp, 2)
    sdps_p2d_df = pd.concat([sdps_acl_p2d_df, sdps_controls_p2d_df, sdps_athletes_p2d_df], ignore_index=True)

    # SH
    sh_acl_metric_df = fetch_metric(sh_acl_metric_fp, 0)
    sh_controls_metric_df = fetch_metric(sh_controls_metric_fp, 1)
    sh_athletes_metric_df = fetch_metric(sh_athletes_metric_fp, 2)
    sh_metric_df = pd.concat([sh_acl_metric_df, sh_controls_metric_df, sh_athletes_metric_df], ignore_index=True)

    sh_acl_p2d_df = fetch_p2d(sh_acl_p2d_fp, 0)
    sh_controls_p2d_df = fetch_p2d(sh_controls_p2d_fp, 1)
    sh_athletes_p2d_df = fetch_p2d(sh_athletes_p2d_fp, 2)
    sh_p2d_df = pd.concat([sh_acl_p2d_df, sh_controls_p2d_df, sh_athletes_p2d_df], ignore_index=True)


    # Cant assemble complete datasets for all tests since the data are from different sensors making different columns.
    #metric_df = pd.concat([olhd_metric_df, sd_metric_df, sdps_metric_df, sh_metric_df], ignore_index=True)
    #p2d_df = pd.concat([olhd_p2d_df, olvj_p2d_df, sd_p2d_df, sdps_p2d_df, sh_p2d_df], ignore_index=True)
    #print(sh_metric_df.head(1000))
    #print(olhd_metric_df.head(1000))
    #print(sd_metric_df.head(100))
    #print(sh_metric_df.head(100))
    #print(p2d_df.head(1000))
    return olhd_metric_df, olhd_p2d_df, olvj_p2d_df, sd_metric_df, sd_p2d_df, sdps_metric_df, sdps_p2d_df, sh_metric_df, sh_p2d_df

def check_metric_features(df, n_features):
    """
    This is a function which uses 2 different types of feature selection to reduce the number of columns our df
    consists of. It combines the top n_features from each test and only keeps the calculated columns. This means
    that the df will most likely consist of more than n_features columns when it's done.

    :param df: The dataframe of which we want to reduce the number of columns in.
    :param n_features: The number of top features we want from each feature search.
    :return: Returns the df with the selected features.
    """

    targets = df['Target_x'].values
    train = df.iloc[:, 0:-2]
    df_columns = df.columns
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train)
    # Score of columns.
    bestfeatures = SelectKBest(score_func=f_classif, k=n_features)
    fit = bestfeatures.fit(train_scaled, targets)
    score_df = pd.DataFrame(fit.scores_)
    columns_df = pd.DataFrame(train.columns)
    feature_scores = pd.concat([columns_df, score_df], axis=1)
    #print(targets)
    feature_scores.columns = ['category', 'Score']
    top_20 = feature_scores.nlargest(n_features, 'Score')
    highscore_features = top_20['category'].tolist()
    print(highscore_features)


    # Plot of feature importance.
    cate_targets = keras.utils.np_utils.to_categorical(targets)
    model = ExtraTreesClassifier()
    model.fit(train_scaled, cate_targets)
    feature_importance = pd.Series(model.feature_importances_, index=train.columns)
    feature_importance.nlargest(n_features).plot(kind='barh')
    important_features = feature_importance.nlargest(n_features)
    top_features = important_features.index.tolist()
    features_to_use = np.unique(highscore_features + top_features).tolist()
    features_to_use.append('Target_x')
    features_to_use.append('Patient_name')
    features_to_remove = [x for x in df_columns if x not in features_to_use]
    df.drop(features_to_remove, axis=1, inplace=True)

    return df, features_to_use

def baseline_dense_model(xtrain, target, epochs, patience):
    """
    NOT BEING USED.
    :param xtrain:
    :param target:
    :param epochs:
    :param patience:
    :return:
    """
    input_shape = xtrain.shape
    train_stop_index = int(0.7 * input_shape[0])

    train_data = xtrain[:train_stop_index, :]
    train_target = target[:train_stop_index, :]

    validation_data = xtrain[train_stop_index:, :]
    validation_target = target[train_stop_index:, :]
    print(validation_target.shape)
    print(validation_data.shape)

    lstm_test_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=2056, activation='relu'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(units=2056, activation='relu'),
        tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.001,
        patience=patience,

        mode="auto",
        restore_best_weights=True)

    lstm_test_model.compile(loss='binary_crossentropy',
                            optimizer=tf.optimizers.Adam(learning_rate=0.001),
                            metrics=['accuracy'])
    history = lstm_test_model.fit(train_data, train_target,
                                  epochs=epochs,
                                  validation_data=(validation_data, validation_target),
                                  callbacks=[callbacks],
                                  shuffle=True)

    best_accuracy = max(history.history.get('val_accuracy'))
    return best_accuracy

def baseline_model(xtrain, ytrain, xval, yval):
    """
    NOT USED ATM.
    Was a function to evaluate the models.
    :param xtrain:
    :param ytrain:
    :param xval:
    :param yval:
    :return:
    """
    lstm_test_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=False),
        # tf.keras.layers.LSTM(256, return_sequences=False),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(units=512, activation='relu'),
        # maxpooling
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='sigmoid')
    ])
    callbacks = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.001,
        patience=40,
        verbose=0,
        mode="auto",
        restore_best_weights=True)

    lstm_test_model.compile(loss='binary_crossentropy',
                            optimizer=tf.optimizers.Adam(learning_rate=0.01),
                            metrics=['accuracy'])
    history = lstm_test_model.fit(xtrain, ytrain,
                                  epochs=600,
                                  verbose=0,
                                  validation_data=(xval, yval),
                                  callbacks=[callbacks],
                                  shuffle=True)

    best_accuracy = max(history.history.get('val_accuracy'))
    return best_accuracy


def check_p2d_features(df, n_features, patience_of_search):
    """
    NOT USED WHEN TRAINING THE MODELS ATM.


    Kolla på scaling av np array direkt istället för att gå tillbaka i df och skala därifrån.
    Se även över om vi kanske borde skala metric data innan vi gör feature selection.
    Se även över hur vi kan lägga till noise för endast en kolumn itaget i numpy format.

    Kanske sluta testa den som ger störst negativ effekt på acc varje itteration också.

    IMPROVEMENTS:
    Set fixed batch size to smooth out the training time and maybe lock/remove more
    than one feature per iteration.
    Change value assigned to the locked indices since they get assigned the prev avg
    value which makes the new avg higher since it was the lowest in the
    prev iteration. Meaning the algorithm might not stop at the peak but instead
    keep removing features even though no improvement will be made.

    This function takes a dataframe and test the data on a simple LSTM network while iteratively trying to remove one
    feature. The networks with the highest accuracy means that that feature affects the models negatively and can be
    removed for a possible improvement. Besides removing one feature we also lock the feature that performed the worst
    when removed from being tested or removed in future iterations to speed up the selection. The locked indices is
    shown in as a output list "Locked indices".
    The feature selection stops when the resulting avg accuracy when a feature is removed is lower than the previous
    iterations avg.
    :param df: A dataframe that we got from fetch p2d.
    :param n_features: Minimum number of features we want the model to have.
    :param patience_of_search: The number of times each model will be tested with each feature removed. Big numbers will
    make this process expensive so keep it relatively small.
    :return:
    """
    dict_with_acurracy = collections.defaultdict(list)
    # shuffle dataframe
    df = df.sample(frac=1).reset_index()
    # Get targets and convert them from single value 0-2 to 3d vector.
    targets = df['Target_y'].values
    cate_targets = keras.utils.np_utils.to_categorical(targets)

    # Reshaping the train data to be of shape (number_of_patients, samples_from_each, number_of_sensors)
    train = df['data'].values.tolist()
    test_df = pd.DataFrame(train)
    train_array = np.array(train).astype(float)
    input_shape = train_array.shape
    train_array_min = train_array.min(axis=(0, 1), keepdims=True)
    train_array_max = train_array.max(axis=(0, 1), keepdims=True)
    scaled_train_array = (2 * (train_array - train_array_min) / (train_array_max - train_array_min)) - 1

    # Creating a loop removing one feature each iteration untill only n_features are left.
    # input_shape[2] - n_features
    n_of_iterations = 0
    improvement = True
    index_to_lock = []
    base_acc_list = []
    while improvement:
        index_to_test = 0
        array_size = scaled_train_array.shape
        if index_to_lock != None:
            if len(index_to_lock) == array_size[2]:
                improvement = False
        # Clearing the dict
        dict_with_acurracy.clear()


        # i loop range input_shape[2]
        for feature in range(array_size[2]):
            print(f'Testing index {index_to_test}')
            if index_to_lock != None:
                print(f'Locked indices: {index_to_lock}')
            if index_to_test in index_to_lock:
                index_to_test += 1
                dict_with_acurracy[index_to_test].append(avg_overall)
                continue
            train_stop_index = int(0.7 * input_shape[0])
            # Dropping one feature from the data set.

            reduced_array = np.delete(scaled_train_array, index_to_test, 2)
            # Slicing array into train and validation sets.
            reduced_train_data = reduced_array[:train_stop_index, :, :]
            print(f'Current array shape: {reduced_train_data.shape}')
            reduced_train_target = cate_targets[:train_stop_index, :]

            reduced_validation_data = reduced_array[train_stop_index:, :, :]
            reduced_validation_target = cate_targets[train_stop_index:, :]
            # Checking baseline acc. Meaning no reduction of number of columns.
            if n_of_iterations == 0 and index_to_test == 0:
                train_data = scaled_train_array[:train_stop_index, :, :]
                print(f'Current array shape: {scaled_train_array.shape}')
                train_target = cate_targets[:train_stop_index, :]

                validation_data = scaled_train_array[train_stop_index:, :, :]
                validation_target = cate_targets[train_stop_index:, :]
                print(f'RUNNING {patience_of_search} TRIAL RUNS WITH NO MODIFICATION OF ARRAY SIZE, THIS MIGHT TAKE A WHILE')
                for i in range(patience_of_search):
                    base_acc = baseline_model(train_data, train_target, validation_data, validation_target)

                    base_acc_list.append(base_acc)
                base_accuracy = sum(base_acc_list) / len(base_acc_list)
                print(f'BASE MODEL ACC: {base_accuracy}')
            # Buildning test-model to compare features.
            for i in range(patience_of_search):
                best_accuracy = baseline_model(reduced_train_data, reduced_train_target, reduced_validation_data, reduced_validation_target)
                print(f'Best accuracy: {best_accuracy}')
                if index_to_test in dict_with_acurracy:
                    dict_with_acurracy[index_to_test].append(best_accuracy)
                else:
                    dict_with_acurracy[index_to_test] = [best_accuracy]
            index_to_test += 1

        accuracy_avg = []
        for index, values in dict_with_acurracy.items():
            accuracy_avg.append(sum(values) / float(len(values)))
        print(accuracy_avg)
        # Evaluating the improvement and breaking if no improvement was made.
        if n_of_iterations != 0:

            prev_avg = avg_overall
            avg_overall = sum(accuracy_avg) / float(len(accuracy_avg))
            print(f'prev: {prev_avg} achieved this iteration: {avg_overall}')
            if prev_avg < avg_overall:
                index_to_lock.append(accuracy_avg.index(min(accuracy_avg)))
                index_to_remove = accuracy_avg.index(max(accuracy_avg))
                print(f'Removing index: {index_to_remove}')
                scaled_train_array = np.delete(scaled_train_array, index_to_remove, 2)
                # Changing indices in the list of locked indices to stay updated after the removal of index_to_remove.
                for i in range(len(index_to_lock)):
                    if index_to_lock[i] > index_to_remove:
                        index_to_lock[i] -= 1
                n_of_iterations += 1
            elif array_size == n_features:
                improvement = False
            else:
                improvement = False
        else:
            avg_overall = sum(accuracy_avg) / float(len(accuracy_avg))
            print(f'Overall acc: {avg_overall}')
            index_to_lock.append(accuracy_avg.index(min(accuracy_avg)))
            index_to_remove = accuracy_avg.index(max(accuracy_avg))
            max_acc = max(accuracy_avg)

            if index_to_remove in index_to_lock:
                print('NOT REMOVING A LOCKED INDEX!')
                break
            if max_acc < base_accuracy:
                print('NO IMPROVEMENT BY REDUCING ARRAY SIZE. KEEP ALL DATA')
                break
            else:
                print(f'Removing index: {index_to_remove}')
                scaled_train_array = np.delete(scaled_train_array, index_to_remove, 2)
                if index_to_remove < index_to_lock[0]:
                    index_to_lock[0] -= 1
                n_of_iterations += 1



    return scaled_train_array, cate_targets


def same_samples(metric_df, p2d_df):
    """
        Compares the patients from the metric and p2d data between ONE exercise so that the same patient and number of
        repetitions exists in both.
        :param metric_df: Metric df.
        :param p2d_df: p2d df.
        :return: Returns new dfs containing the same patients.
    """
    # Checking metric df.
    s = metric_df.merge(p2d_df, left_on='Patient_name', right_on='patient_name', how='left', indicator=True)
    s.dropna(inplace=True)

    new_metric_df = s.iloc[:, :-4]
    new_p2d_df = s.iloc[:, -4:-1]

    return new_metric_df, new_p2d_df


def same_patients(met_df, p_df):
    """
    Compares the patients from the metric and p2d data between the SD and SDPS data.
    :param met_df: Metric df.
    :param p_df: p2d df.
    :return: Returns new dfs containing the same patients.
    """
    met_df['patient_num'] = met_df.Patient_name.str[:3] + met_df.Patient_name.str[-2:]
    #print(sd_df.head(100))
    p_df['Patient_num'] = p_df.patient_name.str[:3] + p_df.patient_name.str[-2:]
    df = met_df.merge(p_df, left_on='patient_num', right_on='Patient_num', how='left', indicator=True)
    df.dropna(inplace=True)
    fixed_met_df = df.iloc[:, :-6]
    fixed_p_df = df.iloc[:, -5:-2]
    return fixed_met_df, fixed_p_df

"""
CHANGE folder_path TO THE ROOT OF THE DATA. The folder rå data below contains the SD AI EXPORT folder etc. See function 
build_sets. To see the expected structure of the folder containing data see paths in the function build_sets.
"""
folder_path = '/home/wubba/umeå_projekt/rå data'
# Importing datasets
olhd_metric_df, olhd_p2d_df, olvj_p2d_df, sd_metric_df, sd_p2d_df, sdps_metric_df, sdps_p2d_df, sh_metric_df, sh_p2d_df = build_sets(folder_path)

# Making sure the same samples exists in both metric and p2d dataframe.
same_sd_metric_df, same_sd_p2d_df = same_samples(sd_metric_df, sd_p2d_df)
same_sdps_metric_df, same_sdps_p2d_df = same_samples(sdps_metric_df, sdps_p2d_df)

# Making same patients and number of reps in sd and sdps data.
same_patient_sd_metric, same_patient_sdps_p2d = same_patients(same_sd_metric_df, same_sdps_p2d_df)
same_patient_sdps_metric, same_patient_sd_p2d = same_patients(same_sdps_metric_df, same_sd_p2d_df)

if not exists('sd_metric.csv'):
    same_patient_sd_metric, saved_sd_metrics = check_metric_features(same_patient_sd_metric, 30)
    same_patient_sd_metric.to_csv('sd_metric.csv', index=False)
    textfile = open('saved_sd_metrics', 'w')
    for metric in saved_sd_metrics:
        textfile.write(metric + '\n')
    textfile.close()

if not exists('sdps_metric.csv'):
    same_patient_sdps_metric, saved_sdps_metrics = check_metric_features(same_patient_sdps_metric, 30)
    same_patient_sdps_metric.to_csv('sdps_metric.csv', index=False)
    textfile = open('saved_sdps_metrics', 'w')
    for metric in saved_sdps_metrics:
        textfile.write(metric + '\n')
    textfile.close()

#sh_metric_df = check_metric_features(sh_metric_df, 20)



# Checking if we can remove any columns of the input data and saving a dataset on disk for easy access.
# This process is quite compute intensive. Improvement of check_p2d_features are possible by setting a bigger batchsize
# and perhaps lock more than one feature per iteration.
#olhd_p2d_train, olhd_p2d_cate_targets = check_p2d_features(olhd_p2d_df, 10, patience_of_search=10)
#np.save('olhd_p2d_training', olhd_p2d_train, allow_pickle=False)
#np.save('olhd_p2d_targets', olhd_p2d_cate_targets, allow_pickle=False)

#olvj_p2d_train, olvj_p2d_cate_targets = check_p2d_features(olvj_p2d_df, 10, patience_of_search=10)
#np.save('olvj_p2d_training', olvj_p2d_train, allow_pickle=False)
#np.save('olvj_p2d_targets', olvj_p2d_cate_targets, allow_pickle=False)


#sd_p2d_train, sd_p2d_cate_targets = check_p2d_features(sd_p2d_df, 10, patience_of_search=10)
#np.save('sd_p2d_training', sd_p2d_train, allow_pickle=False)
#np.save('sd_p2d_targets', sd_p2d_cate_targets, allow_pickle=False)

#sdps_p2d_train, sdps_p2d_cate_targets = check_p2d_features(sdps_p2d_df, 10, patience_of_search=10)
#np.save('sdps_p2d_training', sdps_p2d_train, allow_pickle=False)
#np.save('sdps_p2d_targets', sdps_p2d_cate_targets, allow_pickle=False)

#sh_p2d_train, sh_p2d_cate_targets = check_p2d_features(sh_p2d_df, 10, patience_of_search=10)
#np.save('sh_p2d_training', sh_p2d_train, allow_pickle=False)
#np.save('sh_p2d_targets', sh_p2d_cate_targets, allow_pickle=False)


"""
    
  
    Bygg modell där noise introduceras i för en column i taget, jämnför resultat för att se vilka som ger störst 
    negativ påverkan när de har noise. Detta betyder att denna feature har påverkan på resultat. Om ingen eller låg
    påverkan sker så kan vi ta bort den kolumnen med data. 
    

    
"""

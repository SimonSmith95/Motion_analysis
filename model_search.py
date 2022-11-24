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
# Content: Functions and main training script to search for models using data preprocessed by fetch_data.py.
# ==============================================================================
# Change history:
#    2022-05-01 (SiSm) Created
#    2022-05-26 (SiSm) Submitted to course
#    2022-05-27 (SiSm) Adding open license
# ==============================================================================


import os.path
from fetch_data import same_patient_sd_p2d as sd_df
from fetch_data import same_patient_sdps_p2d as sdps_df
import numpy as np
import pandas as pd
import keras
import sklearn.preprocessing
import tensorflow as tf
from keras_tuner import BayesianOptimization
from keras import backend as K
import random
"""
This file will use data data gathered by fetch_data so in order to run this file the fetch_data file must be executed
before to create the metric data csvs that we use here. Remember to change folder path in fetch_data to your folder
where you keep the data.

This files function is to do a bayesian optimized search of hyper parameters for all layers of the model.
It starts off with searching for models for the metric data then it searches for models handling the p2d data before 
searching for a model which uses the combined output of each of the previous models second to last layer to predict
patient category.  
"""


def preprocess_metric(df):
    """
    This function shuffles a df, separates and makes targets into categorical vector and scales the
    data into values in the range -1 to 1.
    We are using random state when shuffling the df to make sure that the test samples are the same for the p2d data
    as well. Meaning when we use a test set for our combined model none of the models will have seen that data either.
    :param df: a dataframe containing data points, targets and patient names.
    :return: returns np.array of scaled data and categorical targets.
    """
    df['patient_num'] = df.Patient_name.str[:3]
    df['num_and_rep'] = df.Patient_name.str[:3] + df.Patient_name.str[-2:]
    acl_df = df[df['Target_x'] == 0]
    con_df = df[df['Target_x'] == 1]
    ath_df = df[df['Target_x'] == 2]
    df_list = [acl_df, con_df, ath_df]
    train_patients = []
    validation_patients = []
    test_patients = []
    for dfs in df_list:
        patients = {}

        for patient in dfs['patient_num']:
            if patient not in patients:
                patients[patient] = 1
        unique_patients = [i for i in patients.keys()]
        # Sorting the list and then shuffling it to make sure the same patients are in the sets from both exercises.
        unique_patients.sort()

        # Splitting df into train, validation and test  dfs.

        train_stop = round(0.65 * len(unique_patients))
        validation_stop = round(0.85 * len(unique_patients))
        train_patients.extend(unique_patients[:train_stop])
        validation_patients.extend(unique_patients[train_stop:validation_stop])
        test_patients.extend(unique_patients[validation_stop:])

    test_patients.sort()
    validation_patients.sort()
    train_patients.sort()

    train_df = df[df['patient_num'].isin(train_patients)]
    train_df = train_df.sort_values('num_and_rep')
    train_df = train_df.sample(frac=1, random_state=1)
    validation_df = df[df['patient_num'].isin(validation_patients)]
    validation_df = validation_df.sort_values('num_and_rep')
    validation_df = validation_df.sample(frac=1, random_state=2)
    test_df = df[df['patient_num'].isin(test_patients)]
    test_df = test_df.sort_values('num_and_rep')
    test_df = test_df.sample(frac=1, random_state=3)

    train_x, train_y = extract_and_scale_metric(train_df)
    validation_x, validation_y = extract_and_scale_metric(validation_df)
    test_x, test_y = extract_and_scale_metric(test_df)
    return train_x, train_y, validation_x, validation_y, test_x, test_y


def extract_and_scale_metric(df):
    """
    Extracting the Target_x and all data points from the dataframe without the names of the patients.
    :param df: A dataframe containing the metric data.
    :return: A scaled np.array of data and one for targets.
    """
    # Seperating datapoints from targets.
    targets = df['Target_x'].values
    data = df.iloc[:, 0:-4]
    # Scaling values.
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    # Making targets categorical instead of in values 0-2.
    cate_targets = keras.utils.np_utils.to_categorical(targets)
    return data_scaled, cate_targets


def preprocess_p2d(df):
    """
    This function processes the p2d df into arrays which are ready to be fed to a model.
    :param df: dataframe with p2d data in the format given form fetch_data.py
    :return: returns all sets for training.
    """
    # Creating a list with all the patient numbers
    df['patient_num'] = df.patient_name.str[:3]
    df['num_and_rep'] = df.patient_name.str[:3] + df.patient_name.str[-2:]
    acl_df = df[df['Target_y'] == 0]
    con_df = df[df['Target_y'] == 1]
    ath_df = df[df['Target_y'] == 2]
    df_list = [acl_df, con_df, ath_df]
    train_patients = []
    validation_patients = []
    test_patients = []
    for dfs in df_list:
        patients = {}

        for patient in dfs['patient_num']:
            if patient not in patients:
                patients[patient] = 1
        unique_patients = [i for i in patients.keys()]
        # Sorting the list and then shuffling it to make sure the same patients are in the sets from both exercises.
        unique_patients.sort()

        # Splitting df into train, validation and test  dfs.

        train_stop = round(0.65 * len(unique_patients))
        validation_stop = round(0.85 * len(unique_patients))
        train_patients.extend(unique_patients[:train_stop])
        validation_patients.extend(unique_patients[train_stop:validation_stop])
        test_patients.extend(unique_patients[validation_stop:])

    test_patients.sort()
    validation_patients.sort()
    train_patients.sort()

    train_df = df[df['patient_num'].isin(train_patients)]
    train_df = train_df.sort_values('num_and_rep')
    train_df = train_df.sample(frac=1, random_state=1)
    validation_df = df[df['patient_num'].isin(validation_patients)]
    validation_df = validation_df.sort_values('num_and_rep')
    validation_df = validation_df.sample(frac=1, random_state=2)
    test_df = df[df['patient_num'].isin(test_patients)]
    test_df = test_df.sort_values('num_and_rep')
    test_df = test_df.sample(frac=1, random_state=3)

    # Making df into a scaled np.array.
    p2d_train_x, p2d_train_y = extract_and_scale_p2d(train_df)
    p2d_validation_x, p2d_validation_y = extract_and_scale_p2d(validation_df)
    p2d_test_x, p2d_test_y = extract_and_scale_p2d(test_df)

    return p2d_train_x, p2d_train_y, p2d_validation_x, p2d_validation_y, p2d_test_x, p2d_test_y

def extract_and_scale_p2d(df):
    """
       Extracting the Target_x and all data points from the dataframe without the names of the patients.
       :param df: A dataframe containing the p2d data.
       :return: A scaled np.array of data and one for targets.
    """
    targets = df['Target_y'].values
    cate_targets = keras.utils.np_utils.to_categorical(targets)
    train = df['data'].values.tolist()
    train_array = np.array(train).astype(float)
    # Scaling the data.
    train_array_min = train_array.min(axis=(0, 1), keepdims=True)
    train_array_max = train_array.max(axis=(0, 1), keepdims=True)
    scaled_train_array = (2 * (train_array - train_array_min) / (train_array_max - train_array_min)) - 1
    return scaled_train_array, cate_targets

def preprocess_concat_data(metric_csv_file_list, sd_p2d_df, sdps_p2d_df, metric_models, p2d_models):
    """
    This function takes the data and runs it through the pre-trained models and extracts outputs from the models second
    to last layer and concatenates p2d and metric data into one set.

    :param metric_csv_file_list: A list with names of the metric data csvs. Same as for the function preprocess_metric(df)
    :param sd_p2d_df: Same df that is sent to preprocess_p2d(df)
    :param sdps_p2d_df: Same df that is sent to preprocess_p2d(df)
    :return: Returns lists containing the training, validation, test data and list with respective targets.
    """
    # Loading data.
    sd_metric_df = pd.read_csv(metric_csv_file_list[0])
    sd_metric_train_x, sd_metric_train_y, sd_metric_validation_x, sd_metric_validation_y, sd_metric_test_x, sd_metric_test_y = preprocess_metric(sd_metric_df)
    sdps_metric_df = pd.read_csv(metric_csv_file_list[1])
    sdps_metric_train_x, sdps_metric_train_y, sdps_metric_validation_x, sdps_metric_validation_y, sdps_metric_test_x, sdps_metric_test_y = preprocess_metric(
        sdps_metric_df)
    sd_p2d_train_x, sd_p2d_train_y, sd_p2d_validation_x, sd_p2d_validation_y, sd_p2d_test_x, sd_p2d_test_y = preprocess_p2d(sd_p2d_df)
    sdps_p2d_train_x, sdps_p2d_train_y, sdps_p2d_validation_x, sdps_p2d_validation_y, sdps_p2d_test_x, sdps_p2d_test_y = preprocess_p2d(
        sdps_p2d_df)

    # Making lists with data. We only need one set of targets from either metric or p2d data since they are ordered the same way.
    sd_metric_data = [sd_metric_train_x, sd_metric_validation_x, sd_metric_test_x]
    sd_targets = [sd_metric_train_y,  sd_metric_validation_y, sd_metric_test_y]
    sd_p2d_data = [sd_p2d_train_x, sd_p2d_validation_x, sd_p2d_test_x]
    sdps_metric_data = [sdps_metric_train_x, sdps_metric_validation_x, sdps_metric_test_x]
    sdps_targets = [sdps_metric_train_y, sdps_metric_validation_y, sdps_metric_test_y]
    sdps_p2d_data = [sdps_p2d_train_x, sdps_p2d_validation_x, sdps_p2d_test_x]
    # Loading pretrained models.
    sd_dense_model = keras.models.load_model(metric_models[0])
    sdps_dense_model = keras.models.load_model(metric_models[1])
    sd_p2d_model = keras.models.load_model(p2d_models[0])
    sdps_p2d_model = keras.models.load_model(p2d_models[1])

    # Setting up intermediate models to extract outputs.
    sd_dense_layer_list = [layer.name for layer in sd_dense_model.layers]
    sd_dense_inter_model = keras.Model(inputs=sd_dense_model.input,
                                       outputs=sd_dense_model.get_layer(sd_dense_layer_list[-2]).output)

    sdps_dense_layer_list = [layer.name for layer in sdps_dense_model.layers]
    sdps_dense_inter_model = keras.Model(inputs=sdps_dense_model.input,
                                         outputs=sdps_dense_model.get_layer(sdps_dense_layer_list[-2]).output)

    sd_p2d_layer_list = [layer.name for layer in sd_p2d_model.layers]
    sd_p2d_inter_model = keras.Model(inputs=sd_p2d_model.input,
                                     outputs=sd_p2d_model.get_layer(sd_p2d_layer_list[-2]).output)

    sdps_p2d_layer_list = [layer.name for layer in sdps_p2d_model.layers]
    sdps_p2d_inter_model = keras.Model(inputs=sdps_p2d_model.input,
                                       outputs=sdps_p2d_model.get_layer(sdps_p2d_layer_list[-2]).output)


    # Extracting data
    sd_data_sets = []
    sdps_data_sets = []
    for set_number in range(len(sd_metric_data)):
        # SD
        sd_dense_inter_output = sd_dense_inter_model.predict(sd_metric_data[set_number])
        sd_p2d_inter_output = sd_p2d_inter_model.predict(sd_p2d_data[set_number])
        sd_partial_data = np.concatenate((sd_dense_inter_output, sd_p2d_inter_output), axis=1)
        sd_data_sets.append(sd_partial_data)
        # SDPS
        sdps_dense_inter_output = sdps_dense_inter_model.predict(sdps_metric_data[set_number])
        sdps_p2d_inter_output = sdps_p2d_inter_model.predict(sdps_p2d_data[set_number])
        sdps_partial_data = np.concatenate((sdps_dense_inter_output, sdps_p2d_inter_output), axis=1)
        sdps_data_sets.append(sdps_partial_data)

    return sd_data_sets, sd_targets, sdps_data_sets, sdps_targets


def build_dense_model(hp):
    """
    Config of the search space for the dense models.
    :param hp: Param values to build a model from.
    :return: returns a compiled model.
    """
    component_list = []
    for layer_number in range(hp.Int('num_layers_block_1', min_value=1, max_value=4)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_1_layer_{layer_number}_units',
                                                           min_value=74, max_value=2368, step=74), activation='relu'))
        if layer_number == 2 or layer_number == 4:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_1_layer_{layer_number}_dropout',
                                                                  min_value=0.1, max_value=0.5, step=0.1)))

    for layer_number in range(hp.Int('num_layers_block_2', min_value=1, max_value=6)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_2_layer_{layer_number}_units',
                                                           min_value=128, max_value=4096, step=128), activation='relu'))
        if layer_number == 2 or layer_number == 4 or layer_number == 6:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_2_layer_{layer_number}_dropout',
                                                                   min_value=0.1, max_value=0.5, step=0.1)))
    for layer_number in range(hp.Int('num_layers_block_3', min_value=1, max_value=4)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_3_layer_{layer_number}_units',
                                                           min_value=192, max_value=3072, step=192), activation='relu'))
        if layer_number == 2 or layer_number == 4:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_3_layer_{layer_number}_dropout',
                                                                   min_value=0.05, max_value=0.5, step=0.05)))
    output_layer = tf.keras.layers.Dense(3, activation='softmax')
    component_list.append(output_layer)
    model = tf.keras.models.Sequential(component_list)
    lr = hp.Choice('lr_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_concat_model(hp):
    """
      Config of the search space for the concatenated models.
      :param hp: Param values to build a model from.
      :return: returns a compiled model.
    """

    component_list = []
    for layer_number in range(hp.Int('num_layers_block_1', min_value=1, max_value=4)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_1_layer_{layer_number}_units',
                                                           min_value=74, max_value=4096, step=74), activation='relu'))
        if layer_number == 2 or layer_number == 4:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_1_layer_{layer_number}_dropout',
                                                                   min_value=0.05, max_value=0.5, step=0.05)))

    for layer_number in range(hp.Int('num_layers_block_2', min_value=1, max_value=4)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_2_layer_{layer_number}_units',
                                                           min_value=128, max_value=4096, step=128), activation='relu'))
        if layer_number == 2 or layer_number == 4:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_2_layer_{layer_number}_dropout',
                                                                   min_value=0.05, max_value=0.5, step=0.05)))
    for layer_number in range(hp.Int('num_layers_block_3', min_value=1, max_value=4)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_3_layer_{layer_number}_units',
                                                           min_value=81, max_value=2187, step=243), activation='relu'))
        if layer_number == 2 or layer_number == 4:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_3_layer_{layer_number}_dropout',
                                                                   min_value=0.05, max_value=0.5, step=0.05)))
    output_layer = tf.keras.layers.Dense(3, activation='softmax')
    component_list.append(output_layer)
    model = tf.keras.models.Sequential(component_list)
    lr = hp.Choice('lr_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_p2d_model(hp):
    """
      Config of the search-space for the LSTM models.
      :param hp: Param values to build a model from.
      :return: returns a compiled model.
    """
    component_list = []
    for layer_number in range(hp.Int('num_layers_block_1', min_value=1, max_value=2)):
        component_list.append(tf.keras.layers.LSTM(hp.Int(f'block_1_layer_{layer_number}_units',
                                                          min_value=64, max_value=512, step=64), return_sequences=True, activation='relu'))

    component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_1_layer_{layer_number}_dropout',
                                                           min_value=0.1, max_value=0.5, step=0.1)))

    for layer_number in range(hp.Int('num_layers_block_2', min_value=0, max_value=1)):
        component_list.append(tf.keras.layers.LSTM(hp.Int(f'block_2_layer_{layer_number}_units',
                                                          min_value=128, max_value=512, step=128), return_sequences=True, activation='relu'))
        if layer_number > 0:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_2_layer_{layer_number}_dropout',
                                                                   min_value=0.1, max_value=0.5, step=0.1)))
    component_list.append(tf.keras.layers.Flatten())
    for layer_number in range(hp.Int('num_layers_block_3', min_value=1, max_value=3)):
        component_list.append(tf.keras.layers.Dense(hp.Int(f'block_3_layer_{layer_number}_units',
                                                           min_value=192, max_value=2048, step=192), activation='relu'))
        if layer_number == 2:
            component_list.append(tf.keras.layers.Dropout(hp.Float(f'block_3_layer_{layer_number}_dropout',
                                                                   min_value=0.05, max_value=0.5, step=0.05)))
    output_layer = tf.keras.layers.Dense(3, activation='softmax')
    component_list.append(output_layer)
    model = tf.keras.models.Sequential(component_list)
    lr = hp.Choice('lr_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def p2d_model_search(train_x, train_y, validation_x, validation_y, test_x, test_y, search_version, model_name):
    """
    Function that runs a search for number of layers and neurons in each layer for the p2d lstm model.
    :param train_x: The training data.
    :param train_y: Training targets.
    :param validation_x: Validation data.
    :param validation_y: Validation targets
    :param test_x: Test data.
    :param test_y: Test targets.
    :param search_version: Name of the search version.
    :param model_name: Name of the model to create.
    :return:
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                     patience=10, min_lr=1e-7)
    CALLBACKS = [earlystop, reduce_lr]
    if True:
        tuner = BayesianOptimization(
            build_p2d_model,
            objective="val_accuracy",
            max_trials=50,
            executions_per_trial=6,
            directory='LSTM_search_output',
            project_name=search_version)

        tuner.search(
            x=train_x, y=train_y,
            validation_data=(validation_x, validation_y),
            batch_size=32,
            callbacks=CALLBACKS,
            epochs=400,
            shuffle=True)
    # Getting best models found.
    bestmodels = tuner.get_best_models(num_models=4)
    best_model = bestmodels[0]
    second_best = bestmodels[1]
    third_best = bestmodels[2]
    forth_best = bestmodels[3]
    print(f'Test of {model_name}')
    # Testing the models on the test set.
    acc = best_model.evaluate(test_x, test_y)
    acc1 = second_best.evaluate(test_x, test_y)
    acc2 = third_best.evaluate(test_x, test_y)
    acc3 = forth_best.evaluate(test_x, test_y)
    # Comparing results and choosing to save the model with the lowest loss if multiple models with the same acc exists.
    list_with_results = [acc, acc1, acc2, acc3]
    list_with_accuracys = [model_acc[1] for model_acc in list_with_results]
    list_with_loss = [model_loss[0] for model_loss in list_with_results]
    top_accs = []
    top_loss = []
    for n in range(len(list_with_results)):
        if list_with_accuracys[n] == max(list_with_accuracys):
            top_accs.append(n)
            top_loss.append(list_with_loss[n])
    print(top_accs)
    print(top_loss)
    lowest_loss_index = top_loss.index(min(top_loss))
    print(f'lowest index = {lowest_loss_index}')
    best_resulting_model = top_accs[lowest_loss_index]
    if best_resulting_model == 0:
        print('saving model 0')
        best_model.save(model_name)
    elif best_resulting_model == 1:
        print('saving model 1')
        second_best.save(model_name)
    elif best_resulting_model == 2:
        print('saving model 2')
        third_best.save(model_name)
    else:
        print('saving model 3')
        forth_best.save(model_name)

    print(list_with_results)


def metric_model_search(train_x, train_y, validation_x, validation_y, test_x, test_y, search_version, model_name):
    """
       Function that runs a search for number of layers and neurons in each layer for the metric dense model.
       :param train_x: The training data.
       :param train_y: Training targets.
       :param validation_x: Validation data.
       :param validation_y: Validation targets
       :param test_x: Test data.
       :param test_y: Test targets.
       :param search_version: Name of the search version.
       :param model_name: Name of the model to create.
       :return:
    """

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                     patience=100, min_lr=1e-7)
    CALLBACKS = [earlystop, reduce_lr]
    if True:
        tuner = BayesianOptimization(
            build_dense_model,
            objective="val_accuracy",
            max_trials=60,
            executions_per_trial=8,
            directory='Dense_search_output',
            project_name=search_version)

        tuner.search(
            x=train_x, y=train_y,
            validation_data=(validation_x, validation_y),
            batch_size=32,
            callbacks=CALLBACKS,
            epochs=2000,
            shuffle=True)
    # Getting best models found.
    bestmodels = tuner.get_best_models(num_models=4)
    best_model = bestmodels[0]
    second_best = bestmodels[1]
    third_best = bestmodels[2]
    forth_best = bestmodels[3]
    print(f'Test of {model_name}')
    # Testing the models on the test set.
    acc = best_model.evaluate(test_x, test_y)
    acc1 = second_best.evaluate(test_x, test_y)
    acc2 = third_best.evaluate(test_x, test_y)
    acc3 = forth_best.evaluate(test_x, test_y)
    # Comparing results and choosing to save the model with the lowest loss if multiple models with the same acc exists.
    list_with_results = [acc, acc1, acc2, acc3]
    list_with_accuracys = [model_acc[1] for model_acc in list_with_results]
    list_with_loss = [model_loss[0] for model_loss in list_with_results]
    top_accs = []
    top_loss = []
    for n in range(len(list_with_results)):
        if list_with_accuracys[n] == max(list_with_accuracys):
            top_accs.append(n)
            top_loss.append(list_with_loss[n])
    print(top_accs)
    print(top_loss)
    lowest_loss_index = top_loss.index(min(top_loss))
    print(f'lowest index = {lowest_loss_index}')
    best_resulting_model = top_accs[lowest_loss_index]
    if best_resulting_model == 0:
        print('saving model 0')
        best_model.save(model_name)
    elif best_resulting_model == 1:
        print('saving model 1')
        second_best.save(model_name)
    elif best_resulting_model == 2:
        print('saving model 2')
        third_best.save(model_name)
    else:
        print('saving model 3')
        forth_best.save(model_name)

    print(list_with_results)


def final_data_sets(sd_data_sets, sdps_data_sets, opt_models):
    """
    Function that uses the created opt_models to merge outputs into new sets.
    :param sd_data_sets: A list containing [train, validation, test] data in that order.
    :param sdps_data_sets: A list containing [train, validation, test] data in that order.
    :param opt_models: List of the model names for the SD and SDPS data.
    :return: Returns new sets of outputs from the models.
    """
    sd_concat_model = keras.models.load_model(opt_models[0])
    sdps_concat_model = keras.models.load_model(opt_models[1])

    # Setting up intermediate models to extract outputs.
    sd_dense_layer_list = [layer.name for layer in sd_concat_model.layers]
    sd_inter_model = keras.Model(inputs=sd_concat_model.input,
                                 outputs=sd_concat_model.get_layer(sd_dense_layer_list[-2]).output)

    sdps_dense_layer_list = [layer.name for layer in sdps_concat_model.layers]
    sdps_inter_model = keras.Model(inputs=sdps_concat_model.input,
                                   outputs=sdps_concat_model.get_layer(sdps_dense_layer_list[-2]).output)

    # Extracting data
    final_data_sets = []

    for set_number in range(len(sd_data_sets)):
        sd_inter_output = sd_inter_model.predict(sd_data_sets[set_number])
        sdps_inter_output = sdps_inter_model.predict(sdps_data_sets[set_number])
        sd_partial_data = np.concatenate((sd_inter_output, sdps_inter_output), axis=1)
        final_data_sets.append(sd_partial_data)


    return final_data_sets


def concat_model_search(data_sets, targets, search_version, model_name):
    """
    The search for models combining data from previous models. Saving the best performing model for future use.
    :param data_sets: A list containing [train, validation, test] data in that order.
    :param targets: A list containing [train, validation, test] targets in that order.
    :param search_version: Name of the search version.
    :param model_name: Name of final model search.
    :return:
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                     patience=100, min_lr=5e-8)
    CALLBACKS = [earlystop, reduce_lr]
    if True:
        tuner = BayesianOptimization(
            build_concat_model,
            objective="val_accuracy",
            max_trials=80,
            executions_per_trial=8,
            directory='Concat_search_output',
            project_name=search_version)

        tuner.search(
            x=data_sets[0], y=targets[0],
            validation_data=(data_sets[1], targets[1]),
            batch_size=32,
            callbacks=CALLBACKS,
            epochs=1500,
            shuffle=True)
    # Getting best models found.
    bestmodels = tuner.get_best_models(num_models=4)
    best_model = bestmodels[0]
    second_best = bestmodels[1]
    third_best = bestmodels[2]
    forth_best = bestmodels[3]
    print(f'Test of {model_name}')
    # Testing the models on the test set.
    acc = best_model.evaluate(data_sets[2], targets[2])
    acc1 = second_best.evaluate(data_sets[2], targets[2])
    acc2 = third_best.evaluate(data_sets[2], targets[2])
    acc3 = forth_best.evaluate(data_sets[2], targets[2])
    # Comparing results and choosing to save the model with the lowest loss if multiple models with the same acc exists.
    list_with_results = [acc, acc1, acc2, acc3]
    list_with_accuracys = [model_acc[1] for model_acc in list_with_results]
    list_with_loss = [model_loss[0] for model_loss in list_with_results]
    top_accs = []
    top_loss = []
    for n in range(len(list_with_results)):
        if list_with_accuracys[n] == max(list_with_accuracys):
            top_accs.append(n)
            top_loss.append(list_with_loss[n])
    print(top_accs)
    print(top_loss)
    lowest_loss_index = top_loss.index(min(top_loss))
    print(f'lowest index = {lowest_loss_index}')
    best_resulting_model = top_accs[lowest_loss_index]
    if best_resulting_model == 0:
        print('saving model 0')
        best_model.save(model_name)
    elif best_resulting_model == 1:
        print('saving model 1')
        second_best.save(model_name)
    elif best_resulting_model == 2:
        print('saving model 2')
        third_best.save(model_name)
    else:
        print('saving model 3')
        forth_best.save(model_name)

    print(list_with_results)


def model_selection(model, x_data, target_data, num_trials, batch_sz, model_name):
    """
    Retrains the model num_trails times and selects the best one found and saves it.
    :param model: Prev found model.
    :param x_data: A list containing [train, validation, test] data in that order.
    :param target_data: A list containing [train, validation, test] targets in that order.
    :param num_trials: Number of times we will reinitialize the model.
    :param batch_sz: Batch size to use when training.
    :param model_name: Name of the model to create.
    :return:
    """
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=160, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                                                     patience=25, min_lr=5e-7)
    CALLBACKS = [earlystop, reduce_lr]
    models = []
    for model_try in range(num_trials):
        print(f'Model_{model_try}')
        original_weights = model.get_weights()
        model_clone = tf.keras.models.clone_model(model)
        new_weights = model_clone.get_weights()
        lr = 1e-3
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model_clone.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', 'MeanSquaredError'])

        model_clone.fit(x=x_data[0], y=target_data[0],
                        batch_size=batch_sz,
                        epochs=2000,
                        callbacks=CALLBACKS,
                        validation_data=(x_data[1], target_data[1]),
                        shuffle=True,
                        )
        models.append(model_clone)
    accuracy = []
    loss = []
    for modelnr in models:
        result = modelnr.evaluate(x=x_data[2], y=target_data[2])
        accuracy.append(result[1])
        loss.append(result[0])
    print(accuracy)
    print('Motion model evaluation:')
    model.evaluate(x_data[2], target_data[2])
    top_accs = []
    top_loss = []
    for n in range(len(accuracy)):
        if accuracy[n] == max(accuracy):
            top_accs.append(n)
            top_loss.append(loss[n])
    print(top_accs)
    print(top_loss)
    lowest_loss_index = top_loss.index(min(top_loss))
    print(f'lowest index = {lowest_loss_index}')
    best_resulting_model = top_accs[lowest_loss_index]
    print(f'Model to save is model_nr: {best_resulting_model}')
    models[best_resulting_model].save(model_name)


def model_details(model, x_data, target_data):
    summary = model.summary()
    stats = model.evaluate(x_data[2], target_data[2])
    return stats, summary


# Setting up lists containing the information needed for all functions.
csv_names = ['sd_metric.csv', 'sdps_metric.csv']
metric_search_version_name = ['sd_dense_search_0_0', 'sdps_dense_search_0_0']
metric_model_names = ['sd_dense_layer_0_model_1', 'sdps_dense_layer_0_model_1']
p2d_search_version_name = ['sd_lstm_search_0_0', 'sdps_lstm_search_0_0']
p2d_model_names = ['sd_lstm_layer_0_model_1', 'sdps_lstm_layer_0_model_1']
combined_search_version_name = ['sd_concat_dense_search_0_0', 'sdps_concat_dense_search_0_0']
combined_model_names = ['sd_concat_model_0_1', 'sdps_concat_model_0_1']
optimized_model_names = ['sd_concat_opt_model_0', 'sdps_concat_opt_model_0', 'opt_motion_model_1']
last_model_search_version = 'motion_model_search_0_3'
last_model_name = 'motion_model_0_3'
# Metric models.
for i in range(len(csv_names)):
    # Loading data.
    df = pd.read_csv(csv_names[i])
    train_x, train_y, validation_x, validation_y, test_x, test_y = preprocess_metric(df)
    # Checking if models with names specified already have been built or else it searches for models.
    if not os.path.isdir(metric_model_names[i]):
        metric_model_search(train_x, train_y, validation_x, validation_y,
                            test_x, test_y, metric_search_version_name[i], metric_model_names[i])
# P2d models.
for num_models in range(len(p2d_search_version_name)):
    # Checking if p2d models are built.
    if not os.path.isdir(p2d_model_names[num_models]):
        # Getting the specified df imported from fetch_data and doing a model search.
        if num_models == 0:
            p2d_train_x, p2d_train_y, p2d_validation_x, p2d_validation_y, p2d_test_x, p2d_test_y = preprocess_p2d(sd_df)
            p2d_model_search(p2d_train_x, p2d_train_y, p2d_validation_x, p2d_validation_y, p2d_test_x, p2d_test_y,
                             p2d_search_version_name[num_models], p2d_model_names[num_models])
        else:
            p2d_train_x, p2d_train_y, p2d_validation_x, p2d_validation_y, p2d_test_x, p2d_test_y = preprocess_p2d(sdps_df)
            p2d_model_search(p2d_train_x, p2d_train_y, p2d_validation_x, p2d_validation_y, p2d_test_x, p2d_test_y,
                             p2d_search_version_name[num_models], p2d_model_names[num_models])

# Combined models.
sd_data_sets, sd_targets, sdps_data_sets, sdps_targets = preprocess_concat_data(csv_names, sd_df, sdps_df, metric_model_names, p2d_model_names)
for model_num in range(len(combined_model_names)):
    if not os.path.isdir(optimized_model_names[model_num]):
        if model_num == 0:
            concat_model_search(sd_data_sets, sd_targets, combined_search_version_name[model_num], combined_model_names[model_num])
            # Optimize model.
            sd_model = keras.models.load_model(combined_model_names[0])
            model_selection(sd_model, sd_data_sets, sd_targets, 60, 64, optimized_model_names[0])


        else:
            concat_model_search(sdps_data_sets, sdps_targets, combined_search_version_name[model_num], combined_model_names[model_num])
            sdps_model = keras.models.load_model(combined_model_names[1])
            model_selection(sdps_model, sdps_data_sets, sdps_targets, 60, 64, optimized_model_names[1])
# Creating model that uses both sd and sdps data to make a prediction by combining the concat models for both data sets.
final_sets = final_data_sets(sd_data_sets, sdps_data_sets, optimized_model_names)
if not os.path.isdir(optimized_model_names[2]):

    concat_model_search(final_sets, sd_targets, last_model_search_version, last_model_name)
    motion_model = keras.models.load_model(last_model_name)
    model_selection(motion_model, final_sets, sd_targets, 100, 64, optimized_model_names[2])

# Evaluating progress of models.
sd_metric = keras.models.load_model(metric_model_names[0])
sd_p2d = keras.models.load_model(p2d_model_names[0])
sdps_metric = keras.models.load_model(metric_model_names[1])
sdps_p2d = keras.models.load_model(p2d_model_names[1])

sd_metric_train_x, sd_metric_train_y, sd_metric_validation_x, sd_metric_validation_y, sd_metric_test_x, sd_metric_test_y = preprocess_metric(pd.read_csv(csv_names[0]))
sd_metric_data = [sd_metric_train_x, sd_metric_validation_x, sd_metric_test_x]
sd_metric_targets = [sd_metric_train_y, sd_metric_validation_y, sd_metric_test_y]
sdps_metric_train_x, sdps_metric_train_y, sdps_metric_validation_x, sdps_metric_validation_y, sdps_metric_test_x, sdps_metric_test_y = preprocess_metric(pd.read_csv(csv_names[1]))
sdps_metric_data = [sdps_metric_train_x, sdps_metric_validation_x, sdps_metric_test_x]
sdps_metric_targets = [sdps_metric_train_y, sdps_metric_validation_y, sdps_metric_test_y]


sd_p2d_train_x, sd_p2d_train_y, sd_p2d_validation_x, sd_p2d_validation_y, sd_p2d_test_x, sd_p2d_test_y = preprocess_p2d(sd_df)
sd_p2d_data = [sd_p2d_train_x, sd_p2d_validation_x, sd_p2d_test_x]
sd_p2d_targets = [sd_p2d_train_y, sd_p2d_validation_y, sd_p2d_test_y]
sdps_p2d_data = preprocess_p2d(sdps_df)
sdps_p2d_train_x, sdps_p2d_train_y, sdps_p2d_validation_x, sdps_p2d_validation_y, sdps_p2d_test_x, sdps_p2d_test_y = preprocess_p2d(sdps_df)
sdps_p2d_data = [sdps_p2d_train_x, sdps_p2d_validation_x, sdps_p2d_test_x]
sdps_p2d_targets = [sdps_p2d_train_y, sdps_p2d_validation_y, sdps_p2d_test_y]

model_names = optimized_model_names.append(last_model_name)
opt_model = keras.models.load_model(optimized_model_names[2])
best_search_model = keras.models.load_model(last_model_name)
sd_opt_model = keras.models.load_model(optimized_model_names[0])
sdps_opt_model = keras.models.load_model(optimized_model_names[1])

sd_metric_acc, sd_metric_summary = model_details(sd_metric, sd_metric_data, sd_metric_targets)
sdps_metric_acc, sdps_metric_summary = model_details(sdps_metric, sdps_metric_data, sdps_metric_targets)
sd_p2d_acc, sd_p2d_summary = model_details(sd_p2d, sd_p2d_data, sd_p2d_targets)
sdps_p2d_acc, sdps_p2d_summary = model_details(sdps_p2d, sdps_p2d_data, sdps_p2d_targets)
sdps_acc, sdps_summary = model_details(sdps_opt_model, sdps_data_sets, sdps_targets)
sd_acc, sd_summary = model_details(sd_opt_model, sd_data_sets, sd_targets)
bs_model_acc, bs_model_summary = model_details(best_search_model, final_sets, sd_targets)
opt_model_acc, opt_model_summary = model_details(opt_model, final_sets, sd_targets)

acc_list = [sd_metric_acc, sdps_metric_acc, sd_p2d_acc, sdps_p2d_acc, sd_acc[1], sdps_acc[1], bs_model_acc[1], opt_model_acc[1]]



targ_0_count = 0
targ_1_count = 0
targ_2_count = 0
for target in sd_targets[2]:
    if target[0] == 1:
        targ_0_count += 1
    elif target[1] == 1:
        targ_1_count += 1
    elif target[2] == 1:
        targ_2_count += 1
    else:
        print('something wrong with target count.')
target_counts = [targ_0_count, targ_1_count, targ_2_count]
print(f'Target counts of the test set is: {target_counts}')
print(f'Total number of samples in the test set: {len(sd_targets[2])}')
print(f'Model accuracies: {acc_list}')




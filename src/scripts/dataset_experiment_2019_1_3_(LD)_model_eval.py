'''
THIS SCRIPT IS TO FEED THE LCP RECOGNITION MODEL WITH RAW AE DATA, AND SEE WHETHER IT WILL RECORGNIZE IT WELL. MODEL
WILL SLIDE THROUGH ONE 5M POINT AE RAW DATA, USING A WINDOW, WITH A STRIDE.
DATASET: BINARY CLASSIFICATION MODEL LNL (SEEN & UNSEEN leak and no leak)
'''
import sys
import os
sys.path.append('C:/Users/YH/PycharmProjects/AE-signal-model')

from scipy.signal import decimate
import argparse
import gc
import time
import tensorflow as tf
from src.utils.helpers import *

# -------------------------------------------------------------------------------------------------------------ARG PARSE
parser = argparse.ArgumentParser(description='Input some parameters.')
parser.add_argument('--model', default=None, type=str, help='model name to test')
parser.add_argument('--dsf', default=None, type=int, help='Downsample factor')

args = parser.parse_args()

MODEL_NAME_TO_TEST = args.model
DOWNSAMPLE_FACTOR = args.dsf

MODEL_INPUT_LEN = 2000
TEST_UNSEEN_TDMS_FOLDER = 'G:/Experiment_3_1_2019/-3,-2,0,5,7,16,17/1.5 bar/Leak/Test data/'
TEST_SEEN_TDMS_FOLDER = 'G:/Experiment_3_1_2019/-4,-2,2,4,6,8,10/1.5 bar/Leak/Test data/'
RESULT_SAVE_FILENAME = 'C:/Users/YH/PycharmProjects/AE-signal-model/result/{}_result.txt'.format(MODEL_NAME_TO_TEST)
DIR_TO_SAVE_PRED_CSV_SEEN = 'G:/Experiment_3_1_2019/LD_model_Evaluation_Result/{} evaluate with Experiment_2018_1_3_SEEN_Test data/'.format(MODEL_NAME_TO_TEST)
DIR_TO_SAVE_PRED_CSV_UNSEEN = 'G:/Experiment_3_1_2019/LD_model_Evaluation_Result/{} evaluate with Experiment_2018_1_3_UNSEEN_Test data/'.format(MODEL_NAME_TO_TEST)
DIR_TO_SAVE_CM_SEEN = 'C:/Users/YH/Desktop/hooyuheng.master/MASTER_PAPERWORK/My Practical Work------------/Preprocessed Dataset recognition result/LD model (dataset Dec)/{} evaluate with Jan 2019 SEEN Test data/'.format(MODEL_NAME_TO_TEST)
DIR_TO_SAVE_CM_UNSEEN = 'C:/Users/YH/Desktop/hooyuheng.master/MASTER_PAPERWORK/My Practical Work------------/Preprocessed Dataset recognition result/LD model (dataset Dec)/{} evaluate with Jan 2019 UNSEEN Test data/'.format(MODEL_NAME_TO_TEST)
MODEL_POSSIBLE_LABEL = (0, 1, 2, 3, 4, 5)
ACTUAL_LABEL_ALL_CH_SEEN = (2, 1, 1, 2, 3, 4, 5)
# !! the unseen cases labels falls btw trained model outputs, e.g. for 5m, model pred of 4m or 6m also counted as true.
ACTUAL_LABEL_ALL_CH_UNSEEN = ((1, 2), 1, 0, (2, 3), (3, 4))
SEEN_DATA_LABELS = ['sensor@[-4m]',
                    'sensor@[-2m]',
                    'sensor@[2m]',
                    'sensor@[4m]',
                    'sensor@[6m]',
                    'sensor@[8m]',
                    'sensor@[10m]']
UNSEEN_DATA_LABELS = ['sensor@[-3m]',
                      'sensor@[-2m]',
                      'sensor@[0m]',
                      'sensor@[5m]',
                      'sensor@[7m]']
MODEL_POSSIBLE_OUTPUT = ['Leak @ [0m]',
                         'Leak @ [2m]',
                         'Leak @ [4m]',
                         'Leak @ [6m]',
                         'Leak @ [8m]',
                         'Leak @ [10m]']


# instruct GPU to allocate only sufficient memory for this script
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# SLIDING WINDOW CONFIG
window_stride = 10
window_size = (1000, MODEL_INPUT_LEN - 1000)
sample_size_for_prediction = 10000

# saving naming
model_name = MODEL_NAME_TO_TEST
lcp_model = load_model(model_name=model_name)
lcp_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
print(lcp_model.summary())

# file reading
all_tdms_seen = [(TEST_SEEN_TDMS_FOLDER + f) for f in listdir(TEST_SEEN_TDMS_FOLDER) if f.endswith('.tdms')]
print(all_tdms_seen)
all_tdms_unseen = [(TEST_UNSEEN_TDMS_FOLDER + f) for f in listdir(TEST_UNSEEN_TDMS_FOLDER) if f.endswith('.tdms')]

# ------------------------------------------------------------------------------------------------- EVALUATING SEEN DATA
acc_per_ch_al_tdms = []
for file_to_test in all_tdms_seen:
    time_per_file_start = time.time()
    print('DEBUG: ', file_to_test)
    x = file_to_test.split(sep='/')[-1]
    # discard the .tdms
    x = x.split(sep='.')[-2]

    filename_to_save = 'pred_result_[{}]_[{}]_{}'.format(model_name, x, 'SEEN')

    # SAVING CONFIG
    df_pred_save_filename = DIR_TO_SAVE_PRED_CSV_SEEN + filename_to_save + '.csv'

    # test for near
    n_channel_data = read_single_tdms(file_to_test)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-1]  # drop useless channel 8
    # n_channel_data = np.delete(n_channel_data, 3, axis=0)  # drop broken channel 4m (for NoLeak ONLY)

    # print('TDMS data dim: ', n_channel_data.shape)

    temp = []
    if DOWNSAMPLE_FACTOR is not 1:
        for channel in n_channel_data:
            temp.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR))
        n_channel_data = np.array(temp)
        # print('Dim After Downsample: ', n_channel_data.shape)

    total_len = n_channel_data.shape[1]
    total_ch = len(n_channel_data)

    # ensure enough length of data input
    assert total_len > (window_size[0] + window_size[1]), 'Data length is too short, mz be at least {}'.\
        format(window_size[0] + window_size[1])
    window_index = np.arange(window_size[0], (total_len - window_size[1]), window_stride)
    # print('Window Index Len: ', len(window_index))
    # print('Window Index: ', window_index)

    # ---------------------------------------------------------------------------------------------------- EXECUTE MODEL
    # creating index of sliding windows [index-1000, index+5000]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    prediction_all_ch = []
    # for all channels
    for ch_no in range(total_ch):
        temp, model_pred = [], []
        # pb = ProgressBarForLoop(title='Iterating all Samples in ch[{}]'.format(ch_no), end=len(window_index))
        progress = 0
        for index in window_index:
            # pb.update(now=progress)
            data = n_channel_data[ch_no, (index - window_size[0]):(index + window_size[1])]
            # data_norm = scaler.fit_transform(data.reshape(-1, 1)).ravel()  # **normalize
            temp.append(data)

            # detect for last entry
            if progress < (len(window_index) - 1):

                if len(temp) < sample_size_for_prediction:
                    progress += 1
                    continue
                else:
                    progress += 1
                    # print('temp full !')

            # do tis when temp is full
            # reshape
            temp = np.array(temp)
            temp = temp.reshape((temp.shape[0], temp.shape[1], 1))
            # print(temp.shape)
            time_predict_start = time.time()
            prediction = np.argmax(lcp_model.predict(temp), axis=1)

            # # estimation of dist using all class posterior probability
            # estimation = []
            # for p in lcp_model.predict(temp):
            #     estimation.append(p[0]*0 + p[1]*2 + p[2]*4.5 + p[3]*5 + p[4]*8 + p[5]*10)

            time_predict = time.time() - time_predict_start
            model_pred.append(prediction)
            # reset temp
            temp = []
            # free up memory
            gc.collect()

        # pb.destroy()
        model_pred = np.concatenate(model_pred, axis=0)
        # print('Model Prediction Dim: ', model_pred.shape)

        prediction_all_ch.append(model_pred)

    prediction_all_ch = np.array(prediction_all_ch).T
    df_pred = pd.DataFrame(data=prediction_all_ch,
                           columns=SEEN_DATA_LABELS)

    # check the dir exist or nt
    if not os.path.exists(DIR_TO_SAVE_PRED_CSV_SEEN):
        os.makedirs(DIR_TO_SAVE_PRED_CSV_SEEN)

    df_pred.to_csv(df_pred_save_filename)

    # print('Saved --> ', df_pred_save_filename)
    # print('Reading --> ', df_pred_save_filename)
    df_pred = pd.read_csv(df_pred_save_filename, index_col=0)
    prediction_all_ch = df_pred.values.T.tolist()

    # -------------------------------------------------------------------------------------------------- CONFUSION MARIX
    conf_mat = []
    acc_per_ch = []

    # for all channel
    for ch, actual in zip(prediction_all_ch, ACTUAL_LABEL_ALL_CH_SEEN):
        acc = 0
        label_count_per_ch = []
        # count for all labels
        for label in MODEL_POSSIBLE_LABEL:
            count = ch.count(label)
            label_count_per_ch.append(count)
            # for actual class
            if isinstance(actual, tuple):
                if label in actual:
                    acc += count
            else:
                if label is actual:
                    acc += count
        # calc each channel classification acc
        acc_per_ch.append(acc / len(ch))
        # record the class count
        conf_mat.append(label_count_per_ch)

    conf_mat = np.array(conf_mat).T
    acc_per_ch_al_tdms.append(acc_per_ch)
    # merge col label with class accuracy
    col_label_w_acc = []
    for i, j in zip(SEEN_DATA_LABELS, acc_per_ch):
        col_label_w_acc.append(i + '\nacc: {:.4f}'.format(j))

    fig_cm = plot_confusion_matrix(cm=conf_mat,
                                   col_label=col_label_w_acc,
                                   row_label=MODEL_POSSIBLE_OUTPUT,
                                   title='Conf Mat - SEEN',
                                   verbose=False)  # **

    fig_cm_save_filename = direct_to_dir(where='result') + 'cm_' + filename_to_save + '.png'

    # check the dir exist or nt
    if not os.path.exists(DIR_TO_SAVE_CM_SEEN):
        os.makedirs(DIR_TO_SAVE_CM_SEEN)

    fig_cm.savefig(fig_cm_save_filename)

    plt.close('all')
    # print('Confusion Mat. fig saved -->', fig_cm_save_filename)
    # print('\n------------------------------------------------------------\n')
    print('Time taken: {:.4f}s'.format(time.time() - time_per_file_start))


# conclude overall acc for al class
acc_per_ch_al_tdms = np.array(acc_per_ch_al_tdms)
avg_acc_per_ch = np.mean(acc_per_ch_al_tdms, axis=0)
print('AVERAGE ACCURACY ----------------------')
for i, j in zip(SEEN_DATA_LABELS, avg_acc_per_ch):
    print(i + ' acc: {:.4f}'.format(j))

with open(RESULT_SAVE_FILENAME, 'a') as f:
    f.write('\n\n{}\nAVERAGE ACCURACY ----------------------'.format('SEEN LEAK by Dist'))
    for i, j in zip(SEEN_DATA_LABELS, avg_acc_per_ch):
        f.write('\n' + i + ' acc: {:.4f}'.format(j))

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------- EVALUATING UNSEEN DATA
acc_per_ch_al_tdms = []
for file_to_test in all_tdms_unseen:
    time_per_file_start = time.time()
    x = file_to_test.split(sep='/')[-1]
    # discard the .tdms
    x = x.split(sep='.')[-2]

    filename_to_save = 'pred_result_[{}]_[{}]_{}'.format(model_name, x, 'UNSEEN')

    # SAVING CONFIG
    df_pred_save_filename = DIR_TO_SAVE_PRED_CSV_UNSEEN + filename_to_save + '.csv'

    # test for near
    n_channel_data = read_single_tdms(file_to_test)
    n_channel_data = np.swapaxes(n_channel_data, 0, 1)[:-3]  # drop useless channel 8 & sensor at 16, 17m.
    # n_channel_data = np.delete(n_channel_data, 3, axis=0)  # drop broken channel 4m (for NoLeak ONLY)

    # print('TDMS data dim: ', n_channel_data.shape)

    temp = []
    if DOWNSAMPLE_FACTOR is not 1:
        for channel in n_channel_data:
            temp.append(decimate(x=channel, q=DOWNSAMPLE_FACTOR))
        n_channel_data = np.array(temp)
        # print('Dim After Downsample: ', n_channel_data.shape)

    total_len = n_channel_data.shape[1]
    total_ch = len(n_channel_data)

    # ensure enough length of data input
    assert total_len > (window_size[0] + window_size[1]), 'Data length is too short, mz be at least {}'.\
        format(window_size[0] + window_size[1])
    window_index = np.arange(window_size[0], (total_len - window_size[1]), window_stride)
    # print('Window Index Len: ', len(window_index))
    # print('Window Index: ', window_index)

    # ---------------------------------------------------------------------------------------------------- EXECUTE MODEL
    # creating index of sliding windows [index-1000, index+5000]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    prediction_all_ch = []
    # for all channels
    for ch_no in range(total_ch):
        temp, model_pred = [], []
        progress = 0
        for index in window_index:
            data = n_channel_data[ch_no, (index - window_size[0]):(index + window_size[1])]
            # data_norm = scaler.fit_transform(data.reshape(-1, 1)).ravel()  # **normalize
            temp.append(data)

            # detect for last entry
            if progress < (len(window_index) - 1):

                if len(temp) < sample_size_for_prediction:
                    progress += 1
                    continue
                else:
                    progress += 1
                    # print('temp full !')

            # do tis when temp is full
            # reshape
            temp = np.array(temp)
            temp = temp.reshape((temp.shape[0], temp.shape[1], 1))
            # print(temp.shape)
            time_predict_start = time.time()
            prediction = np.argmax(lcp_model.predict(temp), axis=1)

            # # estimation of dist using all class posterior probability
            # estimation = []
            # for p in lcp_model.predict(temp):
            #     estimation.append(p[0]*0 + p[1]*2 + p[2]*4.5 + p[3]*5 + p[4]*8 + p[5]*10)

            time_predict = time.time() - time_predict_start
            model_pred.append(prediction)
            # reset temp
            temp = []
            # free up memory
            gc.collect()

        # pb.destroy()
        model_pred = np.concatenate(model_pred, axis=0)
        # print('Model Prediction Dim: ', model_pred.shape)

        prediction_all_ch.append(model_pred)

    prediction_all_ch = np.array(prediction_all_ch).T
    df_pred = pd.DataFrame(data=prediction_all_ch,
                           columns=UNSEEN_DATA_LABELS)

    # check the dir exist or nt
    if not os.path.exists(DIR_TO_SAVE_PRED_CSV_UNSEEN):
        os.makedirs(DIR_TO_SAVE_PRED_CSV_UNSEEN)

    df_pred.to_csv(df_pred_save_filename)

    # print('Saved --> ', df_pred_save_filename)
    # print('Reading --> ', df_pred_save_filename)
    df_pred = pd.read_csv(df_pred_save_filename, index_col=0)
    prediction_all_ch = df_pred.values.T.tolist()

    # -------------------------------------------------------------------------------------------------- CONFUSION MARIX
    conf_mat = []
    acc_per_ch = []

    # for all channel
    for ch, actual in zip(prediction_all_ch, ACTUAL_LABEL_ALL_CH_UNSEEN):
        acc = 0
        label_count_per_ch = []
        # count for all labels
        for label in MODEL_POSSIBLE_LABEL:
            count = ch.count(label)
            label_count_per_ch.append(count)
            # for actual class
            if isinstance(actual, tuple):
                if label in actual:
                    acc += count
            else:
                if label is actual:
                    acc += count
        # calc each channel classification acc
        acc_per_ch.append(acc / len(ch))
        # record the class count
        conf_mat.append(label_count_per_ch)

    conf_mat = np.array(conf_mat).T
    acc_per_ch_al_tdms.append(acc_per_ch)
    # merge col label with class accuracy
    col_label_w_acc = []
    for i, j in zip(UNSEEN_DATA_LABELS, acc_per_ch):
        col_label_w_acc.append(i + '\nacc: {:.4f}'.format(j))

    fig_cm = plot_confusion_matrix(cm=conf_mat,
                                   col_label=col_label_w_acc,
                                   row_label=MODEL_POSSIBLE_OUTPUT,
                                   title='Conf Mat - UNSEEN',
                                   verbose=False)  # **

    fig_cm_save_filename = direct_to_dir(where='result') + 'cm_' + filename_to_save + '.png'

    # check the dir exist or nt
    if not os.path.exists(DIR_TO_SAVE_CM_UNSEEN):
        os.makedirs(DIR_TO_SAVE_CM_UNSEEN)

    fig_cm.savefig(fig_cm_save_filename)

    plt.close('all')
    # print('Confusion Mat. fig saved -->', fig_cm_save_filename)
    # print('\n------------------------------------------------------------\n')
    print('Time taken: {:.4f}s'.format(time.time() - time_per_file_start))


# conclude overall acc for al class
acc_per_ch_al_tdms = np.array(acc_per_ch_al_tdms)
avg_acc_per_ch = np.mean(acc_per_ch_al_tdms, axis=0)
print('AVERAGE ACCURACY ----------------------')
for i, j in zip(UNSEEN_DATA_LABELS, avg_acc_per_ch):
    print(i + ' acc: {:.4f}'.format(j))

with open(RESULT_SAVE_FILENAME, 'a') as f:
    f.write('\n\n{}\nAVERAGE ACCURACY ----------------------'.format('UNSEEN LEAK by Dist'))
    for i, j in zip(UNSEEN_DATA_LABELS, avg_acc_per_ch):
        f.write('\n' + i + ' acc: {:.4f}'.format(j))


# # saving the values to buffer, for overall calculation
# file_dir = direct_to_dir(where='result') + '{}_acc_buffer.csv'.format(model_name)
#
# if not os.path.exists(file_dir):
#     df = pd.DataFrame(data=avg_acc_per_ch, columns=[FIG_CM_TITLE])
#     df.to_csv(file_dir)
# else:
#     df = pd.read_csv(file_dir, index_col=0)
#     print(df)
#     df[FIG_CM_TITLE] = avg_acc_per_ch
#     df.to_csv(file_dir)
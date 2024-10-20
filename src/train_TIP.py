import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import time
from multiprocessing import cpu_count
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from lib.operation import *
from lib.divideData import divideTipData
from cross_validation_draw import plotDraw, historyDraw
from predict_RBV import predictRBV


# model parameter
sliding_len = 25  # the sliding window length
forNum = 60
revNum = 30
n_input = forNum + revNum - sliding_len + 1
n_classes = 1
batch_size = 8  # batch size
training_epochs = 1000  # iteration epochs
learning_rate = 1e-4  # learning rate


def trainModel(utr_file, scan_file, model_name, output_fold):
    # save params
    save_train_file = '{}/TIP_train_predict.csv'.format(output_fold)
    save_plot = '{}/TIP_train_plot.jpg'.format(output_fold)
    save_train_scatter = '{}/TIP_train_scatter.jpg'.format(output_fold)

    save_val_file = '{}/TIP_val_predict.csv'.format(output_fold)
    save_val_scatter = '{}/TIP_val_scatter.jpg'.format(output_fold)

    # divide origin utr file
    train_data, val_data = divideTipData(utr_file, scan_file)

    # get train set
    train_num = len(train_data)
    train_index = np.random.permutation(train_num)  # disrupt the train set
    train_data = train_data[train_index]
    train_gene, train_seq, train_act, train_input = train_data[:, 0], train_data[:, 1], train_data[:, 2], train_data[:, 3:]
    train_act = np.array(train_act).astype(float)
    train_input = np.array(train_input).astype(float)

    # get val set
    val_num = len(val_data)
    val_index = np.random.permutation(val_num)  # disrupt the train set
    val_data = val_data[val_index]
    val_gene, val_seq, val_act, val_input = val_data[:, 0], val_data[:, 1], val_data[:, 2], val_data[:, 3:]
    val_act = np.array(val_act).astype(float)
    val_input = np.array(val_input).astype(float)

    # Ready to build model
    model = Sequential()
    model.add(Dense(128, input_dim=n_input))
    # model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # model.add(Dropout(0.1))

    model.add(Dense(n_classes))  # output layer

    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # optimizer
    # optimizer = SGD(learning_rate=learning_rate)  # optimizer
    callbacks = [
        ModelCheckpoint(model_name, save_best_only=True, monitor="loss"),
        EarlyStopping(monitor="val_loss", patience=300, verbose=0)
    ]

    model.compile(optimizer=optimizer, loss="mse", metrics=['mse'])
    
    history = model.fit(
        train_input,
        train_act,
        batch_size=batch_size,
        epochs=training_epochs,
        callbacks=callbacks,
        validation_data=(val_input, val_act),
        verbose=1,
        )

    model = load_model(model_name)
    train_pred = model.predict(train_input)
    train_pred = train_pred.reshape(1, train_num)[0]
    val_pred = model.predict(val_input)
    val_pred = val_pred.reshape(1, val_num)[0]
    
    historyDraw(history, save_plot)
    plotDraw(train_act, train_pred, save_train_scatter)
    plotDraw(val_act, val_pred, save_val_scatter)

    # write prediction to saved train file
    train_pred_data = [['Gene Name', 'UTR Sequence', 'Actual Acitivity', 'Actual Acitivity-ln', 'pred_TIP', 'pred_TIP_ln']]
    for i in range(train_num):
        train_pred_data.append([train_gene[i], train_seq[i], 
                          np.exp(train_act[i]), train_act[i], 
                          np.exp(train_pred[i]), train_pred[i]])
    csvSave(save_train_file, train_pred_data)

    # write prediction to saved validation file
    val_pred_data = [['Gene Name', 'UTR Sequence', 'Actual Acitivity', 'Actual Acitivity-ln', 'pred_TIP', 'pred_TIP_ln']]
    for i in range(val_num):
        val_pred_data.append([val_gene[i], val_seq[i], 
                              np.exp(val_act[i]), val_act[i], 
                              np.exp(val_pred[i]), val_pred[i]])
    csvSave(save_val_file, val_pred_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TIP Arguments')
    parser.add_argument('-i', '--input_file', type=str, default='./train/input/sample_TIP_trainset.xlsx', help='dataset file name')
    parser.add_argument('-sf', '--scan_file', type=str, default='./train/input/sample_TIP_scanned.csv', help='scanned dataset file name')
    parser.add_argument('-tmf', '--tip_model_fold', type=str, default='./saved_models/TIP', help='TIP model fold')
    parser.add_argument('-to', '--tip_output_fold', type=str, default='./train/TIP_output', help='TIP output fold')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-p', '--process_sum', type=int, default=cpu_count() // 2 - 1, help='process sum')
    parser.add_argument('-ss', '--server_sum', type=int, default=1, help='server sum')
    parser.add_argument('-sn', '--server_num', type=int, default=0, help='server num')
    args = parser.parse_args()

    if args.process_sum >= cpu_count():
        args.process_sum = cpu_count() // 2 - 1
    
    if not os.path.exists(args.input_file):
        print('ERROR: Please ensure the input file path exists!')
        exit()

    batch_size = args.batch_size
    learning_rate = args.learning_rate

    utr_file = args.input_file
    scan_file = args.scan_file
    model_fold = args.tip_model_fold
    output_fold = args.tip_output_fold

    # create model fold and output fold
    if not os.path.exists(model_fold):
        os.mkdir(model_fold)
    if not os.path.exists(output_fold):
        os.mkdir(output_fold)

    if not os.path.exists(scan_file):
        # wait scanned file
        print('Please waiting for generating scanned file.', end='')
    while not os.path.exists(scan_file):
        time.sleep(5)
        print('.', end='')
    print('\n')

    # train model
    print('Ready to train TIP model.')
    model_name = '{}/TIP_model.h5'.format(model_fold)
    trainModel(utr_file, scan_file, model_name, output_fold)
    print('Finish.')

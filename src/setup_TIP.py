import argparse
from multiprocessing import Process, cpu_count
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import sys
system = sys.platform
import psutil
import math
import time

from lib.operation import *
from lib.log import *
from lib.divideData import divideTipData
from train_RBV import *
from cross_validation_draw import crossValandDraw

# model parameter
sliding_len = 25  # the sliding window length
forNum = 60
revNum = 30
n_input = forNum + revNum - sliding_len + 1
n_classes = 1
batch_size = 8  # batch size
training_epochs = 1000  # iteration epochs
learning_rate = 1e-4  # learning rate


def trainModel(args, model_name, save_val_file, save_pic_file, logger):
    # update parameters
    utr_file = args.input_file
    scan_file = args.scan_file

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
    if logger: logger.info('build the model')

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
        verbose=0,
        )

    # Ready to predict val data set
    if logger: logger.info('Evaluate model on validation data')
    model = load_model(model_name)
    val_pred = model.predict(val_input, verbose=0)
    val_pred = val_pred.reshape(1, val_num)[0]

    # write prediction to saved validation file
    val_pred_data = [['Gene Name', 'UTR Sequence', 'Actual Acitivity', 'Actual Acitivity-ln', 'Pred TIP', 'Pred TIP-ln']]
    for i in range(val_num):
        val_pred_data.append([val_gene[i], val_seq[i], 
                              np.exp(val_act[i]), val_act[i], 
                              np.exp(val_pred[i]), val_pred[i]])
    csvSave(save_val_file, val_pred_data)

     # draw validation scatter
    plotDraw(val_act, val_pred, save_pic_file)


def tipTask(process_num, args, PID_fold, log_fold, save_val_fold, save_pic_fold):
    # Get the PID, bind it to the specified core, log it for process killing
    pid = os.getpid()
    if system == 'win32':
        p = psutil.Process()
        p.cpu_affinity([process_num])
    elif system == 'linux':
        _ = os.system('taskset -pc {} {} '.format(process_num, pid))
    PID_file = '{}/PID_{}.txt'.format(PID_fold, process_num)
    # record PID
    txtSave(PID_file, [str(pid)])

    # create log file
    log_file = '{}/train_log_{}.log'.format(log_fold, process_num)
    checkLog(log_file)
    logger = createLogging(log_file, 'Train TIP model')
    logger.info('-----------------------Process {}(PID-{}) Start--------------------------'.format(process_num, pid))

    # get parameters
    model_num = args.tip_model_num
    model_fold = args.tip_model_fold
    process_sum = args.process_sum

    # train multiple models
    trained_tip_model_num = math.ceil(model_num / process_sum)
    start_model_index = process_num * trained_tip_model_num
    stop_model_index = min([(process_num + 1) * trained_tip_model_num, model_num])
    for model_idx in range(start_model_index, stop_model_index):
        logger.info('-----------------------Train Model {}--------------------------'.format(model_idx))
        # train model
        model_name = '{}/TIP_model_{}.h5'.format(model_fold, model_idx)
        save_val_file = '{}/{}_val_pred.csv'.format(save_val_fold, model_idx)
        save_pic_file = '{}/{}_val_scatter.jpg'.format(save_pic_fold, model_idx)
        trainModel(args, model_name, save_val_file, save_pic_file, logger)
    logger.info('-----------------------Finish--------------------------')


def trainTIP(args):
    # create fold for recording PID file fold
    PID_fold = '{}/PID'.format(args.tip_output_fold)
    if os.path.exists(PID_fold):
        removeFolds(PID_fold)
    os.mkdir(PID_fold)

    # create fold for log file fold
    log_fold = '{}/log'.format(args.tip_output_fold)
    if os.path.exists(log_fold):
        removeFolds(log_fold)
    os.mkdir(log_fold)

    # create fold for predictions file fold
    save_val_fold = '{}/pred_val'.format(args.tip_output_fold)
    if os.path.exists(save_val_fold):
        removeFolds(save_val_fold)
    os.mkdir(save_val_fold)

    # create fold for scatter pictures fold
    save_pic_fold = '{}/scatter_val'.format(args.tip_output_fold)
    if os.path.exists(save_pic_fold):
        removeFolds(save_pic_fold)
    os.mkdir(save_pic_fold)

    # remove old cross validation prediction file
    save_cross_val_file = '{}/cross_validation_pred.csv'.format(args.tip_output_fold)
    if os.path.exists(save_cross_val_file):
        os.remove(save_cross_val_file)
    
    # remove old cross validation picture file
    save_pic_file = '{}/scatter_cross_validation.jpg'.format(args.tip_output_fold)
    if os.path.exists(save_pic_file):
        os.remove(save_pic_file)
    
    # divide tasks, different tasks train different numbered models
    print('Ready to train tip models.')
    process_sum = args.process_sum
    process_list = []  # process pool
    for p_num in range(process_sum):
        p = Process(target=tipTask, args=(p_num, args, PID_fold, log_fold, save_val_fold, save_pic_fold, ))
        p.start()
        process_list.append(p)
    
    # main process wait for child process to finish
    for p in process_list:
        p.join()

    # Ready to cross validation and draw
    print('Cross validation.')
    crossValandDraw(save_cross_val_file, save_pic_file, save_val_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TIP Arguments')
    parser.add_argument('-i', '--input_file', type=str, default='./train/input/sample_TIP_trainset.xlsx', help='dataset file name')
    parser.add_argument('-sf', '--scan_file', type=str, default='./train/input/sample_TIP_scanned.csv', help='scanned dataset file name')
    parser.add_argument('-tmn', '--tip_model_num', type=int, default=2000, help='TIP model num')
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

    scan_file = args.scan_file
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    if not os.path.exists(scan_file):
        # wait scanned file
        print('Please waiting for generating scanned file.', end='')
    while not os.path.exists(scan_file):
        time.sleep(5)
        print('.', end='')
    print('\n')

    trainTIP(args)

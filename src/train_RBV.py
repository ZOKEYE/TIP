from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tqdm import *
import argparse

from lib.operation import *
from lib.divideData import divideUtrData
from cross_validation_draw import crossValandDraw, plotDraw


ATCG_oneHot = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'U': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
}


# model parameter
ntNum = 25  # 5'utr length
n_input = ntNum  # input layer size
kernel_size1 = 9  # first convolutional layer size
n_conv = n_input - kernel_size1 + 1  # input size of second convolutional layer
kernel_size2 = 17  # second convolutional layer size
activation = 'relu'  # activation function
padding = 'valid'  # padding mode ['valid', 'same']
n_channel = 32  # channel size
n_hidden = 16  # number of neurons in fully connected layer
n_classes = 1  # number of neurons in output layer
batch_size = 128  # batch size
training_epochs = 1000  # iteration epochs
learning_rate = 1e-3  # learning rate


def transX(x):
    # one-hot encoding
    return np.array([[ATCG_oneHot[__] for __ in _] for _ in x])


def trainModel(args, model_name, save_test_file, save_pic_file, logger, verbose=0):
    # update parameters
    utr_file = args.dataset
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # divide origin utr file
    train_data, val_data, test_data = divideUtrData(utr_file)

    # get train set
    train_num = len(train_data)
    train_seq, train_act = train_data[:, 1], train_data[:, 2].astype(float)
    train_index = np.random.permutation(train_num)  # disrupt the train set
    train_seq = transX(train_seq[train_index])
    train_act = train_act[train_index]

    # get validation set
    val_num = len(val_data)
    val_seq, val_act = val_data[:, 1], val_data[:, 2].astype(float)
    val_index = np.random.permutation(val_num)  # disrupt the validation set
    val_seq = transX(val_seq[val_index])
    val_act = val_act[val_index]

    # get test set
    test_num = len(test_data)
    test_id, test_seq, test_act = test_data[:, 0], test_data[:, 1], test_data[:, 2].astype(float)
    test_seq = transX(test_seq)

    # Ready to build model
    if logger: logger.info('build the model')

    # construct a sequential model
    model = Sequential()
    # input layer (first convolutional layer)
    model.add(Conv2D(filters=n_channel, kernel_size=(kernel_size1, 4),
                        padding=padding, activation=activation, input_shape=(n_input, 4, 1)))
    # second convolutional layer
    model.add(Conv2D(filters=n_channel, kernel_size=(kernel_size2, 1), 
                        padding=padding, activation=activation, input_shape=(n_conv, 1, 1)))
    model.add(Flatten())  # flatten layer
    model.add(Dense(n_hidden, activation=activation))  # fully connected layer
    model.add(Dense(n_classes))  # output layer

    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # optimizer
    callbacks = [
        ModelCheckpoint(model_name, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=200, verbose=verbose)
    ]

    model.compile(optimizer=optimizer, loss="mse", metrics=['mse'])

    history = model.fit(
        train_seq.reshape(train_num, n_input, 4, 1),
        train_act,
        batch_size=batch_size,
        epochs=training_epochs,
        callbacks=callbacks,
        validation_data=(val_seq.reshape(val_num, n_input, 4, 1), val_act),
        verbose=verbose,
    )

    # Ready to predict test set
    if logger: logger.info('Evaluate model on test data')
    model = load_model(model_name)
    pred_test = model.predict(test_seq.reshape(test_num, ntNum, 4, 1), verbose=0)
    pred_test = pred_test.reshape(1, test_num)[0]

    # write prediction to saved file
    pred_data = [['Gene Name', 'UTR Sequence', 'Actual Acitivity', 'Actual Acitivity-ln', 'pred_RBV', 'pred_RBV_ln']]
    for i in range(test_num):
        pred_data.append([test_data[i, 0], test_data[i, 1], 
                          np.exp(test_data[i, 2]), test_data[i, 2], 
                          np.exp(pred_test[i]), pred_test[i]])

    csvSave(save_test_file, pred_data)

    # draw test scatter
    plotDraw(test_act, pred_test, save_pic_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RBV Arguments')
    parser.add_argument('-rmn', '--rbv_model_num', type=int, default=1, help='RBV model num')
    parser.add_argument('-rmf', '--rbv_model_fold', type=str, default='./saved_models/RBV', help='RBV model fold')
    parser.add_argument('-ro', '--rbv_output_fold', type=str, default='./train/RBV_output', help='RBV output fold')
    parser.add_argument('-d', '--dataset', type=str, default='./train/input/sample_RBV_dataset.xlsx', help='dataset file name')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print('ERROR: Please ensure the input file path exists!')
        exit()

    # check there exists model fold or not
    if not os.path.exists(args.rbv_model_fold):
        os.mkdir(args.rbv_model_fold)
    # check there exists data save fold or not
    if not os.path.exists(args.rbv_output_fold):
        os.mkdir(args.rbv_output_fold)

    # remove old cross validation file 
    save_cross_val_file = '{}/test_cross_validation_pred.csv'.format(args.rbv_output_fold)
    if os.path.exists(save_cross_val_file):
        removeFolds(save_cross_val_file)
    # remove old picture save file
    save_pic_file = '{}/scatter_cross_validation.jpg'.format(args.rbv_output_fold)
    if os.path.exists(save_pic_file):
        removeFolds(save_pic_file)
    # check file fold for test set prediction
    save_test_fold = '{}/pred_test'.format(args.rbv_output_fold)
    if os.path.exists(save_test_fold):
        removeFolds(save_test_fold)
    os.mkdir(save_test_fold)
    # check picture fold for test set prediction
    save_pic_fold = '{}/scatter_test'.format(args.rbv_output_fold)
    if os.path.exists(save_pic_fold):
        removeFolds(save_pic_fold)
    os.mkdir(save_pic_fold)
    
    # train models
    print('Ready to train RBV models.')
    for model_idx in range(args.rbv_model_num):
        model_name = '{}/{}_best_model.h5'.format(args.rbv_model_fold, model_idx)
        save_test_file = '{}/{}_pred_test.csv'.format(save_test_fold, model_idx)
        save_pic_file = '{}/{}_scatter_test.jpg'.format(save_pic_fold, model_idx)
        trainModel(args, model_name, save_test_file, save_pic_file, None, verbose=1)
    
    # cross validation
    print('Cross validation.')
    crossValandDraw(save_cross_val_file, save_pic_file, save_test_fold)

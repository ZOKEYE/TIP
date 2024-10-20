import argparse
from multiprocessing import Process, cpu_count
import sys
system = sys.platform
import psutil
import math
import time

from lib.operation import *
from lib.log import *
from train_RBV import *
from cross_validation_draw import crossValandDraw


def rbvTask(process_num, args, PID_fold, log_fold, save_test_fold, save_pic_fold):
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
    logger = createLogging(log_file, 'Train RBV model')
    logger.info('-----------------------Process {}(PID-{}) Start--------------------------'.format(process_num, pid))

    # get parameters
    rbv_model_num = args.rbv_model_num
    rbv_model_fold = args.rbv_model_fold
    process_sum = args.process_sum

    # train multiple models
    trained_rbv_model_num = math.ceil(rbv_model_num / process_sum)
    start_model_index = process_num * trained_rbv_model_num
    stop_model_index = min([(process_num + 1) * trained_rbv_model_num, rbv_model_num])
    for model_idx in range(start_model_index, stop_model_index):
        logger.info('-----------------------Train Model {}--------------------------'.format(model_idx))
        # train model
        model_name = '{}/{}_best_model.h5'.format(rbv_model_fold, model_idx)
        save_test_file = '{}/{}_pred_test.csv'.format(save_test_fold, model_idx)
        save_pic_file = '{}/{}_scatter_test.jpg'.format(save_pic_fold, model_idx)
        trainModel(args, model_name, save_test_file, save_pic_file, logger)
    logger.info('-----------------------Finish--------------------------')


def trainRBV(args):
    # create fold for recording PID file fold
    PID_fold = '{}/PID'.format(args.rbv_output_fold)
    if os.path.exists(PID_fold):
        removeFolds(PID_fold)
    os.mkdir(PID_fold)

    # create fold for log file fold
    log_fold = '{}/log'.format(args.rbv_output_fold)
    if os.path.exists(log_fold):
        removeFolds(log_fold)
    os.mkdir(log_fold)

    # create fold for predictions file fold
    save_test_fold = '{}/pred_test'.format(args.rbv_output_fold)
    if os.path.exists(save_test_fold):
        removeFolds(save_test_fold)
    os.mkdir(save_test_fold)

    # create fold for scatter pictures fold
    save_pic_fold = '{}/scatter_test'.format(args.rbv_output_fold)
    if os.path.exists(save_pic_fold):
        removeFolds(save_pic_fold)
    os.mkdir(save_pic_fold)

    # remove old cross validation prediction file
    save_cross_val_file = '{}/test_cross_validation_pred.csv'.format(args.rbv_output_fold)
    if os.path.exists(save_cross_val_file):
        os.remove(save_cross_val_file)
    
    # remove old cross validation picture file
    save_pic_file = '{}/scatter_cross_validation.jpg'.format(args.rbv_output_fold)
    if os.path.exists(save_pic_file):
        os.remove(save_pic_file)
    
    # divide tasks, different tasks train different numbered models
    print('Ready to train models.')
    process_sum = args.process_sum
    process_list = []  # process pool
    for p_num in range(process_sum):
        p = Process(target=rbvTask, args=(p_num, args, PID_fold, log_fold, save_test_fold, save_pic_fold, ))
        p.start()
        process_list.append(p)
    
    # main process wait for child process to finish
    for p in process_list:
        p.join()

    # Ready to cross validation and draw
    print('Cross validation.')
    crossValandDraw(save_cross_val_file, save_pic_file, save_test_fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='You could add these parameter.')
    parser.add_argument('-rmn', '--rbv_model_num', type=int, default=2000, help='RBV model num')
    parser.add_argument('-rmf', '--rbv_model_fold', type=str, default='./saved_models/RBV', help='RBV model fold')
    parser.add_argument('-ro', '--rbv_output_fold', type=str, default='./train/RBV_output', help='RBV output fold')
    parser.add_argument('-d', '--dataset', type=str, default='./train/input/sample_RBV_dataset.xlsx', help='dataset file name')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-p', '--process_sum', type=int, default=cpu_count() // 2 - 1, help='process sum')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print('ERROR: Please ensure the input file path exists!')
        exit()

    if args.process_sum >= cpu_count():
        args.process_sum = cpu_count() // 2 - 1

    args.process_sum = 1

    trainRBV(args)

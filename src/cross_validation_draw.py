from lib.operation import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def crossValandDraw(save_cross_file, save_pic_file, save_test_fold):
    # cross validation
    act_dict, pred_dict = dict(), dict()
    save_files = os.listdir(save_test_fold)
    for save_file_name in save_files:
        # read test prediction data
        save_test_file = '{}/{}'.format(save_test_fold, save_file_name)
        f = open(save_test_file, 'r')
        csv_reader = csv.reader(f)
        pred_count = 0
        for pred_data in csv_reader:
            # Skip the first line
            if pred_count == 0:
                pred_count += 1
                continue
            gene, seq, act, pred = pred_data[0], pred_data[1], float(pred_data[3]), float(pred_data[5])
            test_tuple = (gene, seq)
            # store actual activity
            if test_tuple not in act_dict:
                act_dict[test_tuple] = act
            # store/add prediction
            if test_tuple in pred_dict:
                pred_dict[test_tuple].append(pred)
            else:
                pred_dict[test_tuple] = [pred]
    # calculate the median of the prediction
    test_act, test_pred = [], []
    pred_data = [['Gene Name', 'UTR Sequence', 'Actual Acitivity', 'Actual Acitivity-ln', 'Pred Activity', 'Pred Activity-ln']]
    for test_tuple in pred_dict:
        act = act_dict[test_tuple]
        test_act.append(act)

        test_pred_set = np.array(pred_dict[test_tuple])
        pred = np.percentile(test_pred_set, 50)
        test_pred.append(pred)

        pred_data.append([test_tuple[0], test_tuple[1], np.exp(act), act, np.exp(pred), pred])
    csvSave(save_cross_file, pred_data)

    # draw plot
    plotDraw(test_act, test_pred, save_pic_file)


def plotDraw(activity, predict, pic_save_file):
    # Pearson correlation coefficient
    pearson = pearsonr(activity, predict)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    font = {'style': 'italic'}
    min_data = min([min(activity), min(predict)])
    max_data = max([max(activity), min(predict)])
    ticks = [_ for _ in range(int(min_data - 1), int(max_data + 1) + 1)]
    plt.xticks(ticks, ticks, fontsize=15)
    plt.yticks(ticks, ticks, fontsize=15)
    plt.xlabel('Actual Activity', fontdict=font, fontsize=15)
    plt.ylabel('Pred TSV-ln', fontdict=font, fontsize=15)
    plt.scatter(activity, predict, s=10, label='Pearson=%.4f' % pearson)
    plt.legend(loc='lower right', fontsize=15)

    xx = [_ for _ in np.arange(min_data, max_data + 0.1, 0.1)]
    plt.plot(xx, xx, color='silver')
    plt.savefig(pic_save_file)


def historyDraw(history, pic_save_file):
    metric = "mse"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(pic_save_file)
    plt.close()

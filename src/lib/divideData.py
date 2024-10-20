from lib.operation import *
from lib.error_check import *


ntNum = 25


def readUtrFile(file_name):
    # read 5'UTR dataset
    df = pd.read_excel(file_name)
    check_colname(df, ['Gene Name', 'UTR Sequence', 'Actual Activity'])
    GeneName = np.array(df['Gene Name'])  # gene name
    Sequence = np.array(df['UTR Sequence'])  # 5'UTR sequences
    Activity = np.array(df['Actual Activity'])  # actual activity

    check_gene_norepeat(GeneName)
    check_seq_valid(Sequence)
    check_seq_length_equal(Sequence, ntNum)
    check_act_valid(Activity)

    Activity = np.log(Activity.astype(float))  # activity taken as logarithmic
    # combine data
    length = len(GeneName)
    utr_data = np.hstack((GeneName.reshape(length, 1), Sequence.reshape(length, 1)))  # combine gene name with sequence
    utr_data = np.hstack((utr_data, Activity.reshape(length, 1)))  # combine sequence with activity
    utr_data = utr_data[np.argsort(Activity.astype(float))[::-1]]  # sort by activity in descending order
    return utr_data


def readScannedFile(utr_file, scan_file):
    # read rbv model scanning data
    origin_data = csvRead(scan_file)
    RbvDict = dict()
    for i in range(1, len(origin_data)):
        genename = origin_data[i][0]
        check_scanned_rbv(origin_data[i][3:])
        RbvDict[genename] = origin_data[i][3:]

    # read 5'UTR dataset
    df = pd.read_excel(utr_file)
    check_colname(df, ['Gene Name', 'UTR Sequence', 'Actual Activity'])
    GeneName = np.array(df['Gene Name'])  # gene name
    Sequence = np.array(df['UTR Sequence'])  # 5'UTR sequences
    Activity = np.array(df['Actual Activity'])  # actual activity

    check_gene_norepeat(GeneName)
    check_seq_valid(Sequence)
    check_seq_length_equal(Sequence, ntNum)
    check_act_valid(Activity)

    Activity = np.log(Activity.astype(float))  # activity taken as logarithmic
    utr_data = []
    for i in range(len(GeneName)):
        genename, sequence, activity = GeneName[i], Sequence[i], Activity[i]
        rbv_data = np.log(np.array(RbvDict[genename]).astype(float)).astype(str)  # rbv scanned data taken as logarithmic
        rbv_data = np.insert(rbv_data, 0, activity)  # insert activity
        rbv_data = np.insert(rbv_data, 0, sequence)  # insert sequence
        rbv_data = np.insert(rbv_data, 0, genename)  # insert genename
        utr_data.append(rbv_data)
    
    utr_data = np.array(utr_data)
    utr_data = utr_data[np.argsort(Activity)[::-1]]  # sort by activity in descending order
    return utr_data


def getRandomIdx(input_data, idx_num):
    # Get an index of test and validation dataset selections
    data_size = input_data.shape[0]  # dataset size

    # Divide the intervals and take a random number in each interval
    min_range = np.linspace(0, data_size, num=idx_num + 1).astype(int)  # left edge
    max_range = np.delete(min_range, [0])  # edge 
    min_range = np.delete(min_range, [min_range.size - 1])

    chosen_idx = np.array([], dtype=int)
    for i in range(min_range.size):
        # Pick a random index in each interval
        random_idx = np.random.randint(low=min_range[i], high=max_range[i])
        chosen_idx = np.append(chosen_idx, random_idx)
    return chosen_idx


def divideUtrData(utr_file, percent=0.1):
    # Get the original utr dataset
    utr_data = readUtrFile(utr_file)
    data_size = utr_data.shape[0]  # dataset size
    test_num = int(data_size * percent)  # number of test dataset
    val_num = test_num  # number of validation dataset
    train_num = data_size - test_num - val_num

    # get test set
    test_idx = getRandomIdx(utr_data, test_num)
    test_data = utr_data[test_idx]  # select test set according to index
    train_data = np.delete(utr_data, test_idx, axis=0)  # remove test set from train set

    # get validation set
    val_idx = getRandomIdx(train_data, val_num)
    val_data = train_data[val_idx]  # select validation set according to index

    # get train set
    train_data = np.delete(train_data, val_idx, axis=0)  # remove validation set from train set

    return train_data, val_data, test_data


def divideTipData(utr_file, scan_file, percent=0.2):
    train_data = readScannedFile(utr_file, scan_file)
    train_num = len(train_data)
    val_num = int(train_num * percent)

    # get validation set
    val_idx = getRandomIdx(train_data, val_num)
    val_data = train_data[val_idx]  # select validation set according to index
    train_data = np.delete(train_data, val_idx, axis=0)  # remove validation set from train set

    return train_data, val_data


def divideRBVPredictData(args):
    # get params
    source_file = args.input_file  # input file
    server_sum = args.server_sum  # server sum
    server_num = args.server_num  # server num
    core_sum = args.process_sum  # coresum

    # Read input file in xlsx form, and get the gene names and sequence arrays.
    Gene, Sequence, Activity = [], [], []
    df = pd.read_excel(source_file)
    check_colname(df, ['Gene Name', 'UTR Sequence'])
    Gene = np.array(df['Gene Name']).astype(str)  # gene name
    check_gene_norepeat(Gene)
    Sequence = np.array(df['UTR Sequence'])  # 5'UTR sequences
    check_seq_valid(Sequence)
    check_seq_length_grater(Sequence, ntNum)
    Input_Length = len(Sequence)
    try:
        Activity = np.array(df['Actual Activity'])  # actual activity
    except:
        Activity = np.array([0] * Input_Length)

    # equalize tasks to each server, and equalize sequences
    per_server_num = Input_Length // server_sum
    residual = Input_Length % server_sum
    base = (np.zeros(shape=server_sum).astype(int) + 1) * per_server_num
    for _ in range(residual):
        base[_] += 1
    stop_index = np.cumsum(base)

    # receive quests based on server number
    origin_gene, origin_sequence = [], []  # origin data
    slide_gene, slide_sequence = [], []  # 25nt slice based on origin data
    if server_num == 0:
        left = 0
    else:
        left = stop_index[server_num - 1]
    right = stop_index[server_num]
    for index in range(left, right):
        gene = Gene[index]
        sequence = Sequence[index].upper()
        origin_gene.append(gene)
        origin_sequence.append(sequence)
        # ready to slice
        for t in range(len(sequence) - ntNum + 1):
            sub_gene = '{}_{}'.format(gene, t)
            sub_seq = sequence[t:t + ntNum]
            slide_gene.append(sub_gene)
            slide_sequence.append(sub_seq)
    origin_gene = np.array(origin_gene)
    origin_sequence = np.array(origin_sequence)
    slide_gene = np.array(slide_gene)
    slide_sequence = np.array(slide_sequence)

    # Distribute the sequences picked up by this server equally to each core(process)
    slide_data_len = len(slide_sequence)
    per_core_num = slide_data_len // core_sum
    _res = slide_data_len % core_sum
    _base = (np.zeros(shape=core_sum).astype(int) + 1) * per_core_num
    for _ in range(_res):
        _base[_] += 1
    stop_core_index = np.cumsum(_base)

    divided_slide_gene = [slide_gene[0:stop_core_index[0]]]
    divided_slide_sequence = [slide_sequence[0:stop_core_index[0]]]
    for core_num in range(1, core_sum - 1):
        divided_slide_gene.append(slide_gene[stop_core_index[core_num - 1]:stop_core_index[core_num]])
        divided_slide_sequence.append(slide_sequence[stop_core_index[core_num - 1]:stop_core_index[core_num]])
    divided_slide_gene.append(slide_gene[stop_core_index[core_sum - 2]:])
    divided_slide_sequence.append(slide_sequence[stop_core_index[core_sum - 2]:])
    return origin_gene, origin_sequence, divided_slide_gene, divided_slide_sequence, Activity


def divideTIPPredictData(args):
    # get params
    source_file = args.input_file  # input file
    scanned_file = args.scan_file  # scanned file
    server_sum = args.server_sum  # server sum
    server_num = args.server_num  # server num
    core_sum = args.process_sum  # coresum

    # Read input file in xlsx form, and get the gene names and sequence arrays.
    Gene, Sequence, Activity = [], [], []
    df = pd.read_excel(source_file)
    check_colname(df, ['Gene Name', 'UTR Sequence'])
    Gene = np.array(df['Gene Name']).astype(str)  # gene name
    check_gene_norepeat(Gene)
    Sequence = np.array(df['UTR Sequence'])  # 5'UTR sequences
    check_seq_valid(Sequence)
    check_seq_length_equal(Sequence, 90)
    Input_Length = len(Sequence)
    try:
        Activity = np.array(df['Actual Activity'])  # actual activity
    except:
        Activity = np.array([0] * Input_Length)
        args.draw_flag = 0

    # read scanned rbv data
    origin_scanned_data = csvRead(scanned_file)
    ScannedRBV = []
    for idx in range(1, len(origin_scanned_data)):
        ScannedRBV.append(origin_scanned_data[idx][3:])
    ScannedRBV = np.log(np.array(ScannedRBV).astype(float))

    # equalize tasks to each server, and equalize sequences
    per_server_num = Input_Length // server_sum
    residual = Input_Length % server_sum
    base = (np.zeros(shape=server_sum).astype(int) + 1) * per_server_num
    for _ in range(residual):
        base[_] += 1
    stop_index = np.cumsum(base)

    # receive quests based on server number
    origin_gene, origin_sequence, origin_rbv_data = [], [], []  # origin data
    if server_num == 0:
        left = 0
    else:
        left = stop_index[server_num - 1]
    right = stop_index[server_num]
    for idx in range(left, right):
        gene = Gene[idx]
        sequence = Sequence[idx]
        rbv_data = ScannedRBV[idx]
        origin_gene.append(gene)
        origin_sequence.append(sequence)
        origin_rbv_data.append(rbv_data)
    origin_gene = np.array(origin_gene)
    origin_sequence = np.array(origin_sequence)
    origin_rbv_data = np.array(origin_rbv_data)

    # Distribute the sequences picked up by this server equally to each core(process)
    data_len = len(origin_gene)
    per_core_num = data_len // core_sum
    _res = data_len % core_sum
    _base = (np.zeros(shape=core_sum).astype(int) + 1) * per_core_num
    for _ in range(_res):
        _base[_] += 1
    stop_core_index = np.cumsum(_base)

    divided_gene = [origin_gene[0:stop_core_index[0]]]
    divided_sequence = [origin_sequence[0:stop_core_index[0]]]
    divided_rbv_data = [origin_rbv_data[0:stop_core_index[0]]]
    for core_num in range(1, core_sum - 1):
        divided_gene.append(origin_gene[stop_core_index[core_num - 1]:stop_core_index[core_num]])
        divided_sequence.append(origin_sequence[stop_core_index[core_num - 1]:stop_core_index[core_num]])
        divided_rbv_data.append(origin_rbv_data[stop_core_index[core_num - 1]:stop_core_index[core_num]])
    divided_gene.append(origin_gene[stop_core_index[core_sum - 2]:])
    divided_sequence.append(origin_sequence[stop_core_index[core_sum - 2]:])
    divided_rbv_data.append(origin_rbv_data[stop_core_index[core_sum - 2]:])
    return origin_gene, origin_sequence, divided_gene, divided_sequence, divided_rbv_data, Activity


from lib.operation import *

ATCGset = {'A', 'T', 'U', 'C', 'G'}


def check_colname(df, col_names):
    for col_name in col_names:
        try:
            _ = df[col_name]
        except:
            print('Error: There is an issue with the column name! Please check again!!!!!!!!!!!!!!!!!!!!')
            exit()

def check_gene_norepeat(genenames):
    dictionary = set()
    for genename in genenames:
        if genename in dictionary or '_' in genename:
            print('ERROR: Duplicate gene names are not allowed! And gene names must not contain underscores!!!!!!!!!!!!!!!!!!!!')
            exit()
        dictionary.add(genename)

def check_seq_valid(sequences):
    for sequence in sequences:
        seqset = set(sequence.upper())
        if not seqset.issubset(ATCGset):
            print('Error: The 5\'UTR sequence contains letters other than A, T, U, C, or G!!!!!!!!!!!!!!!!!!!!')
            exit()

def check_seq_length_equal(sequences, len_limit):
    for sequence in sequences:
        if len(sequence) != len_limit:
            print('ERROR: The 5\'UTR sequence length must be fixed at {} nucleotides!!!!!!!!!!!!!!!!!!!!'.format(len_limit))
            exit()

def check_seq_length_grater(sequences, len_limit):
    for sequence in sequences:
        if len(sequence) < len_limit:
            print('ERROR: The 5\'UTR sequence length must be fixed at {} nucleotides!!!!!!!!!!!!!!!!!!!!'.format(len_limit))
            exit()

def check_act_valid(activities):
    try:
        activities = activities.astype(float)
    except:
        print('Error: The actual activity label is abnormal! Please check if it is in numerical form and must be greater than 0!!!!!!!!!!!!!!!!!!!!')
        exit()
    activities = np.sign(activities)
    if sum(activities) != len(activities):
        print('Error: The actual activity label is abnormal! Please check if it is in numerical form and must be greater than 0!!!!!!!!!!!!!!!!!!!!')
        exit()
    activities = np.log(activities.astype(float))

def check_scanned_tsv(tsv_data, limit_len=66):
    if len(tsv_data) != limit_len:
        print('Error: Please check if the TIA scan matrix length is fixed at 66!!!!!!!!!!!!!!!!!!!!')
        exit()

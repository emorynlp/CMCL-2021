import numpy as np

golden_pad_data = {}
with open('pad.tsv', 'r') as fr:
    for line in fr.readlines():
        fs = line.strip().split('\t')
        golden_pad_data[fs[0]] = [float(x) for x in fs[1:]]


def load_data(data_name):
    emotions = []
    with open('emotions.txt', 'r') as fr:
        emotions = [x.strip() for x in fr.readlines()]

    basic_emotions = ['joyful', 'trusting', 'afraid', 'surprised',
                        'sad', 'disgusted', 'angry', 'anticipating']
    em_with_pad = []
    em_wo_pad = []
    em_wo_pad_str = []
    em_golden_p_values = []
    em_golden_a_values = []
    em_golden_d_values = []
    for i, em in enumerate(emotions):
        if em in golden_pad_data.keys():
            em_with_pad.append(i)
            em_golden_p_values.append(golden_pad_data[em][0])
            em_golden_a_values.append(golden_pad_data[em][1])
            em_golden_d_values.append(golden_pad_data[em][2])
        else:
            em_wo_pad.append(i)
            em_wo_pad_str.append(em)

    print('len of em_with_pad: {}'.format(len(em_with_pad)))
    mtx_file_prefix = 'emb_avg_3layer_avg/EDQA_none.emb.tsv.emb'

    mtx_files = ['{}_run{}.npy'.format(mtx_file_prefix, i) for i in [1, 2, 3]]

    dist_matrix_array_0 = np.load(mtx_files[0])
    dist_matrix_array_1 = np.load(mtx_files[1])
    dist_matrix_array_2 = np.load(mtx_files[2])
    avg_mtx = (dist_matrix_array_0 + dist_matrix_array_1 + dist_matrix_array_2)/3
    assert avg_mtx.shape == (32, 256)

    X_train = avg_mtx[em_with_pad]
    X_test  = avg_mtx

    if data_name == 'p':
        y_train = np.array(em_golden_p_values)
    elif data_name == 'a':
        y_train = np.array(em_golden_a_values)
    elif data_name == 'd':
        y_train = np.array(em_golden_d_values)

    return X_train, y_train, X_test, emotions 

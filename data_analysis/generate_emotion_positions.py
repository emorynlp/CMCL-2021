import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
from sklearn.metrics.pairwise import cosine_similarity

basic_emotions = ['joyful', 'trusting', 'afraid', 'surprised',
                    'sad', 'disgusted', 'angry', 'anticipating']

primary_be_pairs = []
for i, _ in enumerate(basic_emotions):
    j = ((i + 1) % len(basic_emotions))
    primary_be_pairs.append((basic_emotions[i], basic_emotions[j]))

secondary_be_pairs = []
for i, _ in enumerate(basic_emotions):
    j = ((i + 2) % len(basic_emotions))
    secondary_be_pairs.append((basic_emotions[i], basic_emotions[j]))

third_be_pairs = []
for i, _ in enumerate(basic_emotions):
    j = ((i + 3) % len(basic_emotions))
    third_be_pairs.append((basic_emotions[i], basic_emotions[j]))

basic_emotions_pairs = primary_be_pairs + secondary_be_pairs + third_be_pairs


emotions = []
with open('emotions.txt', 'r') as fr:
    emotions = [x.strip() for x in fr.readlines()]

def get_avg_be(em_a, em_b, mtx):
    idx_a = emotions.index(em_a)
    idx_b = emotions.index(em_b)
    res = []
    for i in range(9):
        w_b = 0.1 * (i + 1)
        w_a = 1 - w_b
        res.append(w_a * mtx[idx_a] + w_b * mtx[idx_b])
    return res


if __name__ == '__main__':
    mtx_file_prefix = 'emb_avg_3layer_avg/EDQA_none.emb.tsv.emb'
    threshold = 0.0

    mtx_files = ['{}_run{}.npy'.format(mtx_file_prefix, i) for i in [1, 2, 3]]

    dist_matrix_array_0 = np.load(mtx_files[0])
    dist_matrix_array_1 = np.load(mtx_files[1])
    dist_matrix_array_2 = np.load(mtx_files[2])
    avg_mtx = (dist_matrix_array_0 + dist_matrix_array_1 + dist_matrix_array_2)/3
    #avg_mtx = dist_matrix_array_0
    assert avg_mtx.shape == (32, 256)

    be_interm_pos = {}
    for be_a, be_b in basic_emotions_pairs:
        avg_be_arr = get_avg_be(be_a, be_b, avg_mtx)
        key = '{}_{}'.format(be_a, be_b)
        be_interm_pos[key] = avg_be_arr

    em_pos = {}
    all_em_pos = [[[] for i in range(len(emotions))]]
    for i, em in enumerate(emotions):
        if em in basic_emotions:
            continue
        em_vec = np.array([avg_mtx[i]])
        max_k = ''
        max_score = 0
        max_i = -1
        for k, avg_be_arr in be_interm_pos.items():
            for idx, avg_be in enumerate(avg_be_arr):
                avg_be = np.array([avg_be])
                score = cosine_similarity(avg_be, em_vec)[0][0]
                if score > max_score:
                    max_score = score
                    max_k = k
                    max_i = idx

        if max_score >= threshold:
            em_pos[em] = (max_k, max_i, max_score)
        else:
            print('max score of emotion {} is less than TH'.format(em))

    for k, v in em_pos.items():
        print('{0}\t{1}\t{2}\t{3:.2f}'.format(k, v[0], v[1], v[2]))


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from stanfordcorenlp import StanfordCoreNLP
from random import shuffle, seed
import string
import torch
import torchvision.models as models
from PIL import Image
import skimage.io

import json
import numpy as np
import operator
from unidecode import unidecode
import h5py
import os
from glob import glob
import time
from tqdm import tqdm
from relation_to_adj_matrix import head_to_tree, tree_to_adj

path = "/data/home/wangyouze/stanford-corenlp/stanford-corenlp-4.2.2/"
nlp = StanfordCoreNLP(path)

def story_pro(annotations, params):
    print('len_annotations: ', len(annotations))

    image_feat_num = params['image_features']
    img_feat_num = json.load(open(image_feat_num, 'r'))
    print('len_feat_num', len(img_feat_num))
    count = 0
    story = []

    for i in tqdm(range(0, len(annotations), 5)):
    # for i in tqdm(range(0, 500, 5)):
        try:
            story_id = annotations[i][0]['story_id']

            img_id1, order1 = int(annotations[i][0]["photo_flickr_id"]), annotations[i][0][
                "worker_arranged_photo_order"]
            img_id2, order2 = int(annotations[i + 1][0]["photo_flickr_id"]), annotations[i + 1][0][
                "worker_arranged_photo_order"]
            img_id3, order3 = int(annotations[i + 2][0]["photo_flickr_id"]), annotations[i + 2][0][
                "worker_arranged_photo_order"]
            img_id4, order4 = int(annotations[i + 3][0]["photo_flickr_id"]), annotations[i + 3][0][
                "worker_arranged_photo_order"]
            img_id5, order5 = int(annotations[i + 4][0]["photo_flickr_id"]), annotations[i + 4][0][
                "worker_arranged_photo_order"]

            # if not (str(img_id1) in img_feat_num):
            #     count += 1
            #     print('img_id1:', img_id1)
            #     continue
            # else:
            #     img_str_id1 = str(img_id1)
            #
            # if not (str(img_id2) in img_feat_num):
            #     count += 1
            #     print('img_id2', img_id2)
            #     continue
            # else:
            #     img_str_id2 = str(img_id2)
            #
            # if not (str(img_id3) in img_feat_num):
            #     count += 1
            #     print('img_id3', img_id3)
            #     continue
            # else:
            #     img_str_id3 = str(img_id3)
            #
            # if not (str(img_id4) in img_feat_num):
            #     count += 1
            #     print('img_id4', img_id4)
            #     continue
            # else:
            #     img_str_id4 = str(img_id4)

            if not (str(img_id5) in img_feat_num):
                # print(str(img_id5))
                # print()
                count += 1
                print('img_id5', img_id5)
                print(count)
                continue
            else:
                img_str_id5 = str(img_id5)

            story1 = annotations[i][0]['text']
            # print('story1: ', story1)
            story2 = annotations[i + 1][0]['text']
            story3 = annotations[i + 2][0]['text']
            story4 = annotations[i + 3][0]['text']
            story5 = annotations[i + 4][0]['text']

            story1_rep = story1.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[', '[ ').replace(']', ' ]')
            story2_rep = story2.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[', '[ ').replace(']', ' ]')
            story3_rep = story3.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[', '[ ').replace(']', ' ]')
            story4_rep = story4.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[', '[ ').replace(']', ' ]')
            story5_rep = story5.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[', '[ ').replace(']', ' ]')

            # print('story1_rep: ', story1_rep)

            story1_taken = nlp.word_tokenize(story1)
            story2_taken = nlp.word_tokenize(story2)
            story3_taken = nlp.word_tokenize(story3)
            story4_taken = nlp.word_tokenize(story4)
            story5_taken = nlp.word_tokenize(story5)
            # print(story1_taken)

            if len(story1_taken) > params['max_length']:
                count += 1
                continue
            if len(story2_taken) > params['max_length']:
                count += 1
                continue
            if len(story3_taken) > params['max_length']:
                count += 1
                continue
            if len(story4_taken) > params['max_length']:
                count += 1
                continue
            if len(story5_taken) > params['max_length']:
                count += 1
                continue

            sent1 = nlp.dependency_parse(story1_rep)
            # print('sent1: ', sent1)
            sent2 = nlp.dependency_parse(story2_rep)
            sent3 = nlp.dependency_parse(story3_rep)
            sent4 = nlp.dependency_parse(story4_rep)
            sent5 = nlp.dependency_parse(story5_rep)

            depen_list = [(sent1, order1), (sent2, order2), (sent3, order3), (sent4, order4), (sent5, order5)]
            depen_list = sorted(depen_list, key=operator.itemgetter(1))

            depen_list = [depen_list[0][0], depen_list[1][0], depen_list[2][0], depen_list[3][0]]
            # print('depen_list: ', depen_list)

            # label = annotations[i]['split']

            story_rep_list = [(story1_rep, order1), (story2_rep, order2), (story3_rep, order3),
                              (story4_rep, order4), (story5, order5)]
            story_rep_list = sorted(story_rep_list, key=operator.itemgetter(1))
            story_rep_four = [story_rep_list[0][0], story_rep_list[1][0], story_rep_list[2][0], story_rep_list[3][0]]
            story_rep_last = [story_rep_list[4][0]]
            # print('story_rep_last:', story_rep_last)

            story_list = [(story1, order1), (story2, order2), (story3, order3), (story4, order4), (story5, order5)]
            story_list = sorted(story_list, key=operator.itemgetter(1))

            story_token_list = [(story1_taken, order1), (story2_taken, order2), (story3_taken, order3),
                                (story4_taken, order4), (story5_taken, order5)]
            story_token_list = sorted(story_token_list, key=operator.itemgetter(1))

            story_token_list = [story_token_list[0][0], story_token_list[1][0], story_token_list[2][0],
                                story_token_list[3][0], story_token_list[4][0]]

            img_id_list = [(img_id1, order1), (img_id2, order2), (img_id3, order3), (img_id4, order4), (img_str_id5, order5)]
            img_id_list = sorted(img_id_list, key=operator.itemgetter(1))

            ordered_stories_four = [story_list[0][0], story_list[1][0], story_list[2][0], story_list[3][0]]
            ordered_stories_last = story_list[4][0]
            # print(ordered_stories_last)
            order_last_img_id = img_id_list[4][0]
            # print(order_last_img_id)
            story.append({'story_id': story_id,
                          'story_token_list': story_token_list,
                          'stories_four': story_rep_four,
                          # 'sent_rep': story_rep_four,
                          'sent_depen': depen_list,
                          'stories_last': ordered_stories_last,
                          'last_img_id': order_last_img_id,
                          'split': annotations[i][0]['split']})
        except json.decoder.JSONDecodeError:
            continue
    # print(story[1])
    nlp.close()
    with open('story10.json', 'w') as ff:
        json.dump(story, ff)
    # json.dump(story, open('story.json', 'w'))
    ff.close()

    return story


def build_vocab(storys, params):
    count_thr = params['word_count_threshold_test']

    counts = {}
    # for story in storys:
    #     print("*"*50)
    #     token = story['story_token_list'][:4]
    #     for sentence in token:
    #         print("#"*50)
    #         print('sentence', sentence)
    #         for w in sentence:
    #             print('w', w)
    #             counts[w] = counts.get(w, 0) + 1

    for story in storys:
        last_token = story['story_token_list'][4]
        # print('last_token: ', last_token)
        for w in last_token:
            # print('w: ', w)
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    # print('top words and their counts')
    # print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    # print('total_words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    # print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    # print('number of words in vocab would be %d' % (len(vocab), ))
    # print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for story in storys:
        token = story['story_token_list']
        txt = token[-1]
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    # print('max length sentence in raw data: ', max_len)
    # print('sentence length distribution (count, numbers of words):')
    sum_len = sum(sent_lengths.values())
    # for i in range(max_len+1):
    #     print('%2d: %10d    %f%%' % (i, sent_lengths.get(i, 0),sent_lengths.get(i, 0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        vocab.append('UNK')
    # vocab['PADDING'] = 0

    # print('vocab: ', vocab)
    for story in storys:
        story['story_end'] = []
        txt = story['story_token_list'][-1]
        last = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
        story['story_end'].append(last)

    return vocab

def build_src_vocab(storys, params):
    count_thr = params['word_count_threshold']

    counts = {}
    for story in storys:
        # print("*"*50)
        token = story['story_token_list'][:4]
        for sentence in token:
            # print("#"*50)
            # print('sentence', sentence)
            for w in sentence:
                # print('w', w)
                counts[w] = counts.get(w, 0) + 1

    # for story in storys:
    #     last_token = story['story_token_list'][4]
    #     print('last_token: ', last_token)
    #     for w in last_token:
    #         print('w: ', w)
    #         counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    # print('top words and their counts')
    # print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    # print('total_words:', total_words)


    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    # print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    # print('number of words in vocab would be %d' % (len(vocab),))
    # print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for story in storys:
        token = story['story_token_list']
        txt = token[-1]
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    # print('max length sentence in raw data: ', max_len)
    # print('sentence length distribution (count, numbers of words):')
    sum_len = sum(sent_lengths.values())
    # for i in range(max_len + 1):
    #     print('%2d: %10d    %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        vocab.append('UNK')

    # vocab['PADDING'] = 0

    # print('vocab: ', vocab)
    # for story in storys:
    #     story['story_end'] = []
    #     txt = story['story_token_list'][-1]
    #     last = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
    #     story['story_end'].append(last)

    return vocab

def encode_story_four(storys, params, src_wtoi):
    max_length = 40
    first = []
    second = []
    third = []
    four = []

    for i, story in enumerate(storys):

        # max_length = params['max_length']
        max_length = 40
        sent_insts = story['stories_four']
        # print('word_insts: ', sent_insts)

        for j, sents in enumerate(sent_insts):

            if j == 0:
                sent_lsit = np.zeros((40), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    # print('word: ', word)
                    # print('src_wtoi: ', src_wtoi)
                    if word in src_wtoi:
                        if i < max_length:
                            sent_lsit[i] = (src_wtoi[word])
                    else:
                        continue
                # print('sent_insts_len:', len(sent_insts))
                # sent_lsit = np.array(sent_lsit)
                first.append(sent_lsit)
                # print('first_encoder: ', first[0])
                # print('first_len: ', len(first[0]))

            elif j == 1:
                sent_lsit = np.zeros((41), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    # print('word: ', word)
                    # print('src_wtoi: ', src_wtoi)
                    if word in src_wtoi:
                        if i < max_length:
                            sent_lsit[i] = (src_wtoi[word])
                    else:
                        continue
                # print('sent_insts_len:', len(sent_insts))
                # sent_lsit = np.array(sent_lsit)
                second.append(sent_lsit)
                # print('second_encoder: ', second[0])
                # print('second_len: ', len(second[0]))
            elif j == 2:
                sent_lsit = np.zeros((42), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    # print('word: ', word)
                    # print('src_wtoi: ', src_wtoi)
                    if word in src_wtoi:
                        if i < max_length:
                            sent_lsit[i] = (src_wtoi[word])
                    else:
                        continue
                # print('sent_insts_len:', len(sent_insts))
                sent_lsit = np.array(sent_lsit)
                third.append(sent_lsit)
                # print('third_encoder: ', third[0])
                # print('third_len: ', len(third[0]))
            else:
                sent_lsit = np.zeros((43), dtype='uint32')
                for i, word in enumerate(sents.split(' ')):
                    # print('word: ', word)
                    # print('src_wtoi: ', src_wtoi)
                    if word in src_wtoi:
                        if i < max_length:
                            sent_lsit[i] = (src_wtoi[word])
                    else:
                        continue
                # print('sent_insts_len:', len(sent_insts))
                # sent_lsit = np.array(sent_lsit)
                four.append(sent_lsit)
                # print('four_encoder: ', four[0])
                # print('four_len: ', len(four[0]))


    # print('first: ', first)
    return first, second, third, four


def encode_story_last(storys, params, wtoi):
    # print("*"*50)

    max_length = 40
    N = len(storys)
    M = len(storys)
    # print('M', M)

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    story_counter = 0
    counter = 1

    for i, story in enumerate(storys):
        # print('#'*50)

        n = 1
        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(story['story_end']):
            # print('s', s)
            label_length[story_counter] = min(max_length, len(s))
            story_counter += 1
            for k, w in enumerate(s):
                # print('w_last ', w)
                if k < max_length:
                    Li[j, k] = wtoi[w]


        label_arrays.append(Li)
        # print('label_arrays: ', label_arrays)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)
    # print("%"*50)
    # print('type_L: ', type(L))
    # print('M: ', M)
    # print('L.shape[0]: ', L.shape[0])
    assert L.shape[0] == M, 'length don\'t match that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    # print('encoded story to array of size', L.shape)
    return L, label_start_ix, label_end_ix, label_length

def relation_to_adj_matrix(relation, sent):

    # depend = storys['sent_depen']

    head = head_to_tree(relation)

    if sent == 'sent1':
        adj_mat = tree_to_adj(40, head, sent)
    if sent == 'sent2':
        adj_mat = tree_to_adj(41, head, sent)
    if sent == 'sent3':
        adj_mat = tree_to_adj(42, head, sent)
    if sent == 'sent4':
        adj_mat = tree_to_adj(43, head, sent)

    return adj_mat

def adj_matrix(storys):

    adj1, adj2, adj3, adj4 = [], [], [], []

    for i, story in enumerate(storys):
        # print('%'*50)

        depend = story['sent_depen']
        # print('depend_0: ', depend[0])

        sent1 = depend[0]
        sent2 = depend[1]
        sent3 = depend[2]
        sent4 = depend[3]

        # print('sent1: ', sent1)
        adj1.append(relation_to_adj_matrix(sent1, 'sent1'))
        # print('adj1: ', len(adj1))
        adj2.append(relation_to_adj_matrix(sent2, 'sent2'))
        adj3.append(relation_to_adj_matrix(sent3, 'sent3'))
        adj4.append(relation_to_adj_matrix(sent4, 'sent4'))

    return adj1, adj2, adj3, adj4

def main(params):
    file = open(params['input_json'], 'r')
    data = json.load(file)
    anno = data
    print(len(data))
    print(anno[1])
    story = story_pro(anno, params)
    file.close()
    # file = open('story5.json', 'r')
    # story = json.load(file)
    # file.close()

    vocab = build_vocab(story, params)
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    src_vocab = build_src_vocab(story, params)
    src_itow = {i: w for i, w in enumerate(src_vocab)}
    src_wtoi = {w: i for i, w in enumerate(src_vocab)}

    first, second, third, four = encode_story_four(story, params, src_wtoi)
    # print('first_len: ', len(first))
    f_fe = h5py.File(params['output_h5_fe'] + '_label1.h5', 'w')
    f_fe.create_dataset('sent1', dtype='uint32', data=first)

    f_fe.create_dataset('sent2', dtype='uint32', data=second)
    f_fe.create_dataset('sent3', dtype='uint32', data=third)
    f_fe.create_dataset('sent4', dtype='uint32', data=four)
    f_fe.close()
    print('Down f_fe')

    adj1, adj2, adj3, adj4 = adj_matrix(story)
    # # print('adj1_len: ', len(adj1))
    #
    f_adj = h5py.File(params['output_h5_adj'] + '_label1.h5', 'w')
    f_adj.create_dataset('adj1', dtype='uint32', data=adj1)
    f_adj.create_dataset('adj2', dtype='uint32', data=adj2)
    f_adj.create_dataset('adj3', dtype='uint32', data=adj3)
    f_adj.create_dataset('adj4', dtype='uint32', data=adj4)
    f_adj.close()
    print('Down f_adj')



    L, label_start_ix, label_end_ix, label_length = encode_story_last(story, params, wtoi)
    # print('L: ', L)


    # print('first_len: ', len(first))
    print('L_len:', len(L))
    # N = len(anno)
    f_lb = h5py.File(params['output_h5']+'_label1.h5', 'w')
    f_lb.create_dataset('labels', dtype='uint32', data=L)
    f_lb.create_dataset('label_start_ix', dtype='uint32', data=label_start_ix)
    f_lb.create_dataset('label_end_ix', dtype='uint32', data=label_end_ix)
    f_lb.create_dataset('label_length', dtype='uint32', data=label_length)
    f_lb.close()
    print('Down f_lb')

    story_four=[]
    for i, imgs in enumerate(story):
        story_four.append(imgs['stories_four'])
    with open(params['output_story_four'], 'w') as f_four:
        json.dump(story_four, f_four)
    f_four.close()
    print('Down f_four')

    out = {}
    out['src_word_to_ix'] = src_wtoi
    out['src_ix_to_word'] = src_itow
    out['tgt_word_to_ix'] = wtoi
    out['tgt_ix_to_word'] = itow

    out['images'] = []
    for i, img in enumerate(story):
        jimg = {}
        jimg['split'] = img['split']
        jimg['id'] = img['last_img_id']
        jimg['story_id'] = img['story_id']
        jimg['story_end'] = img['stories_last']
        # jimg['story_four'] = img['stories_four']
        # jimg['sent_rep'] = img['sent_rep']
        # jimg['sent_depen'] = img['sent_depen']
        out['images'].append(jimg)

    with open(params['output_json'], 'w') as ff:
        json.dump(out, ff)
    ff.close()

    # json.dump(out, open(params['output_json'], 'w'))
    print('wrote', params['output_json'])
    file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='annotation.json', help='')
    parser.add_argument('--output_json', default='data_res10.json', help='')
    parser.add_argument('--output_h5', default='data_tgt10', help='')
    parser.add_argument('--output_story_four', default='data_four10.json', help='')
    parser.add_argument('--output_h5_fe', default='data_src10', help='')
    parser.add_argument('--output_h5_adj', default='data_adj10', help='')
    parser.add_argument('--image_root', default='/home/chuan/HC/AREL-master', help='')
    parser.add_argument('--image_features', default='feat_num.json', help='')

    parser.add_argument('--max_length', default=20, type=int, help='')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='')
    parser.add_argument('--word_count_threshold_test', default=1, type=int, help='')

    args = parser.parse_args()
    params = vars(args)

    main(params)

# nlp.close()
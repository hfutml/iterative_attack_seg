import numpy as np
import json
import torch
import random
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

        adj1 = np.array(adj1, dtype=np.int)
        adj2 = np.array(adj2, dtype=np.int)
        adj3 = np.array(adj3, dtype=np.int)
        adj4 = np.array(adj4, dtype=np.int)

    return torch.from_numpy(adj1), torch.from_numpy(adj2), torch.from_numpy(adj3), torch.from_numpy(adj4)

def encode_story_four(storys, src_wtoi):
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
                        if i < max_length:
                            sent_lsit[i] = random.randint(0, len(src_wtoi.keys()) - 1)
                        # continue
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
                        if i < max_length:
                            sent_lsit[i] = random.randint(0, len(src_wtoi.keys()) - 1)
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
                        if i < max_length:
                            sent_lsit[i] = random.randint(0, len(src_wtoi.keys()) - 1)
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
                        if i < max_length:
                            sent_lsit[i] = random.randint(0, len(src_wtoi.keys()) - 1)
                        continue
                # print('sent_insts_len:', len(sent_insts))
                # sent_lsit = np.array(sent_lsit)
                four.append(sent_lsit)
                # print('four_encoder: ', four[0])
                # print('four_len: ', len(four[0]))

    first = np.array(first, dtype=np.int_)
    second = np.array(second, dtype=np.int_)
    third = np.array(third, dtype=np.int_)
    four = np.array(four, dtype=np.int_)
    # print('first: ', first)
    return torch.from_numpy(first), torch.from_numpy(second), torch.from_numpy(third), torch.from_numpy(four)

def story_pro(story, nlp):
    story_token_list = []
    story_depen_list = []
    story_four_list = []
    story1 = story['sent1']
    # print('story1: ', story1)
    story2 = story['sent2']
    story3 = story['sent3']
    story4 = story['sent4']

    story1_rep = story1.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[',
                                                                                                         '[ ').replace(
        ']', ' ]')
    story2_rep = story2.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[',
                                                                                                         '[ ').replace(
        ']', ' ]')
    story3_rep = story3.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[',
                                                                                                         '[ ').replace(
        ']', ' ]')
    story4_rep = story4.replace(' .', '').replace(' !', '').replace(" ?", '').replace(" '", ' ').replace('[',
                                                                                                         '[ ').replace(
        ']', ' ]')
    story_four_list.append(story1_rep)
    story_four_list.append(story2_rep)
    story_four_list.append(story3_rep)
    story_four_list.append(story4_rep)

    story1_taken = nlp.word_tokenize(story1)
    story2_taken = nlp.word_tokenize(story2)
    story3_taken = nlp.word_tokenize(story3)
    story4_taken = nlp.word_tokenize(story4)
    # print(story1_taken)
    story_token_list.append(story1_taken)
    story_token_list.append(story2_taken)
    story_token_list.append(story3_taken)
    story_token_list.append(story4_taken)


    sent1 = nlp.dependency_parse(story1_rep)
    sent2 = nlp.dependency_parse(story2_rep)
    sent3 = nlp.dependency_parse(story3_rep)
    sent4 = nlp.dependency_parse(story4_rep)

    story_depen_list.append(sent1)
    story_depen_list.append(sent2)
    story_depen_list.append(sent3)
    story_depen_list.append(sent4)
    story['stories_four'] = story_four_list
    story['story_token_list'] = story_token_list
    story['sent_depen'] = story_depen_list
    return [story]

def gen_adj(story_text, nlp):
    story = story_pro(story_text, nlp)
    src_vocab = json.load(open("/data/home/wangyouze/projects/others/MMT/data/VIST-E/data_res10.json"))['src_word_to_ix']
    # src_vocab = json.load(open("/mfs/wenbo/wyz_work/MGCL/adversarial_attack/multimodal_attack/LSMDC/data-20/data_res10.json"))['src_word_to_ix']
    src_wtoi = {w: i for i, w in enumerate(src_vocab)}
    first, second, third, four = encode_story_four(story, src_wtoi)

    return first, second, third, four
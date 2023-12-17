"""
修改bert_attack, 在每次计算候选词评分的时候进行图片攻击
以gold ending作为监督信号
"""

import torch
import torch.nn as nn
import copy
from transformers import BatchEncoding
import torch.nn.functional as F
import numpy as np
# import modules.losses as losses
import sys
sys.path.append('/data/home/wangyouze/projects/others/MMT/')
import utils.misc as utils
import sacrebleu
import random
from scipy import spatial
from spellchecker import SpellChecker
import modules.losses as losses

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·', '[]', '[', ']']
filter_words = set(filter_words)




class BertAttackFusion():
    def __init__(self,  ref_net, tokenizer, image_attacker, words_dict,  opt, device, cls=True):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.image_attacker = image_attacker
        self.opt = opt
        self.words_dict = None
        self.embed = torch.load("/data/home/wangyouze/projects/others/MMT/data/VIST-E/embedding/embedding_enc.pt")
        self.device = device
        self.words_dict = words_dict


    def attack(self, net, images, no_attack_ending, info, k=10, num_perturbation=2, threshold_pred_score=3.0, max_length=80, batch_size=1):
        device = self.ref_net.device
        print('num_perturbation=', num_perturbation)
        gold_story_end = info['story_end'].tolist()

        texts = list(info['src1'])[0] + ' [SEP] ' + list(info['src2'])[0] + ' [SEP] ' + list(info['src3'])[
                0] + ' [SEP] ' + list(info['src4'])[0]

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length,
                                     return_tensors='pt').to(device)

        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, k, -1)  # seq-len k

        # original state
        gold_ending_seq_ = info['gts'][0][0].tolist()
        gold_ending_len = len(info['story_end'][0].split(' '))
        gold_ending_seq = []
        for t in range(40):
                if t < gold_ending_len:
                    gold_ending_seq.append(gold_ending_seq_[t])
                else:
                    gold_ending_seq.append(2256)
        gold_ending_seq = np.array(gold_ending_seq)
        gold_ending_seq = torch.LongTensor(gold_ending_seq).squeeze(0)

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        criterion = torch.nn.CrossEntropyLoss()
        # LMC = losses.LanguageModelCriterion()
        final_adverse = []
        for i, text in enumerate([texts]):
            # word importance eval
            important_scores = self.get_important_scores(images, info, net, gold_ending_seq, batch_size, max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0
            attack_success = False
            for top_index in list_of_index:
                if change >= num_perturbation or attack_success == True:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if tgt_word == "[SEP]":
                    continue
                if keys[top_index[0]][0] > max_length - 2:
                    continue
                substitutes_character = self.selectBug(tgt_word)

                substitutes_word = word_predictions[i, keys[top_index[0]+1][0]:keys[top_index[0]+1][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]+1][0]:keys[top_index[0]+1][1]]

                substitutes_word = get_substitues(substitutes_word, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             threshold_pred_score)

                substitutes = substitutes_character + substitutes_word
                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]

                for i_, substitute in enumerate(substitutes):

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''

                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))

                # replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                #                                     max_length=max_length, return_tensors='pt').to(device)
                info_replace = copy.deepcopy(info)
                replace_output_list = []
                image_adv_list = []
                perturbation_list = []
                loss_list = []
                sents_f_list = []

                for j in range(len(replace_texts)):
                    # masked_text_tokens_input = self.tokenizer.decode(replace_text_input['input_ids'][j])
                    replace_text = replace_texts[j].split('[SEP]')
                    # text = masked_text_tokens_input.replace('[CLS]', '').split('[SEP]')
                    # print('text:', text)
                    info_replace['src1'] = [replace_text[0]]
                    info_replace['src2'] = [replace_text[1]]
                    info_replace['src3'] = [replace_text[2]]
                    info_replace['src4'] = [replace_text[3]]

                    # text_adv_output = net(images, info_replace, mode='sample')[1]
                    # text_adv_embeds = text_adv_output[:, 0, :].detach()

                    image_attack = self.image_attacker.attack(images, 10)

                    for _ in range(10):
                        image_diversity = next(image_attack)
                        replace_output = net(image_diversity, info_replace)[1]

                        labels = gold_ending_seq.unsqueeze(0).to(device)
                        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), gold_ending_seq)))
                        masks = np.zeros([1, 1, 40], dtype='float32')
                        for ix, row in enumerate(masks):
                            row[:nonzeros[ix]] = 1
                        masks = torch.from_numpy(masks).to(device)
                        # loss = LMC(replace_output, labels, masks)
                        loss = criterion(replace_output.squeeze(0), labels.squeeze(0))

                        loss.backward()
                    # replace_output = net(images, info_replace)
                    replace_output_list.append(replace_output)
                    images_adv, perturbation = next(image_attack)
                    image_adv_list.append(images_adv)
                    perturbation_list.append(perturbation)
                    loss_list.append(loss.item())

                    with torch.no_grad():
                        seq_f, seqLogprobs_f = net(images_adv, info_replace, opt=self.opt)
                    sents_f = utils.decode_sequence(net.MMT.vocab, seq_f)[0]
                    sents_f_list.append(sents_f)
                    bleu_1 = sacrebleu.sentence_bleu(sents_f, gold_story_end).score
                    bleu_2 = sacrebleu.sentence_bleu(no_attack_ending, gold_story_end).score

                    if bleu_1 <= bleu_2 / 2:
                        attack_success = True
                        break
                if attack_success:
                    candidate_idx = len(loss_list) - 1
                else:
                    loss = torch.tensor(loss_list)
                    candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]
                images_adv = image_adv_list[candidate_idx]
                perturbation = perturbation_list[candidate_idx]
                ending_f = sents_f_list[candidate_idx]
                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))



        return final_adverse, images_adv, perturbation, ending_f, attack_success
    def selectBug(self, original_word):
        bugs = self.generateBugs(original_word, self.embed, typo_enabled=True)

        res = []
        for k, v in bugs.items():
            res.append(v)

        return res

    def generateBugs(self, word, glove_vectors, sub_w_enabled=True, typo_enabled=True):
        bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}

        if (len(word) <= 2):
            return bugs

        bugs["insert"] = self.bug_insert(word)
        bugs["delete"] = self.bug_delete(word)
        bugs["swap"] = self.bug_swap(word)
        bugs["sub_C"] = self.bug_sub_C(word)
        if (typo_enabled):
            bugs["typoW"] = self.bug_typo(bugs['swap'])

        if (not sub_w_enabled):
            return bugs
        bugs["sub_W"] = self.bug_sub_W(word, glove_vectors)

        return bugs

    def bug_typo(self, word):
        spell = SpellChecker(distance=10)
        candidates = spell.candidates(word)
        if candidates == None:
            candidates = {word}
        chosen_candidate_typo = random.choice(list(candidates))
        return chosen_candidate_typo

    def bug_insert(self, word):
        if (len(word) >= 6):
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        res = res[0:point] + " " + res[point:]
        return res

    def bug_delete(self, word):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        # print("hi")
        # print(res[7:])
        return res

    def bug_swap(self, word):
        if (len(word) <= 4):
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        # print(points)
        a = points[0]
        b = points[1]

        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_sub_C(self, word):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def bug_sub_W(self, word, glove_vectors):
        # if self.words_dict[word] not in glove_vectors:
        #     return word
        if word in self.words_dict.keys():
            word_id = self.words_dict[word]
        else:
            word_id = random.randint(0, len(self.words_dict))

        closest_neighbors = self.find_closest_words(glove_vectors[word_id, :], glove_vectors, word)[1:6]

        return random.choice(closest_neighbors)
        # return closest_neighbors # Change later

    def find_closest_words(self, point, glove_vectors, w):
        res = []
        point = point.cpu().detach().numpy()
        for k, v in self.words_dict.items():
            # print(v)
            embed = glove_vectors[v, :].cpu().detach().numpy()
            tmp = spatial.distance.euclidean(embed, point)
            res.append([k, tmp])
        res = sorted(res, key=lambda x:x[1])
        return [v[0] if v[0] != w else ' ' for v in res]
        # return sorted(res, key=lambda x:x[1])
        # return sorted(self.words_dict.keys(), key=lambda word: spatial.distance.euclidean(glove_vectors(torch.LongTensor([self.words_dict[word]])), point))

    def get_key_neighbors(self):
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl",
            "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm",
            "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }

        # By visual proximity
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'

        return neighbors

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            if words[i] == '[SEP]':
                continue
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    def check_out(self, text):
        tmp = []
        if len(text.split(' ')) > 40:
            for i, t in enumerate(text.split(' ')):
                if i<40:
                    tmp.append(t)
            text = ' '.join(tmp)
        return text


    def get_important_scores(self, image, info , net, gold_ending_seq, batch_size, max_length):
        # origin_embed = origin_output[1].flatten(1).detach()
        device = net.device
        text = list(info['src1'])[0] + ' [SEP] ' + list(info['src2'])[0] + ' [SEP] ' + list(info['src3'])[
            0] + ' [SEP] ' + list(info['src4'])[0]

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True,
                                               max_length=max_length*4,
                                               return_tensors='pt').to(device)
            # image_batch = image.repeat(len(masked_text_input.input_ids), 1, 1, 1)
            masked_output_batch = []
            for j in range(len(masked_text_input.input_ids)):
                masked_text_tokens_input = self.tokenizer.decode(masked_text_input['input_ids'][j])
                text = masked_text_tokens_input.split('[SEP]')
                text_0 = self.check_out(text[0])
                # text_1 = self.check_out(text[1])
                # text_2 = self.check_out(text[2])
                # text_3 = self.check_out(text[3])
                info['src1'] = [text_0.replace('[CLS]', '')]
                info['src2'] = [text[1]]
                info['src3'] = [text[2]]
                info['src4'] = [text[3].replace('[PAD]', '')]

                masked_output = net(image,info)
                masked_output_batch.append(masked_output[1].squeeze(0))
            masked_output = torch.stack(masked_output_batch)
            # if self.cls:
            #     masked_embed = masked_output[:, 0, :].detach()
            # else:
            # masked_embed = masked_output.flatten(1).detach()
            masked_embed = masked_output[:, 0, :].detach()
            masked_embeds.append(masked_embed)

        masked_embeds = torch.cat(masked_embeds, dim=0)

        # criterion = torch.nn.KLDivLoss(reduction='none')
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        label = torch.zeros(len(masked_texts), masked_embeds.shape[-1]).scatter_(1, gold_ending_seq.repeat(len(masked_texts), 1), 1 ).to(device)
        import_scores = criterion(masked_embeds, label)

        return import_scores



def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
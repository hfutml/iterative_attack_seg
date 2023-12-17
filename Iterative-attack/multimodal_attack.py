
from stanfordcorenlp import StanfordCoreNLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from process_text import gen_adj
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained("/data/home/wangyouze/projects/github/bert-base-uncased/")

path = '/data/home/wangyouze/stanford-corenlp/stanford-corenlp-4.2.2'
nlp = StanfordCoreNLP(path)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


class combineNet(nn.Module):
    def __init__(self, resnet, MMT, device):
        super(combineNet, self).__init__()
        self.resnet = resnet
        self.MMT = MMT

        self.device = device
        self.mean = mean.unsqueeze(1).unsqueeze(2).to(self.device)
        self.std = std.unsqueeze(1).unsqueeze(2).to(self.device)

        self.info = json.load(open("/data/home/wangyouze/projects/others/MMT/data/VIST-E/data_res10.json"))
        self.src_ix_to_word = self.info['src_word_to_ix']
    def convert_to_bert_ids(self, story):

        seq_len = 40
        sent1 = np.zeros((1, seq_len + 2), dtype=np.int_)
        sent2 = np.zeros((1, seq_len + 2), dtype=np.int_)
        sent3 = np.zeros((1, seq_len + 2), dtype=np.int_)
        sent4 = np.zeros((1, seq_len + 2), dtype=np.int_)

        def sent_to_bert_ids(input_sent):
            sent = bert_tokenizer.tokenize(input_sent)
           
            sent = bert_tokenizer.convert_tokens_to_ids(sent)[:40]
            for _ in range(40-len(sent)):
                sent.append(2256)
            try:
                tail = sent.index(0)
                sent.insert(tail, bert_tokenizer.convert_tokens_to_ids('[SEP]'))
            except ValueError:
                sent.append(bert_tokenizer.convert_tokens_to_ids('[SEP]'))

            sent.insert(0, bert_tokenizer.convert_tokens_to_ids('[CLS]'))
            sent = np.array(sent)
            return sent
        
        src1 = sent_to_bert_ids(story['sent1'])
        # sent1 = src1
        src2 = sent_to_bert_ids(story['sent2'])
        
        src3 = sent_to_bert_ids(story['sent3'])
       
        src4 = sent_to_bert_ids(story['sent4'])

        return src1[np.newaxis,:], src2[np.newaxis,:], src3[np.newaxis,:], src4[np.newaxis,:]

    def forward(self, img, info, opt={}, need_grad=True):
        sent1 = list(info['src1'])[0]
        sent2 = list(info['src2'])[0]
        sent3 = list(info['src3'])[0]
        sent4 = list(info['src4'])[0]
        # masks = info['mask']
        # labels = info['label']
        story = {'sent1':sent1, 'sent2':sent2, 'sent3':sent3, 'sent4':sent4}
        src1, src2, src3, src4 = self.convert_to_bert_ids(story)
        src1, src2, src3, src4 = torch.from_numpy(src1.astype(int)).cuda(), torch.from_numpy(src2.astype(int)).cuda(), torch.from_numpy(src3.astype(int)).cuda(), torch.from_numpy(src4.astype(int)).cuda()


        x = self.resnet(img.unsqueeze(0))
        x = x.permute(0, 2, 3, 1)
        conv_feats = x.view(-1, x.shape[-1])
        conv_feats = conv_feats.unsqueeze(0)
       

        conv_masks = np.zeros(conv_feats.shape[:2], dtype=np.bool_)  # size(batch, numbers_att)
        # print('data[att_masks]: ', data['att_masks'].shape)
        for i in range(len(conv_feats)):
            conv_masks[i, :conv_feats[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if conv_masks.sum() == conv_masks.size:
            # print('att_batch is None')
            conv_masks = None
    
      
        seq, seqLogprobs = self.MMT._sample(conv_feats, conv_masks, src1, src2, src3, src4, opt=opt)
          

        return seq, seqLogprobs

class MultiModalAttacker(nn.Module):
    def __init__(self, net, image_attacker, text_attacker, device, cls=True):
        super(MultiModalAttacker, self).__init__()
       
        self.net = net
        self.image_attacker = image_attacker
        self.text_attacker = text_attacker
        self.device = device
        self.cls = cls

        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.crossEntropy = nn.CrossEntropyLoss()
    def forwrd(self, images, info, no_attack_ending, num_iters=20, k=10, max_len=40, alpha=3.0):
            # origin_output = self.net(images, info, mode='sample')[1]
        print('origin_text: ', list(info['src1'])[0] + ' [SEP] ' + list(info['src2'])[0] + ' [SEP] ' + list(info['src3'])[
            0] + ' [SEP] ' + list(info['src4'])[0])

        # with torch.no_grad():
        text_adv, image_adv, perturbation, ending_f, attack_success = self.text_attacker.attack(self.net, images, no_attack_ending, info, k)
        print('attack_context:', text_adv[0])

        return image_adv, perturbation, text_adv, ending_f, attack_success


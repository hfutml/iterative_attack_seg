'''
image attack:PGD,
text attack:BerttAttack

图片加载方式修改为npy加载
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import matplotlib
import cv2
matplotlib.use('Agg')

import skimage.io
from PIL import Image
from skimage.transform import resize

from dataloader_new import *

from image_attack import ImageAttacker
from textBugger_new_2 import BertAttackFusion
from multimodal_attack import MultiModalAttacker, combineNet
from tokenization_bert import BertTokenizer
from transformers import BertForMaskedLM

import sys
sys.path.append('/data/home/wangyouze/projects/others/MMT/')
import utils.misc as utils
from utils import opts
from model import MMT as MMT_modal
import time
import numpy as np
from dataloader_new import *
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(2022)
np.random.seed(200)
torch.manual_seed(200)
torch.cuda.manual_seed(2022)

def attack(opt):
    loader = DataLoader(opt, split='test')
    # while True:
    #     info = loader.get_batch('test')

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.vocab = loader.get_vocab()
    opt.src_vocab = loader.get_src_vocab()
    MMT = MMT_modal(opt).cuda()
    del opt.vocab
   
    MMT.load_state_dict(torch.load(opt.model))
    MMT.cuda()

    resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
    # Remove linear and pool layers (since we're not doing classification)
    modules = list(resnet.children())[:-2]
    resnet = nn.Sequential(*modules)
    resnet = resnet.cuda()
    resnet.eval()

    model = combineNet(resnet, MMT, device=device)
    model.cuda()
    model.resnet.eval()
    model.MMT.eval()
   
    
    tokenizer = BertTokenizer.from_pretrained("/data/home/wangyouze/MultimodalAttack/bert-base-uncased/")
    ref_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    ref_model = ref_model.to(device)
    ref_model.eval()

    words_dict = json.load(open("/data/home/wangyouze/projects/others/MMT/data/VIST-E/data_res10.json"))['src_word_to_ix']
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_attacker = ImageAttacker(preprocess=images_normalize, bounding=(0, 1), cls=False)
    text_attacker = BertAttackFusion(ref_model, tokenizer, image_attacker=image_attacker, words_dict=words_dict, opt=opt, device=device,  cls=False)
    
    multi_attacker = MultiModalAttacker(model, image_attacker, text_attacker, device)

    split = 'test'
    i = 0
    f = open("/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT.txt", 'w', encoding='utf-8')
    res = []
    while True:
        i += 1
        print('i=',i)
        info = loader.get_batch(split)

        print('-----------------------------------------------')

        story_id = info['story_id'][0]
        print('story_id:', info['story_id'][0])

        I = info['image'].squeeze(0)
        story_gold_end = info['story_end'][0]
        print('gold_ending:', story_gold_end)

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = torch.tensor(I).permute(2, 1, 0).float().cuda()
        img = I.clone()

        origin_text = list(info['src1'])[0] + ' [SEP] ' + list(info['src2'])[0] + ' [SEP] ' + \
                      list(info['src3'])[0] + ' [SEP] ' + list(info['src4'])[0]
        with torch.no_grad():
            seq, seqLogprobs = model(I, info, opt=opt)

        sents = utils.decode_sequence(model.MMT.vocab, seq)[0]
        print('no attack ending:', sents)
        attack_start = time.time()
        img_adv, perturbation, text_adv, ending_f, attack_success = multi_attacker.forwrd(img, info, sents,
                                                   num_iters=20, alpha=3.0)
        
        print('attack_success:', attack_success)
        attack_end_time = time.time() - attack_start
        print('text_attack_time:',attack_end_time )
        fake_img = torch.clamp(img_adv, 0, 1).permute([1, 2, 0]).data.cpu().numpy()[:, :, [2, 1, 0]]
        cv2.imwrite('/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/new_2/fake_images/fake_{}.png'.format(story_id), (fake_img * 255).astype(int))
       

        print('adversarial attack ending:', ending_f)

        res = str(story_id) + '\t' + str(story_gold_end) + '\t' + sents + '\t' + ending_f + '\t' + \
              origin_text + '\t' + text_adv[0] + '\t' + str(attack_end_time) + '\t' + str(attack_success)
        print(res)
        f.write(res+'\n')
        torch.cuda.empty_cache()

    f.close()


opt = opts.parse_opt()
attack(opt)

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from pycocoevalcap.cider.cider import Cider
import numpy as np
from rouge import Rouge
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sacrebleu
from nltk.translate.chrf_score import sentence_chrf
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
# model = SentenceTransformer("johngiorgi/declutr-small")
sim_model = SentenceTransformer('all-MiniLM-L6-v2')
from scipy.spatial.distance import cosine
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("johngiorgi/declutr-small")
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
import sys
sys.path.append('/data/home/wangyouze/projects/others/MMT/')
from metrics import show_all_scores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "gpt2-large"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)



# print(sacrebleu.sentence_bleu('now the man and someone watch their other UNK', ['now someone proofreads his handwritten last will and testament']))
# print(sacrebleu.sentence_bleu('the man steps out', ['now someone proofreads his handwritten last will and testament']))
# print(sentence_bleu(['later they return to the dorm'.split(' ')], 'the UNK casts the UNK driver casts its UNK view inches on the ship UNK UNK over the UNK ship'.split(' '), weights=[0.25, 0.25, 0.25, 0.25]))
def get_ppl_score(text):
    # text = "we might just look like a random group of drunk partygoers here , but we were n't [SEP] we were all here for one reason -- -to celebrate the end of our friend [ male ] s cancer treatment [SEP] i gave a speech about him , and i teared up badly [SEP] we gave him a superman shirt , and he put it on that is what he is to all of us -- a superman"
    text = text.replace('[SEP]', '.')
    encodings = tokenizer(text, return_tensors="pt")

    max_length = gpt2_model.config.n_positions
    stride = 100
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    ppl = float(ppl.cpu().detach().numpy())
    # print(ppl)
    return ppl
def calculate_similarity(origin_text, attack_text):
    sim = []
    origin_text = origin_text.replace('[SEP]', ',')
    attack_text = attack_text.replace('[SEP]', ',')

    text = [origin_text, attack_text]

    embeddings = sim_model.encode(text)
    semantic_sim = 1 - cosine(embeddings[0], embeddings[1])
    print(semantic_sim)
    sim.append(semantic_sim)

    return sum(sim) / len(sim)
def compute_score(predfile):

    chrf_origin = []
    chrf_attack = []
    bleu_origin = []
    bleu_attack = []
    cost_time = []

    res = 0
    story_id_list = []
    perturbation_rate_list = []
    perp_list = []
    sim_list = []
    true_count = 0
    total_count = 0

    gs_list = []
    ps_list = []
    fs_list = []

    # meteor_origin = []
    # meteor_attack = []
    # cider_origin = []
    # cider_attack = []
    # rouge_origin = []
    # rouge_attack = []
    with open(predfile, 'r', encoding="utf-8") as pf:
        for i, line in enumerate(pf):
            print(i)
            new_line = line.strip().split('\t')
            if len(new_line) != 8:
                print('Warning the length of the lien is not 4')
                break
            story_id = new_line[0]

            if story_id in story_id_list:
                continue
            story_id_list.append(story_id)
            gs = new_line[1]
            attack_ending = new_line[3]
            ps = new_line[2]
            origin_text = new_line[4]
            attack_text = new_line[5]

            
            bleu_1 = sacrebleu.sentence_bleu(attack_ending, [gs]).score
            bleu_2 = sacrebleu.sentence_bleu(ps, [gs]).score
            if bleu_2 == 0:
                continue
            gs_list.append(gs)
            ps_list.append(ps)
            fs_list.append(attack_ending)
            
            total_count += 1
            if new_line[-1] == 'True':
                true_count += 1
            
            num_perturbation = 2
            perturbation_rate = num_perturbation / len(origin_text)
            perturbation_rate_list.append(perturbation_rate)
            sim = calculate_similarity(origin_text, attack_text)
            sim_list.append(sim)
            print('sim:', sim)
            perp = get_ppl_score(attack_text)
            perp_list.append(perp)
            print('perp:', perp)

            bleu_attack.append(bleu_1)
            bleu_origin.append(bleu_2)
            if bleu_1 <= bleu_2 /2:
                res += 1
                print('-------------------------------')
                print(story_id)
                print(bleu_1)
                print(bleu_2)
                print('-------------------------------')

                time = float(new_line[6])
                print(time)
                cost_time.append(time)
            chrf_origin.append(sentence_chrf(gs, ps))
            chrf_attack.append(sentence_chrf(gs, attack_ending))


    _, origin_meteor_results, origin_cider_results, origin_rouge_results, origin_rsum = show_all_scores(gs_list, ps_list, n=4)
    _, attack_meteor_results, attack_cider_results, attack_rouge_results, attack_rsum = show_all_scores(gs_list, fs_list, n=4)
    meteor = (origin_meteor_results['METEOR'] - attack_meteor_results['METEOR']) /origin_meteor_results['METEOR']
    cider = (origin_cider_results['CIDEr'] - attack_cider_results['CIDEr']) / origin_cider_results['CIDEr']
    rouge_l = (origin_rouge_results['ROUGE-L'] - attack_rouge_results['ROUGE-L']) / origin_rouge_results['ROUGE-L']
    

    # return round(bleu_1, 4), round(bleu_2, 4), round(bleu_3, 4), round(bleu_4,  4)
    chrf_origin = sum(chrf_origin) / len(chrf_origin)
    chrf_attack = sum(chrf_attack) / len(chrf_attack)
    chrf = (chrf_origin - chrf_attack) / chrf_origin

    bleu_origin = sum(bleu_origin) / len(bleu_origin)
    bleu_attack = sum(bleu_attack) / len(bleu_attack)
    bleu = (bleu_origin - bleu_attack) / bleu_origin

    sim = sum(sim_list) / len(sim_list)
    perp = sum(perp_list) / len(perp_list)
    p_rate = sum(perturbation_rate_list) / len(perturbation_rate_list)
    avg_time = sum(cost_time)/ len(cost_time)
    return {'true_count':true_count, 'res':res, 'total_count':total_count, 'ASR':res / total_count, 'bleu':bleu, 'chrf':chrf, 'meteor':meteor, 'cider':cider, 'rouge_l':rouge_l, 'sim':sim, 'perp':perp,'p_rate':p_rate,'avg_time':avg_time}
    # return true_count, res, total_count, bleu, chrf, meteor, cider, rouge_l, sim, perp, p_rate, avg_time
    # return res, count, bleu, chrf

if __name__=="__main__":
    print("Preparation for evaluation")

   
    # pred_file = "/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_all.txt"
    # pred_file = '/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/new_2/results_MMT_1.txt'
    # pred_file = '/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/results_MMT_loss_4.txt'
    # pred_file = "/data/home/wangyouze/projects/others/MMT/Co-attack/output/results_MMT_co_attack.txt"
    # pred_file = "/data/home/wangyouze/projects/others/MMT_lsmdc/Co-attack/output/results_MMT_co_attack_lsmdc_new_2.txt"
    # pred_file = "/data/home/wangyouze/projects/others/MMT_lsmdc/Iterative-attack/outputs/results_MMT_all.txt"

    # pred_file = '/data/home/wangyouze/projects/others/kNN/pytorch_translate/MMT/output/results_lsmdc_kNN.txt'
    pred_file = "/data/home/wangyouze/projects/others/MMT/Iterative-attack/outputs/epsilon/results_MMT_1.txt"
    score = compute_score(pred_file)
    print('score:', score)



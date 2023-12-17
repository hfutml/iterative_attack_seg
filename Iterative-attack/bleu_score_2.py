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



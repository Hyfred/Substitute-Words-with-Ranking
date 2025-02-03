
import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Model
import torch
from tqdm import tqdm
import json
import math
import matplotlib.pyplot as plt
import re
from torch.nn.functional import cosine_similarity
import pickle
import os
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def read_prediction(data_path = './confusion_data.json'):
    '''
    Input: prediction file path (str)
    Splits the file into:
    1. changenotchange
    2. changecorrect
    3. changeincorrect
    4. notchangenotchange
    5. notchangechange
    :return: each (sentence; substitution pair)
    '''
    with open(data_path, 'r') as file:
        loaded_dict = json.load(file)
    changecorrect = loaded_dict['changecorrect']
    changeincorrect = loaded_dict['changeincorrect']
    notchangechange = loaded_dict['notchangechange']
    changenotchange = loaded_dict["changenotchange"]
    notchangenotchange = loaded_dict["notchangenotchange"]

    cc_rand = loaded_dict['changecorrect_randomsample']
    cinc_rand = loaded_dict['changeincorrect_randomsample']
    notcc_rand = loaded_dict['notchangechange_randomsample']
    cnotc_rand = loaded_dict["changenotchange_randomsample"]
    notcnotc_rand = loaded_dict["notchangenotchange_randomsample"]

    cc_model = loaded_dict['changecorrect_samplefrommodel']
    cinc_model = loaded_dict['changeincorrect_samplefrommodel']
    notcc_model = loaded_dict['notchangechange_samplefrommodel']
    cnotc_model = loaded_dict["changenotchange_samplefrommodel"]
    notcnotc_model = loaded_dict["notchangenotchange_samplefrommodel"]

    # n1,n2,n3 = 0,0,0
    # for sentence in changecorrect:
    #     n1+=len(changecorrect[sentence])
    # for sentence in changeincorrect:
    #     n2+=len(changeincorrect[sentence])
    # for sentence in notchangechange:
    #     n3+=len(notchangechange[sentence])
    # print(n1,n2,n3)

    return changecorrect, changeincorrect, notchangechange, changenotchange, notchangenotchange, \
        cc_rand, cinc_rand, notcc_rand, cnotc_rand,notcnotc_rand,\
        cc_model, cinc_model, notcc_model,cnotc_model,notcnotc_model

def calculate_p_value():
    '''
    Input: each (sentence; substitution pair)
    sample from model/random distribution
    :return: p value
    '''

def proportion_less_than_zero(lst):
    '''calculate proportion less than 0'''
    count_less_than_zero = sum(1 for num in lst if num < 0)
    proportion = count_less_than_zero / len(lst)
    return proportion

def call_model(ori_sent, sub_sent, random_sent,model_sample_sent, model):
    '''
    input: ori_sent[str], sub_sent[list],
    output: ratio[list]
    '''

    srctotgt = []

    for modified_sentence in sub_sent:
        srctotgt_score = model.score([ori_sent], [modified_sentence], batch_size=1)[0]
        srctotgt.append(srctotgt_score) #math.exp(ratio)

    batch_size = 250  # Choose your desired batch size

    random_s2t = []

    # for i in range(len(random_sent)):
    #
    #     # for modified_sentence in random_sent[i]:
    #     for j in range(0, len(random_sent[i]), batch_size):
    #         batch_random_sent = random_sent[i][j:j + batch_size]
    #         exact_batch_size = len(batch_random_sent)
    #         batch_ori = [ori_sent]*exact_batch_size
    #         srctotgt_score = model.score(batch_ori, batch_random_sent, batch_size=exact_batch_size)
    #         random_s2t.extend(srctotgt_score)

    # waiting for deleting
    # for i in range(len(random_sent)):
    #     # ratio_standard = srctotgt[i]
    #     for modified_sentence in random_sent[i]:
    #         srctotgt_score = model.score([ori_sent], [modified_sentence], batch_size=1)[0]
    #
    #         # Store the score and replacement word
    #         random_s2t.append(srctotgt_score)  # math.exp(ratio)

    model_s2t = []

    # for i in range(len(model_sample_sent)):
    #     # ratio_standard = srctotgt[i]
    #     for modified_sentence in model_sample_sent[i]:
    #         srctotgt_score = model.score([ori_sent], [modified_sentence], batch_size=1)[0]
    #         model_s2t.append(srctotgt_score)

    for i in range(len(model_sample_sent)):
        if len(model_sample_sent[i])==0:
            model_s2t.append([])
            continue
        single_tgt_sample = []
        for j in range(0, len(model_sample_sent[i]), batch_size):
            batch_random_sent = model_sample_sent[i][j:j + batch_size]
            exact_batch_size = len(batch_random_sent)
            batch_ori = [ori_sent] * exact_batch_size
            srctotgt_score = model.score(batch_ori, batch_random_sent, batch_size=exact_batch_size)
            single_tgt_sample.append(srctotgt_score)
        model_s2t.extend(single_tgt_sample)

    return srctotgt, random_s2t, model_s2t


def calculate_ratio(sent_pair, randomsample, samplebymodel, model='bartscore'):
    '''
    load data json file
    if mode='train': split train:dev as 9:1
    if mode='test': do nothing
    '''
    if model == 'bartscore':
        from bartscore import BARTScorer
        model = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        model.load(path='bartscore/bart_score.pth')

    srctotgts, random_s2ts, model_s2ts = [],[],[]
    random_sents = list(randomsample.values())
    model_sample_sents = list(samplebymodel.values())

    for i, (ori_sent, sub_sent) in tqdm(enumerate(sent_pair.items())):


        random_sent = []#random_sents[i]
        model_sample_sent = model_sample_sents[i]
        # if model_sample_sent == []:
        #     print(1)
        srctotgt, random_s2t, model_s2t = call_model(ori_sent, sub_sent, random_sent, model_sample_sent, model)
        srctotgts.append(srctotgt)
        random_s2ts.append(random_s2t)
        model_s2ts.append(model_s2t)

    return srctotgts, random_s2ts, model_s2ts

def plot_distribution(ratio, filename):
    '''
    input: ratio[list]
    plot distribution of ratio
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.histplot(ratio, kde=True, stat='density', log_scale=False, ax=ax)
    plt.show()
    if filename:
        fig.savefig(filename)  # Save the plot with the given filename

    plt.boxplot(ratio)
    plt.title('Boxplot')
    plt.show()

def plot_p_distribution(ratio, filename):
    '''
    input: ratio[list]
    plot distribution of ratio
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    # Convert the list to a NumPy array
    log_data = np.array(ratio)

    # Plot the distribution in log domain with original value ticks
    plt.figure(figsize=(8, 6))
    plt.hist(log_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    # Set original value ticks on the x-axis
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(ticks=plt.xticks()[0], labels=[f"{round(np.exp(x), 2)}" for x in plt.xticks()[0]])

    plt.title('Distribution of Data in Log Domain')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    if filename:
        plt.savefig(filename)  # Save the plot with the given filename

'''
 1. specify the category
 2. ratio distribution
     1. calculate ratio by llm
     2. plot distribution
 3. p value distribution
     1. calculate p value for each sent(sample from model/random)
     2. plot distribution
'''

def save_read_json(srctotgts, random_s2ts, model_s2ts, save_name, mode):
    '''save list to json or read json to list'''
    if mode == 'save':
        save_data = {}
        save_data['srctotgts'] = srctotgts
        save_data['random_s2ts'] = random_s2ts
        save_data['model_s2ts'] = model_s2ts
        with open(save_name, 'w') as f:
            json.dump(save_data, f)
    elif mode == 'read':
        with open(save_name, 'r') as f:
            save_data = json.load(f)
        srctotgts = save_data['srctotgts']
        random_s2ts = save_data['random_s2ts']
        model_s2ts = save_data['model_s2ts']
        return srctotgts, random_s2ts, model_s2ts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_argument('--mode', type=str, default='changecorrect')
    args = parser.parse_args()

    #cc, cinc, notcc, cc_randomsample, cinc_randomsample, notcc_randomsample, cc_samplefrommodel, cinc_samplefrommodel, notcc_samplefrommodel  = read_prediction()
    # srctotgts, tgttosrcs,ratios ,random_s2ts,random_t2ss,randomratios, model_s2ts, model_t2ss,modelratios = calculate_ratio(cc, cc_randomsample, cc_samplefrommodel, model='bartscore')
    # print(len(srctotgts),len(tgttosrcs),len(ratios))
    # print(len(random_s2ts), len(random_t2ss), len(randomratios))
    # print(len(model_s2ts), len(model_t2ss), len(modelratios))

    cc, cinc, notcc,cnotc,notcnotc, cc_rand, cinc_rand, notcc_rand,cnotc_rand,notcnotc_rand, cc_model, cinc_model, notcc_model,cnotc_model,notcnotc_model = read_prediction()
    print(len(cc),len(cinc),len(notcc),len(cnotc),len(notcnotc))
    print(len(cc_rand),len(cinc_rand),len(notcc_rand),len(cnotc_rand),len(notcnotc_rand))
    print(len(cc_model),len(cinc_model),len(notcc_model),len(cnotc_model),len(notcnotc_model))


    def flatten_len(nested_list):
        num = 0
        for item in nested_list.values():
            num+=len(item)
        return num

    print(flatten_len(cc), flatten_len(cinc), flatten_len(notcc), flatten_len(cnotc), flatten_len(notcnotc))
    print(len(cc_rand),len(cinc_rand),len(notcc_rand),len(cnotc_rand),len(notcnotc_rand))
    print(flatten_len(cc_model), flatten_len(cinc_model), flatten_len(notcc_model), flatten_len(cnotc_model), flatten_len(notcnotc_model))


    # save_read_json(srctotgts, tgttosrcs,ratios ,random_s2ts,random_t2ss,randomratios, model_s2ts, model_t2ss,modelratios, 'cc_bartscore', mode='save')

    # plot_distribution(ratio, 'ratio.png')
    # plot_p_distribution(p_values_random_sample,'p_values_random_sample.png')
    # plot_p_distribution(p_values_sample_from_model,'p_values_sample_from_model.png')

    # ratio, p_values_random_sample, p_values_sample_from_model = calculate_ratio(cinc, cinc_randomsample, cinc_samplefrommodel, model='gpt2')
    # print(len(ratio),len(p_values_random_sample),len(p_values_sample_from_model))
    # print(proportion_less_than_zero(ratio))
    # save_read_json(ratio, p_values_random_sample, p_values_sample_from_model, 'cinc_single_power-10', mode='save')


    # srctotgts, tgttosrcs,ratios ,random_s2ts,random_t2ss,randomratios, model_s2ts, model_t2ss,modelratios = save_read_json(None, None,None ,None,None,None, None, None,None, 'cc_bartscore', mode='read')

    srctotgts, random_s2ts, model_s2ts = calculate_ratio(cc, cc_rand, cc_model, model='bartscore')
    print(len(srctotgts))
    print(len(random_s2ts))
    print(len(model_s2ts))

    save_read_json(srctotgts, random_s2ts, model_s2ts,'cc_bartscore', mode='save')

    srctotgts, random_s2ts, model_s2ts = calculate_ratio(cinc, cinc_rand, cinc_model, model='bartscore')
    print(len(srctotgts))
    print(len(random_s2ts))
    print(len(model_s2ts))

    save_read_json(srctotgts, random_s2ts, model_s2ts,'cinc_bartscore', mode='save')

    srctotgts, random_s2ts, model_s2ts = calculate_ratio(notcc, notcc_rand, notcc_model, model='bartscore')
    print(len(srctotgts))
    print(len(random_s2ts))
    print(len(model_s2ts))

    save_read_json(srctotgts, random_s2ts, model_s2ts,'notcc_bartscore', mode='save')

    srctotgts, random_s2ts, model_s2ts = calculate_ratio(cnotc, cnotc_rand, cnotc_model, model='bartscore')
    print(len(srctotgts))
    print(len(random_s2ts))
    print(len(model_s2ts))

    save_read_json(srctotgts, random_s2ts, model_s2ts,'cnotc_bartscore', mode='save')

    srctotgts, random_s2ts, model_s2ts = calculate_ratio(notcnotc, notcnotc_rand, notcnotc_model, model='bartscore')
    print(len(srctotgts))
    print(len(random_s2ts))
    print(len(model_s2ts))

    save_read_json(srctotgts, random_s2ts, model_s2ts,'notcnotc_bartscore', mode='save')

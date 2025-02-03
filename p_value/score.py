import json
import math
import collections
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import numpy as np
from transformers import BertTokenizer
import random
from bertspsv import BertForMaskedLM_dropout
import torch
from tqdm import tqdm
import re

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get the BERT uncased vocabulary
vocab = tokenizer.get_vocab()
vocab = list(vocab.keys())


net = BertForMaskedLM_dropout.from_pretrained('bert-base-uncased')
net.cuda()

changecorrect = {}
changeincorrect = {}
notchangechange = {}
changenotchange = {}
notchangenotchange = {}

changecorrect_randomsample = {}
changeincorrect_randomsample = {}
notchangechange_randomsample = {}
changenotchange_randomsample = {}
notchangenotchange_randomsample = {}

changecorrect_samplefrommodel = {}
changeincorrect_samplefrommodel = {}
notchangechange_samplefrommodel = {}
changenotchange_samplefrommodel = {}
notchangenotchange_samplefrommodel = {}


def find_occurrence_times(lst, start, end):
    '''detect which occurrence of the given index target word'''
    if start > len(lst):
        print(1)

    element = lst[start]
    occurrences = [i for i, x in enumerate(lst) if x == element]
    if start in occurrences:
        occurrence_index = occurrences.index(start) + 1
    else: occurrence_index = 1

    return occurrence_index


def create_sent_pair(sentence, tgt_word, prediction, sent_token, start, end):
    '''create the replacement sentence
    if only one tgt word in sentence. use regex
    else detech which orrurence time of the tgt word then replace it
    '''
    showup_time = sentence.count(tgt_word)
    if showup_time == 1:
        replaced_sentence = sentence.replace(tgt_word, prediction, 1)
    else:
        occurrence_time = find_occurrence_times(sent_token, start, end)
        #replaced_sentence = sentence.replace(tgt_word, prediction, occurrence_time)
        replaced_sentence = replace_nth_occurrence(sentence, tgt_word, prediction, occurrence_time) # change the occurrence time of tgt
    return replaced_sentence

def sample_replacement(num_words, vocab):
    '''sample number words from vacabulary by random'''
    sampled_words = random.sample(vocab, num_words)
    return sampled_words


def exclude_keys_from_dicts(list_sent, dict1, dict2):
    '''obtain index list of sent after exclude the index in dict1&2'''
    list_length = len(list_sent)
    excluded_indexes = set()

    # obtain index range of given two dicts
    for key in dict1.keys():
        start, end = map(int, key.split(":"))
        excluded_indexes.update(range(start, end + 1))
    for key in dict2.keys():
        start, end = map(int, key.split(":"))
        excluded_indexes.update(range(start, end + 1))

    # generate the index list excluse those index has appeared already.
    result = [str(i) for i in range(list_length) if i not in excluded_indexes]

    return result


def replace_nth_occurrence(input_string, word1, word2, occurrence):
    # Define the regex pattern to match word1
    pattern = re.compile(r'\b' + re.escape(word1) + r'\b')
    # Counter to keep track of occurrences
    count = 0
    # Function to replace nth occurrence
    def replace_word(match):
        nonlocal count
        count += 1
        if count == occurrence:
            return word2
        else:
            return match.group(0)

    # Replace nth occurrence using the defined function
    result = pattern.sub(replace_word, input_string)

    return result

def sample_replacement_from_model(num_words, vocab, sentence, template, prediction):
    '''sample number words from vacabulary by model distribution'''
    sent = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(sent)

    tmp = ['[CLS]'] + tokenizer.tokenize(template) + ['[SEP]']
    tmp_ids = tokenizer.convert_tokens_to_ids(tmp)

    offset = None
    for i, (elem1, elem2) in enumerate(zip(input_ids, tmp_ids)):
        if elem1 != elem2:
            offset = i
            break


    masked_ids = torch.tensor([input_ids]).cuda()
    segment_ids = torch.tensor([[0]*len(sent)]).cuda()
    classification_logits = net(offset, masked_ids, token_type_ids=segment_ids, output_hidden_states=True, return_dict=True, output_attentions = True)

    topk1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prediction))[0]
    tgt_logit = classification_logits[0][offset].cpu().tolist()
    tgt_logit.pop(topk1)
    # Normalize the probabilities
    normalized_probs = np.array(tgt_logit) / np.sum(tgt_logit)
    vocab_remove_topk1 = vocab.copy()
    vocab_remove_topk1.pop(topk1)

    sampled_words = np.random.choice(vocab_remove_topk1, num_words, p=normalized_probs, replace=False).tolist()

    return sampled_words


def eval(pred, gold):

    with open(gold, 'r', errors='ignore') as f:
        goldjson = json.load(f, strict=False)

    if type(pred) != type('123'):
        outputjson = pred
    else:
        with open(pred, 'r', errors='ignore') as f:
            outputjson = json.load(f, strict=False)

    def NDCG(pred, gold):

        # Discounted Cumulative Gain for predictions
        pred_score = [gold.get(i,0) for i in pred]
        DCG_pred, idx = 0, 1
        for i, c in enumerate(pred_score):
            idx += 1
            DCG_pred += c / math.log(idx, 2)

        # Discounted Cumulative Gain for the ground truth
        gold = sorted(gold.values(), reverse=True) + [0] * len(pred)
        DCG_gold, idx = 0, 1
        for i, c in enumerate(pred_score):
            idx += 1
            DCG_gold += gold[i] / math.log(idx, 2)

        res = DCG_pred/DCG_gold

        return res

    detection_tp, detection_pred, detection_gold = 0, 0, 0
    detection_tp_level, detection_gold_level = collections.defaultdict(int), collections.defaultdict(int)
    detection_tp_type, detection_gold_type= [0, 0, 0], [0, 0, 0]
    weighted_detection_tp, weighted_detection_gold = 0, 0
    recommendation_tp, recommendation_pred, recommendation_gold = 0, 0, 0
    e2e_tp, e2e_pred, e2e_gold = 0, 0, 0
    ndcg, recommendation_acc = 0, 0
    substitute_word_num, total_word_num, gold_substitute_word_num = 0, 0, 0

    e2e_tp_type, e2e_gold_type = [0, 0, 0], [0, 0, 0]
    recommendation_acc_type = [0, 0, 0]
    cnotc, cc, cinc, notcnotc, notcc = 0,0,0,0,0

    for idx, v in tqdm(list(goldjson.items())[0:]):
        #print(outputjson[idx]['substitute_topk'][0][0][2])
        #total_word_num += len(outputjson[idx]['input_words'])
        sentence = v['sentence']
        sent_token = v['sentence_split']
        total_word_num += len(v['sentence_split'])
        substitute_word_num += sum([i[0][2]-i[0][1] for i in outputjson[idx]['substitute_topk']])

        pred_res = {"%d:%d" % (k[1],k[2]):v for k,v in outputjson[idx]['substitute_topk']}
        gold_res = v['substitutes']
        gold_res_clip = {}
        gold_res_clip_type = [{}, {}]
        for k, vsub, typenum in gold_res: 
            thiskdict = {}
            for kk, vv in vsub.items():
                thiskdict[kk] = vv
                gold_res_clip["%d:%d" % (k[0],k[1])] = thiskdict
                gold_res_clip_type[typenum-1]["%d:%d" % (k[0],k[1])] = thiskdict
        gold_substitute_word_num += len(gold_res_clip)

        pred_detection = set(pred_res.keys())
        gold_detection = set(gold_res_clip.keys())

        improvable_target = {k:sum(v.values()) for k, v in gold_res_clip.items()}
        for k, v in improvable_target.items():
            detection_gold_level[v-1] += 1
            if k in pred_detection: detection_tp_level[v-1] += 1

        tp_substitute_num, pred_substitute_num, gold_substitute_num = 0, 0, 0
        tp_substitute_num_type, gold_substitute_num_type = [0, 0, 0], [0, 0, 0]
        sample_number = 1000 # for sample word from vocabulary

        for k, v in gold_res_clip.items():
            if k in pred_res:
                pred_substitute_num += len(pred_res[k])
                gold_substitute_num += len(v)
                tp_substitute_num += len(set(pred_res[k]) & set(v.keys()))
                ndcg += NDCG(pred_res[k], v)
                if pred_res[k][0] in v: recommendation_acc += 1; e2e_tp += 1

                # replace rule: if only one tgt word in sentence. use regex otherwise detokenizer
                start, end = map(int, k.split(':'))  # Parse the slice string into start and end indices
                #tgt_word = " ".join(sent_token[start:end])
                if end-1==start:
                    tgt_word = sent_token[start]
                else:
                    tgt_word = TreebankWordDetokenizer().detokenize(sent_token[start:end])
                prediction = pred_res[k][0]
                replaced_sentence = create_sent_pair(sentence, tgt_word, prediction, sent_token, start, end)

                # sample_words = sample_replacement(sample_number,vocab) # sample by random
                # sample_sentence = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for sample_word in sample_words]

                try:
                    sample_words_from_model = sample_replacement_from_model(sample_number, vocab, sentence, replaced_sentence, prediction)
                    sample_sentence_from_model = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for
                                      sample_word in sample_words_from_model]

                    if pred_res[k][0] in v: #changecorrect
                        cc += 1
                        if sentence not in changecorrect:
                            changecorrect[sentence] = [replaced_sentence]
                        else:
                            changecorrect[sentence].append(replaced_sentence)

                        # if sentence not in changecorrect_randomsample:
                        #     changecorrect_randomsample[sentence] = [sample_sentence]
                        # else:
                        #     changecorrect_randomsample[sentence].append(sample_sentence)

                        if sentence not in changecorrect_samplefrommodel:
                            changecorrect_samplefrommodel[sentence] = [sample_sentence_from_model]
                        else:
                            changecorrect_samplefrommodel[sentence].append(sample_sentence_from_model)

                    else: #changeincorrect
                        cinc += 1
                        if sentence not in changeincorrect:
                            changeincorrect[sentence] = [replaced_sentence]
                        else:
                            changeincorrect[sentence].append(replaced_sentence)

                        # if sentence not in changeincorrect_randomsample:
                        #     changeincorrect_randomsample[sentence] = [sample_sentence]
                        # else:
                        #     changeincorrect_randomsample[sentence].append(sample_sentence)

                        if sentence not in changeincorrect_samplefrommodel:
                            changeincorrect_samplefrommodel[sentence] = [sample_sentence_from_model]
                        else:
                            changeincorrect_samplefrommodel[sentence].append(sample_sentence_from_model)
                except Exception as e:
                    print(e)
                    #print(tgt_word, prediction)

            else: # changeNotchange
                cnotc += 1
                # replace rule: if only one tgt word in sentence. use regex otherwise detokenizer
                start, end = map(int, k.split(':'))  # Parse the slice string into start and end indices
                # #tgt_word = " ".join(sent_token[start:end])
                if end-1==start:
                    tgt_word = sent_token[start]
                else:
                    tgt_word = TreebankWordDetokenizer().detokenize(sent_token[start:end])
                replaced_sentence = sentence # same word as tgt since pred not change
                
                # # sample_words = sample_replacement(sample_number,vocab) # sample by random
                # # sample_sentence = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for sample_word in sample_words]
                template_sentence = create_sent_pair(sentence, tgt_word, '#demo#', sent_token, start, end)
                #
                try:
                    sample_words_from_model = sample_replacement_from_model(sample_number, vocab, sentence, template_sentence, tgt_word)
                    sample_sentence_from_model = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for
                                      sample_word in sample_words_from_model]
                    if sentence not in changenotchange:
                        changenotchange[sentence] = [replaced_sentence]
                    else:
                        changenotchange[sentence].append(replaced_sentence)

                    # if sentence not in changenotchange_randomsample:
                    #     changenotchange_randomsample[sentence] = [sample_sentence]
                    # else:
                    #     changenotchange_randomsample[sentence].append(sample_sentence)

                    if sentence not in changenotchange_samplefrommodel:
                        changenotchange_samplefrommodel[sentence] = [sample_sentence_from_model]
                    else:
                        changenotchange_samplefrommodel[sentence].append(sample_sentence_from_model)
                except Exception as e:
                    #print(e)
                    print(tgt_word)

        for k, v in pred_res.items(): # notchangechange
            if k not in gold_res_clip:
                notcc+=1
                start, end = map(int, k.split(':'))  # Parse the slice string into start and end indices
                if start>len(sent_token)-1:
                    #print('error')
                    continue
                #tgt_word = " ".join(sent_token[start:end])
                if end-1==start:
                    tgt_word = sent_token[start]
                else:
                    tgt_word = TreebankWordDetokenizer().detokenize(sent_token[start:end])
                prediction = pred_res[k][0]
                replaced_sentence = create_sent_pair(sentence, tgt_word, prediction, sent_token, start, end)
                
                # sample_words = sample_replacement(sample_number, vocab)
                # sample_sentence = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for sample_word in sample_words]
                
                try:
                    sample_words_from_model = sample_replacement_from_model(sample_number, vocab, sentence, replaced_sentence, prediction)
                    sample_sentence_from_model = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for
                                      sample_word in sample_words_from_model]
                
                    if sentence not in notchangechange:
                        notchangechange[sentence] = [replaced_sentence]
                    else:
                        notchangechange[sentence].append(replaced_sentence)
                
                    # if sentence not in notchangechange_randomsample:
                    #     notchangechange_randomsample[sentence] = [sample_sentence]
                    # else:
                    #     notchangechange_randomsample[sentence].append(sample_sentence)
                
                    if sentence not in notchangechange_samplefrommodel:
                        notchangechange_samplefrommodel[sentence] = [sample_sentence_from_model]
                    else:
                        notchangechange_samplefrommodel[sentence].append(sample_sentence_from_model)
                except Exception as e:
                    #print(e)
                    print(tgt_word, prediction)

        # notchangenotchange
        notchangenotchange_idx =  exclude_keys_from_dicts(sent_token, gold_res_clip, pred_res)
        for start in notchangenotchange_idx:
            notcnotc += 1
            start = int(start)
            end=start+1
            tgt_word = " ".join(sent_token[start:end])
            replaced_sentence = sentence  # same word as tgt since pred not change
            
            # sample_words = sample_replacement(sample_number, vocab)  # sample by random
            # sample_sentence = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for sample_word in
            #                    sample_words]
            template_sentence = create_sent_pair(sentence, tgt_word, '#demo#', sent_token, start, end)
            
            try:
                sample_words_from_model = sample_replacement_from_model(sample_number, vocab, sentence, template_sentence, tgt_word)
                sample_sentence_from_model = [create_sent_pair(sentence, tgt_word, sample_word, sent_token, start, end) for
                                              sample_word in sample_words_from_model]
                if sentence not in notchangenotchange:
                    notchangenotchange[sentence] = [replaced_sentence]
                else:
                    notchangenotchange[sentence].append(replaced_sentence)
            
                # if sentence not in notchangenotchange_randomsample:
                #     notchangenotchange_randomsample[sentence] = [sample_sentence]
                # else:
                #     notchangenotchange_randomsample[sentence].append(sample_sentence)
            
                if sentence not in notchangenotchange_samplefrommodel:
                    notchangenotchange_samplefrommodel[sentence] = [sample_sentence_from_model]
                else:
                    notchangenotchange_samplefrommodel[sentence].append(sample_sentence_from_model)
            except Exception as e:
                print(e)
                print(tgt_word)

        for typeid in range(2):
            for k, v in gold_res_clip_type[typeid].items():
                if k in pred_res:
                    gold_substitute_num_type[typeid] += len(v)
                    tp_substitute_num_type[typeid] += len(set(pred_res[k]) & set(v.keys()))
                    if pred_res[k][0] in v:
                        recommendation_acc_type[typeid] += 1; e2e_tp_type[typeid] += 1

        weighted_detection_dict = {k:sum(v.values()) for k, v in gold_res_clip.items()}

        tp_select = list(pred_detection & gold_detection)
        detection_tp += len(tp_select)
        detection_pred += len(pred_detection)
        detection_gold += len(gold_detection)
        
        for typeid in range(2):
            detection_tp_type[typeid] += len(list(pred_detection & set(gold_res_clip_type[typeid].keys())))
            detection_gold_type[typeid] += len(gold_res_clip_type[typeid].keys())
            e2e_gold_type[typeid] += len(gold_res_clip_type[typeid].keys())

        weighted_detection_tp += sum([weighted_detection_dict[i] for i in tp_select])
        weighted_detection_gold += sum(weighted_detection_dict.values())

        recommendation_tp += tp_substitute_num
        recommendation_pred += pred_substitute_num
        recommendation_gold += gold_substitute_num

        e2e_pred += len(pred_res)
        e2e_gold += len(gold_res_clip)

    print(cnotc, cc, cinc, notcnotc, notcc)
        # break

    p_detection = detection_tp / detection_pred
    r_detection = detection_tp / detection_gold
    r_detection_type = []
    for typeid in range(2):
        r_detection_type.append(detection_tp_type[typeid] / detection_gold_type[typeid])

    r_detection_level = {}
    for i in range(100):
        if detection_gold_level[i]: r_detection_level[i] = (detection_tp_level[i]/detection_gold_level[i]) 

    f_detection = 2 * p_detection * r_detection / (p_detection+r_detection)
    f_detection_05 = 1.25 * p_detection * r_detection / (0.25 * p_detection+r_detection)
    weighted_acc_detection = weighted_detection_tp/weighted_detection_gold

    p_recommendation = recommendation_tp / recommendation_pred
    r_recommendation = recommendation_tp / recommendation_gold
    f_recommendation = 2 * p_recommendation * r_recommendation / (p_recommendation + r_recommendation )
    f_recommendation_05 = 1.25 * p_recommendation * r_recommendation / (0.25 *p_recommendation + r_recommendation )
    ndcg_recommendation = ndcg/detection_tp
    acc_recommendation = recommendation_acc/detection_tp
    acc_recommendation_type = []
    for typeid in range(2):
        acc_recommendation_type.append(recommendation_acc_type[typeid] / detection_tp_type[typeid])

    p_e2e = e2e_tp / e2e_pred
    r_e2e = e2e_tp / e2e_gold

    r_e2e_type = []
    for typeid in range(2):
        r_e2e_type.append(e2e_tp_type[typeid] / e2e_gold_type[typeid])

    f_e2e = (2 * p_e2e * r_e2e / (p_e2e+r_e2e)) if p_e2e+r_e2e else 0
    f_e2e_05 = (1.25 * p_e2e * r_e2e / (0.25 *p_e2e+r_e2e)) if p_e2e+r_e2e else 0

    print(substitute_word_num)
    print(total_word_num)
    substitute_rate = substitute_word_num / total_word_num
    #substitute_rate = substitute_word_num

    return {
        "p_detection": p_detection,
        "r_detection": r_detection,
        "r_detection_type": r_detection_type,
        "f_detection": f_detection,
        "f_detection_05": f_detection_05,
        "weighted_acc_detection": weighted_acc_detection,
        "p_recommendation": p_recommendation,
        "r_recommendation": r_recommendation,
        "f_recommendation": f_recommendation,
        "f_recommendation_05": f_recommendation_05,
        "ndcg_recommendation": ndcg_recommendation,
        "acc_recommendation": acc_recommendation,
        "acc_recommendation_type": acc_recommendation_type,
        "p_e2e": p_e2e, 
        "r_e2e": r_e2e,
        "r_e2e_type": r_e2e_type,
        "f_e2e": f_e2e,
        "f_e2e_05": f_e2e_05,
        "substitute_rate": substitute_rate
    }



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, default='./sws_test.json')
    parser.add_argument("--pred", type=str, default="./res.json")
    parser.add_argument("--name", type=str, default='score_example')
    args = parser.parse_args()

    res = eval(args.pred, args.gold)

    # print(json.dumps(res, indent=4))

    print("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
        args.name,
        res['p_detection'],
        res['r_detection'],
        res['f_detection_05'],
        res['weighted_acc_detection'],
        res['substitute_rate'],
        res['ndcg_recommendation'],
        res['acc_recommendation'],
        res['p_e2e'],
        res['r_e2e'],
        res['f_e2e_05']
        ))

    confusion_data = {}
    confusion_data["changecorrect"] = changecorrect
    confusion_data["changeincorrect"] = changeincorrect
    confusion_data["notchangechange"] = notchangechange
    confusion_data["changenotchange"] = changenotchange
    confusion_data["notchangenotchange"] = notchangenotchange
    
    confusion_data["changecorrect_randomsample"] = changecorrect_randomsample
    confusion_data["changeincorrect_randomsample"] = changeincorrect_randomsample
    confusion_data["notchangechange_randomsample"] = notchangechange_randomsample
    confusion_data["changenotchange_randomsample"] = changenotchange_randomsample
    confusion_data["notchangenotchange_randomsample"] = notchangenotchange_randomsample
    
    confusion_data["changecorrect_samplefrommodel"] = changecorrect_samplefrommodel
    confusion_data["changeincorrect_samplefrommodel"] = changeincorrect_samplefrommodel
    confusion_data["notchangechange_samplefrommodel"] = notchangechange_samplefrommodel
    confusion_data["changenotchange_samplefrommodel"] = changenotchange_samplefrommodel
    confusion_data["notchangenotchange_samplefrommodel"] = notchangenotchange_samplefrommodel
    
    # Specify the filename
    filename = 'confusion_data.json'
    
    # Open the file in write mode and save the dictionary using dump
    with open(filename, 'w') as file:
        json.dump(confusion_data, file)
    
    print(f"The dictionary has been saved to {filename}")
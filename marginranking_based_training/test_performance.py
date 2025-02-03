import warnings

warnings.filterwarnings("ignore")

from config import Config

args = Config()

import os

import re
import time
import nltk
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from logger import logging, logging_params
from utils import load_state
from data import get_tokenizer
from train_utils import (
    set_seed,
    load_train_data,
    evaluate_mlm,
    getSubstitutePairs,
)

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

np.set_printoptions(threshold=np.inf)
tokenizer = get_tokenizer()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid


# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def load_model():
    if args.model_choice == 'mlm_cls':
        from model import mlm_cls as Model
    elif args.model_choice == 'seq2seq':
        from model import seq2seq as Model
    else:
        assert 1 == 2, 'Invalid argument model_choice'
    return Model



def evaluate(net, test_loader, args, mode='eval'):
    acc = 0;
    topkcov = 0;
    rank = []
    out_labels = [];
    true_labels = []
    net.eval()

    logging("Evaluating test samples...")
    res_dict = {}

    # Begin evaluating
    with torch.no_grad():
        if args.print_log:
            enumerate_loader = tqdm(enumerate(test_loader), total=len(test_loader))
        else:
            enumerate_loader = enumerate(test_loader)

        for i, data in enumerate_loader:

            input_ids, labels, mask_pos, input_tokens = data

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                mask_pos = mask_pos.cuda()
                #w_input = w_input.cuda()  # .detach()
                #w_label = w_label.cuda()  # .detach()

            if args.model_choice == 'mlm_cls':
                # print('Model:', type(net))
                # print('Devices:', net.device_ids)
                pred_sent_tokenids, classification_logits = net(input_ids, labels, mask_pos, 
                                                                is_training=False)
                (accuracy_batch, topkcov_batch, rank_batch, topk_pred), (o, l) = evaluate_mlm(classification_logits,
                                                                                              labels, mask_pos,
                                                                                              gettopk=True)

                input_sents_this_batch = input_ids.cpu().tolist()
                sent_lengths_this_batch = [sum([1 if k != args.pad_id else 0 for k in input_sents_this_batch[ii]]) for
                                           ii in range(len(o))]
                pred_sent_tokenids = [pred_sent_tokenids[ii][:sent_lengths_this_batch[ii]] for ii in range(len(o))]
                input_sents_this_batch = [input_sents_this_batch[ii][:sent_lengths_this_batch[ii]] for ii in
                                          range(len(o))]

                for ii in range(len(o)):
                    pred_words = tokenizer.decode(pred_sent_tokenids[ii][1:-1])
                    idx, input_words = input_tokens[ii].split("=+=+")
                    if args.plm_type == 'bert':
                        res = getSubstitutePairs(nltk.word_tokenize(pred_words),
                                                 nltk.word_tokenize(input_words.lower()), topk_pred[ii],
                                                 input_sents_this_batch[ii])
                    elif args.plm_type == 'roberta_base':
                        res = getSubstitutePairs(nltk.word_tokenize(pred_words),
                                                 nltk.word_tokenize(input_words.lower()), topk_pred[ii],
                                                 input_sents_this_batch[ii])
                        for i in range(len(res)):
                            if res[i][2] == -1: continue
                            for j in range(len(res[i][2])):
                                if res[i][2][j][1][0] == 'Ġ':
                                    res[i][2][j][1] = res[i][2][j][1][1:]

                    res_dict[idx] = {'input_words': nltk.word_tokenize(input_words),
                                     "pred_words": nltk.word_tokenize(pred_words),
                                     "substitute_topk": [[[k, lpos, rpos], [rr[1] for rr in r] if r != -1 else [v]] for
                                                         k, v, r, lpos, rpos in res]}

            elif args.model_choice == 'seq2seq':
                _, output_sequence = net(input_ids, labels, is_training=False)
                out_sents = tokenizer.batch_decode(output_sequence, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
                o, l, accuracy_batch, topkcov_batch, rank_batch = [], [], 0, 0, [[]]

                input_sents_this_batch = input_ids.cpu().tolist()
                sent_lengths_this_batch = [sum([1 if k != args.pad_id else 0 for k in input_sents_this_batch[ii]]) for
                                           ii in range(len(input_tokens))]
                input_sents_this_batch = [input_sents_this_batch[ii][:sent_lengths_this_batch[ii]] for ii in
                                          range(len(input_tokens))]

                for ii in range(len(input_tokens)):
                    pred_words = out_sents[ii]
                    idx, input_words = input_tokens[ii].split("=+=+")
                    res = getSubstitutePairs(nltk.word_tokenize(pred_words), nltk.word_tokenize(input_words),
                                             [[[-1, i]] for i in nltk.word_tokenize(pred_words)],
                                             input_sents_this_batch[ii])
                    for i in range(len(res)):
                        if res[i][2] == -1: continue
                        for j in range(len(res[i][2])):
                            if res[i][2][j][1][0] == 'Ġ':
                                res[i][2][j][1] = res[i][2][j][1][1:]

                    res_dict[idx] = {'input_words': nltk.word_tokenize(input_words),
                                     "pred_words": nltk.word_tokenize(pred_words),
                                     "substitute_topk": [[[k, lpos, rpos], [rr[1] for rr in r] if r != -1 else [v]] for
                                                         k, v, r, lpos, rpos in res]}

            out_labels += o
            true_labels += l
            acc += accuracy_batch
            topkcov += topkcov_batch
            rank += rank_batch

            # # count the persentage of proposing new substitutes
            # # NOT adapted on predicting multi-tokens ! #TBD
            # pred_batch_tokens, input_batch_tokens = sum(pred_sent_tokenids, []), sum(input_sents_this_batch, [])
            # propose_token_num += sum([0 if pred_batch_tokens[i] == input_batch_tokens[i] else 1 for i in range(len(input_batch_tokens))])
            # all_token_num += len(input_batch_tokens)

    accuracy = acc / (i + 1)
    topkcoverage = topkcov / (i + 1)
    rank = sum(rank, [])  # squeeze(0)
    rank_np = np.array(rank)
    avgrank, midrank = (sum(rank) / len(rank)) if rank else 0, np.median(rank_np)

    results = {
        "accuracy": accuracy,
        "topkcoverage": topkcoverage,
        "avgrank": avgrank,
        "midrank": midrank
    }

    with open(args.res_path + '_%s.json' % (mode), 'w') as f:
        json.dump(res_dict, f, indent=4)

    import sys
    sys.path.append('../../../evaluation/')
    from score import eval
    if mode == 'eval':
        results = eval(res_dict, args.dev_data_path)
    elif mode == 'test':
        results = eval(res_dict, args.test_data_path)
    logging("***** Eval results *****")
    for key in results.keys():
        logging("  %s = %s" % (key, str(results[key])))
    return results, (out_labels, true_labels)

import os

#.pth.tar
def find_files_with_extension(path, extension="last.3.pth.tar"):
    file_list = []

    # Check if the path exists
    if os.path.exists(path):
        # List all files in the directory
        files = os.listdir(path)

        # Filter files that end with the specified extension
        file_list = [file for file in files if file.endswith(extension)]

    return file_list

def infer(args):
    from data import load_dataloaders

    _, _, test_loader, _, _, test_len = load_dataloaders(args, "test")

    logging("Loaded %d Testing samples." % test_len)
    logging("Loading model...")

    Model = load_model()
    net = Model(args)
    if torch.cuda.is_available(): net.cuda()
    all_ckpt = find_files_with_extension(args.ckpt_path)

    # for ckpt_path in all_ckpt:
    #     checkpoint_path = os.path.join(args.ckpt_path, ckpt_path)
    #     print(ckpt_path)
    #     checkpoint = torch.load(checkpoint_path)
    #     logging("Loaded checkpoint model %s." % checkpoint_path)
    checkpoint = torch.load('../sws_margin_sow.pth.tar')
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        # best_pred = checkpoint['best_f_e2e_05']
        net.load_state_dict(checkpoint['state_dict'])
        logging("Loaded model and optimizer.")

    logging("Starting infering process...")

    # try:
    evaluate(net, test_loader, args, "test")
    # except Exception as e:
    #     print(e)


if __name__ == "__main__":

    set_seed(args.seed)
    logging_params()

    if args.infer:
        infer(args)

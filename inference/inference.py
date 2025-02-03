import warnings

warnings.filterwarnings("ignore")

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
import re
import random

# from logger import logging, logging_params
# from utils import load_state
# from data import get_tokenizer
# from train_utils import (
#     set_seed,
#     load_train_data,
#     evaluate_mlm,
#     getSubstitutePairs,
# )

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

np.set_printoptions(threshold=np.inf)
# tokenizer = get_tokenizer()


# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
from transformers import BertTokenizer as Tokenizer
tokenizer = Tokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def load_model():
      from model import mlm_cls as Model
      return Model


def preprocess_test_data(data_path='../sws_test.json'):
    with open(data_path, 'r') as f:
        j = json.load(f)
    testres = []
    for k, v in tqdm(j.items()):
        testres.append([(k, v['sentence']), [], [], []])
    return testres


testres = preprocess_test_data('../sws_test.json')

def preprocess_sentence(tokenizer, sent_list):
      '''input: [('9137937-00000030895551899852-8', 'However, the speaker'), [], [], []]
      output: tokenize sentence
      '''
      sent = sent_list[0][1]
      sent = re.sub("\s+", " ", sent)
      sent = re.sub("、", ",", sent)
      sent = re.sub("\(", " (", sent)
      sent = re.sub("\)", ") ", sent)
      sent = re.sub("\[", " [", sent)
      sent = re.sub("\]", "] ", sent)
      sent = re.sub(",", ", ", sent)
      sent = re.sub("\.", ". ", sent) # although it will cause "a.m." -> "a. m.", but make more sentence ends with a space, which is more relavent with substitution
      sent = re.sub("\?", "? ", sent)
      sent = re.sub("!", "! ", sent)
      sent = re.sub(" ,", ",", sent)
      sent = re.sub(" \.", ".", sent)
      sent = re.sub(" \?", "?", sent)
      sent = re.sub(" !", "!", sent)
      sent = re.sub("\s+", " ", sent)
      sent = re.sub("\. \"", ".\"", sent)
      sent = re.sub("\" \.", "\".", sent)
      sent = re.sub(", \"", ".\"", sent)
      sent = re.sub("\" ,", "\".", sent)
      while sent and sent[-1] == ' ': sent = sent[:-1]
      while sent and sent[0] == ' ': sent = sent[1:]
      if sent[-1] not in '.?!\"\'':
            # print(sent) 
            sent += '.'
      sent_token = [tokenizer.cls_token] + tokenizer.tokenize(sent) + [tokenizer.sep_token]
      input_ids = tokenizer.convert_tokens_to_ids(sent_token)
      input_ids = torch.tensor([input_ids])
      return input_ids

def find_top2word(input_ids, pred_sent_tokenids, classification_logits, tokenizer):
    """
    Compare input_ids and pred_sent_tokenids to find differences, select an index based on the differences,
    and decode the top 2 sorted classification logits for the selected index.

    Args:
    - input_ids (list): The first list of token IDs.
    - pred_sent_tokenids (list): The second list of token IDs to compare with input_ids.
    - classification_logits (torch.Tensor): A tensor of logits, typically from a model's output.
    - tokenizer: Tokenizer instance for decoding token IDs.

    Returns:
    - list: Decoded top 2 tokens for the selected index.
    """
    # Find indices of differences between input_ids and pred_sent_tokenids
    differences = [i for i, (x, y) in enumerate(zip(input_ids, pred_sent_tokenids)) if x != y]

    # Select index based on differences
    if len(differences) > 1:
        index = random.choice(differences)
    elif len(differences) == 1:
        index = differences[0]
    else:
        index = random.randint(1, len(input_ids) - 2)
      
    original_word = tokenizer.decode(input_ids[index])

    # Sort classification logits for the selected index
    sorted_values, sorted_indices = torch.sort(classification_logits[0][index], descending=True)

    # Decode top 2 sorted tokens
    top_tokens = [tokenizer.decode(i.tolist()) for i in sorted_indices[:3]]

    return (index, original_word), top_tokens

# load model
Model = load_model()
net = Model()
checkpoint = torch.load('../sws_margin_sow.pth.tar')
net.load_state_dict(checkpoint['state_dict'])

with open('human_study_ground_truth', 'w') as file:
      for i in range(len(testres)):
            input_ids = preprocess_sentence(tokenizer, testres[i])

            pred_sent_tokenids, classification_logits = net(input_ids, 
                                                      is_training=False)
            output = tokenizer.decode(pred_sent_tokenids[0])
            (index, original_word), top_tokens=find_top2word(input_ids.tolist()[0], pred_sent_tokenids[0], classification_logits, tokenizer)
            if original_word == top_tokens[0]:
                  change_option = 0
                  testres[i][2].append(top_tokens[1:])
            else:
                  change_option = 1
                  testres[i][2].append(top_tokens[:2])
                  
            testres[i][1].append((index, original_word))
            
            testres[i][3].append(change_option)
            # print(testres[i][0][1])
            # print(output[6:])
            file.write(', '.join(map(str, testres[i])) + '\n')



print(1)
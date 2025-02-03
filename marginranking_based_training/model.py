import os
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BartForConditionalGeneration
    )
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from config import Config
args = Config()

from transformers import BertTokenizer as Tokenizer
from bartscore import BARTScorer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import collections
from scipy.stats import spearmanr
import numpy as np

import string

class mlm_cls(nn.Module):
    '''
    classical MLM with new initialized MLM head
    '''
    def __init__(self, args):#, device):
        super().__init__()
        if args.plm_type == 'bert':
            self.bert = BertModel.from_pretrained(args.bert_path, output_hidden_states=True)

            config = BertConfig.from_pretrained(args.bert_path)
            self.cls = BertOnlyMLMHead(config) # Linear -> GELU -> LayerNorm -> Linear

        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.special_tokens = self.tokenizer.all_special_ids
        self.punctuation = string.punctuation
        self.model = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        #self.model = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        self.model.load(path='../llama_p_value/bartscore/bart_score.pth')
        self.lamb = 1
        self.pos_hash = {
            'plural noun': ('NNS', 'NNPS'),
            'preposition': ('IN'),
            'adverb': ('RB', 'RBR', 'RBS', 'WRB'),
            'conjunction': ('CC'),
            'verb': ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'),
            'noun': ('NN', 'NNP'),
            'phrase': (),
            'interjection': (),
            'pronoun': ('PRP$', 'PRP'),
            'adjective': ('JJ', 'JJR', 'JJS')
        }
        self.rev_pos_hash = collections.defaultdict(lambda: '')
        for k, v in self.pos_hash.items():
            for vv in v:
                self.rev_pos_hash[vv] = k
    def initial_trainlayer_weight(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        # for param in self.bert.encoder.layer[-3].parameters():
        #     param.requires_grad = True
        # for param in self.bert.encoder.layer[-2].parameters():
        #     param.requires_grad = True
        # for param in self.bert.encoder.layer[-1].parameters():
        #     param.requires_grad = True
        for layer in self.bert.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = True

    def create_unchanged_mask(self,input_id, unchanged_mask):
        batch_size = input_id.size(0)

        input_id_sent = []

        for batch_index in range(batch_size):
            sent_id = input_id[batch_index].tolist()
            cutoff = sent_id.index(102)
            input_id_sent.append(self.tokenizer.decode(sent_id[1:cutoff]))


        self_change_mask = []
        for sent in input_id_sent:
            sent_tokenize = self.tokenizer.tokenize(sent)
            poses = nltk.pos_tag(sent_tokenize)
            # Create mask tensor
            mask = torch.zeros(input_id.size(1), dtype=torch.int)

            for i, (word, pos) in enumerate(poses):
                if i <len(poses)-1: #-1 because poses[i+1] 
                    if (pos in self.rev_pos_hash and word not in set(stopwords.words('english')) #and word not in self.punctuation #we allow self.punctuation
                            and not poses[i+1][0].startswith("##") and not word.startswith("##")):# and i!=0 and i!=len(pos):
                        mask[i+1] = 1 #i+1 because count in start token
                if i == len(poses)-1: #we wanna consider the last token '.'
                    mask[i+1] = 1 #i+1 because count in start token

            self_change_mask.append(mask)
        self_change_mask = torch.stack(self_change_mask).to(unchanged_mask.device)
        self_train_unchange = (unchanged_mask * self_change_mask)
        self_train_unchange = (self_train_unchange == 1).int()
        #self_train_unchange = self_change_mask

        def randomly_keep_k_values(mask, k=5):
            import random
            new_mask = torch.zeros_like(mask)
            for i in range(mask.size(0)):
                # Find indices of 1s
                ones_indices = (mask[i] == 1).nonzero().view(-1)
                # Randomly select k indices if there are more than k 1s
                if len(ones_indices) > k:
                    selected_indices = random.sample(ones_indices.tolist(), k)
                else:
                    selected_indices = ones_indices.tolist()
                # Set selected indices to 1 in the new mask
                new_mask[i, selected_indices] = 1
            return new_mask

        self_train_unchange_mask = randomly_keep_k_values(self_train_unchange)
        return self_train_unchange_mask

    def cosine_similarity_loss(self, logit_tensor, comparison_number):
        # Calculate cosine similarity
        cosine_sim = nn.functional.cosine_similarity(logit_tensor, comparison_number, dim=1)

        # Calculate the mean squared error as the loss
        loss = torch.sum(-torch.exp(cosine_sim))  # Substracting 1 to push cosine_sim towards 1

        return loss


    def calculate_loss(self, input_id, sent_logit_all, mask_id, labels=None,margin=1, option='target', k=5, alpha=1):

        #probabilities = nn.functional.log_softmax(sent_logit_all, dim=-1)

        batch_size = sent_logit_all.size(0)

        oris = []
        subs = []
        normal_oris = []
        normal_cutoff = []#store group cutoff for normalize

        logit_targets = []
        logit_labels = []
        for batch_index in range(batch_size):
            # find all the place with value 1 in the current batch
            positions = torch.nonzero(mask_id[batch_index]).squeeze(1)
            sent_id = input_id[batch_index].tolist()
            cutoff = sent_id.index(102)
            sent_id = sent_id[1:cutoff]
            oris.extend(
                [self.tokenizer.decode(sent_id)] * (k) * len(positions))  # create original sentence

            normal_oris.append(self.tokenizer.decode(sent_id))
            normal_cutoff.append(k*len(positions))

            for pos_index in positions.tolist():
                # extract current position's logit_target
                logit_target = sent_logit_all[batch_index, pos_index]#pos_index because start token

                sorted_values, sorted_indices = torch.sort(logit_target, descending=True)
                count = 0
                i = 0
                topk_idx = []
                topk_to_take = []
                while count < k:
                    
                    if sorted_indices[i] not in self.special_tokens:
                        topk_idx.append(sorted_indices[i].item())
                        topk_to_take.append(i)
                        count += 1
                    i += 1

                result_model = sorted_values[topk_to_take]

                logit_targets.append(result_model)

                ori = list(sent_id) 
                pos_index = pos_index - 1  # becuase we remove the start token for bartscore calculation
                for idx, value in enumerate(topk_idx):
                    ori[pos_index] = value
                    subs.append(self.tokenizer.decode(ori))

        # Find the length of the longest string
        longest_length = max(len(s) for s in oris)
        # print(longest_length)
        if longest_length<50:
            batch_size=500
        elif 50<=longest_length<100:
            batch_size=250
        elif 100<=longest_length<200:
            batch_size=150
        elif 200<=longest_length<500:
            batch_size=100  
        elif 500<=longest_length<800:
            batch_size=25
        else:
            batch_size=5

        ori_scores = self.model.score(normal_oris, normal_oris, batch_size=batch_size)
        result_scores = self.model.score(oris, subs, batch_size=batch_size)
        
        #normalize result_scores
        data_list = np.array(result_scores)
        index = 0
        normalized_score = []
        # Iterate through each group, performing segmentation and normalization
        for cutoff, norm_value in zip(normal_cutoff, ori_scores):
            group = data_list[index:index + cutoff]
            # Normalize the group: subtract the corresponding norm_value from each element and compute the exponent  
            normalized_group = np.exp(group - norm_value)
            # Add the processed group data to the result list  
            normalized_score.extend(normalized_group)
            # Update the index
            index += cutoff


        logit_targets = torch.stack(logit_targets) #[sub size,top k]

        # if labels!=None:
        #     logit_labels = torch.stack(logit_labels)
        device = logit_targets.device # Get the device of logit_targets
        result_scores = torch.tensor(result_scores, device=device).reshape(logit_targets.size(0), -1)
        result_ratios = torch.tensor(normalized_score, device=device).reshape(logit_targets.size(0), -1)

        #total_sum = torch.sum(torch.tensor(normalized_score, device=device).reshape(logit_targets.size(0), -1)).item()/logit_targets.size(0)
        total_sum = torch.sum(result_ratios).item() / logit_targets.size(0)
        print(total_sum)

        cosine_correlation = torch.sum(nn.functional.cosine_similarity(logit_targets, result_scores, dim=1)).item()/logit_targets.size(0)
        print(cosine_correlation)
        # cosine_correlation = torch.sum(
        #     nn.functional.cosine_similarity(logit_targets, result_scores, dim=1)).item() / logit_targets.size(0)
        # print(cosine_correlation)

        if option=='ce':
            sorted_tensor = self.custom_sort(logit_targets, result_scores)
            loss = self.RankingLoss(sorted_tensor, summary_score=None, margin=margin, gold_margin=0,
                                    gold_weight=1, no_gold=True, no_cand=False)/logit_targets.size(0)

        elif option=='target':
            sorted_tensor = self.custom_sort(logit_targets, result_scores)
            loss = self.RankingLoss(sorted_tensor, summary_score=logit_labels, margin=margin, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False)/logit_targets.size(0)


        elif option=='weightce':
            normalized_tensor = torch.nn.functional.softmax(result_scores, dim=1) # Apply softmax along dimension 1
            loss = -torch.sum(torch.mul(logit_targets, normalized_tensor))/logit_targets.size(0)


        elif option=='argmax':
            max_values1, _ = result_scores.max(dim=1, keepdim=True)  # argmax
            normalized_tensor = torch.where(result_scores == max_values1, torch.tensor(1.0), torch.tensor(0.0))
            loss = -torch.sum(torch.mul(logit_targets, normalized_tensor)) /logit_targets.size(0)

        elif option=='sow':
            loss = -torch.sum(torch.mul(result_scores, torch.exp(logit_targets))) /logit_targets.size(0)

        elif option=='cos_sow1':
            loss1 = self.cosine_similarity_loss(logit_targets, result_scores) / logit_targets.size(0)
            loss2 = -torch.sum(torch.mul(result_scores, torch.exp(logit_targets))) /logit_targets.size(0)
            loss = loss1 + loss2

        elif option=='cos_sow2':
            loss1 = self.cosine_similarity_loss(logit_targets, result_scores) / logit_targets.size(0)
            loss2 = -torch.sum(torch.mul(torch.exp(result_scores), torch.exp(logit_targets))) /logit_targets.size(0)
            loss = loss1 + loss2

        elif option=='cos_sow3':
            loss1 = self.cosine_similarity_loss(logit_targets, result_ratios) / logit_targets.size(0) # bad result normally if only this
            loss2 = -torch.sum(torch.mul(torch.exp(result_scores), logit_targets)) /logit_targets.size(0) # too weak
            loss = loss1 + loss2 # doesn't work

        elif option=='margin_sow1':
            sorted_tensor = self.custom_sort(logit_targets, result_scores)
            loss1 = self.RankingLoss(sorted_tensor, summary_score=None, margin=margin, gold_margin=0,
                                    gold_weight=1, no_gold=True, no_cand=False)/logit_targets.size(0)
            loss2 = -torch.sum(torch.mul(result_scores, torch.exp(logit_targets))) /logit_targets.size(0)
            loss = loss1 + loss2

        elif option=='margin_sow2':
            sorted_tensor = self.custom_sort(logit_targets, result_scores)
            loss1 = self.RankingLoss(sorted_tensor, summary_score=None, margin=margin, gold_margin=0,
                                    gold_weight=1, no_gold=True, no_cand=False)/logit_targets.size(0)
            loss2 = -torch.sum(torch.mul(torch.exp(result_scores), torch.exp(logit_targets))) /logit_targets.size(0)
            loss = loss1 + loss2

        elif option == 'margin_sow1_logit':
            sorted_tensor = self.custom_sort(logit_targets, result_scores)
            loss1 = self.RankingLoss(sorted_tensor, summary_score=None, margin=margin, gold_margin=0,
                                     gold_weight=1, no_gold=True, no_cand=False) / logit_targets.size(0)
            loss2 = -torch.sum(torch.mul(result_scores, logit_targets)) / logit_targets.size(0)
            loss = loss1 + loss2

        return loss,total_sum,cosine_correlation



    def custom_sort(self, tensor1, tensor2):
        # Stack the two tensors along the third dimension for sorting
        concatenated = torch.stack([tensor1, tensor2], dim=2)
        # Sort based on the second column (tensor2) in descending order
        sorted_concatenated, indices = torch.sort(concatenated[:, :, 1], descending=True)
        # Reorder the first column (tensor1) using the sorted indices
        sorted_tensor1 = torch.gather(concatenated[:, :, 0], 1, indices)
        return sorted_tensor1

    def RankingLoss(self, score, summary_score=None, margin=1, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
        ones = torch.ones_like(score)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)
        # candidate loss
        n = score.size(1)
        if not no_cand:
            for i in range(1, n):
                pos_score = score[:, :-i]
                neg_score = score[:, i:]
                pos_score = pos_score.contiguous().view(-1)
                neg_score = neg_score.contiguous().view(-1)
                ones = torch.ones_like(pos_score)
                loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='sum')
                loss = loss_func(pos_score, neg_score, ones)
                TotalLoss += loss
        if no_gold:
            return TotalLoss
        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones_like(pos_score)
        loss_func = torch.nn.MarginRankingLoss(gold_margin)
        TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
        return TotalLoss

    def forward(self, input_ids, labels, mask_pos, is_training=True):

        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state # [bsz, sent_length, hidden_size]

        logits_all = self.cls(last_hidden_state) # [bsz, sent_length, vocab_size]
        probabilities = nn.functional.log_softmax(logits_all, dim=-1) # [bsz, sent_length, vocab_size]

        if is_training:


            unchanged_mask = torch.ones_like(mask_pos, dtype=torch.int32) # Create a tensor for cross entropy

            self_train_unchange_mask = self.create_unchanged_mask(input_ids, unchanged_mask) #create target word poistion

            contras_loss,total_sum,cosine_correlation = self.calculate_loss(input_ids, probabilities, self_train_unchange_mask, labels=None,margin=0.5,option='margin_sow1')

            return contras_loss,total_sum,cosine_correlation, (logits_all.permute(0, 2, 1))

        else:

            pred_sent_tokenids = self.cls(last_hidden_state).max(2)[1].cpu().tolist()
            return pred_sent_tokenids, (logits_all.permute(0, 2, 1))

class seq2seq(nn.Module):
    '''
    BART for seq2seq generation.
    '''
    def __init__(self, args):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(args.bart_path)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, input_ids, labels, is_training=True):

        loss = self.bart(input_ids=input_ids, labels=labels).loss
        if is_training: return loss, 0
        output_sequence = self.bart.generate(input_ids, num_beams=4, min_length=1, max_length=input_ids.shape[1])
        return loss, output_sequence

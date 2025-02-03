import os
import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertConfig,
    BartForConditionalGeneration
    )
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class mlm_cls(nn.Module):
    '''
    classical MLM with new initialized MLM head
    '''
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        config = BertConfig.from_pretrained('bert-base-uncased')
        self.cls = BertOnlyMLMHead(config) # Linear -> GELU -> LayerNorm -> Linear
        self.lsm = torch.nn.Softmax(dim=2)  # softmax

    def forward(self, input_ids):

        outputs = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state # [bsz, sent_length, hidden_size]

        logits_all = self.cls(last_hidden_state) # [bsz, sent_length, vocab_size]
        #logits_all = logits_all.permute(0, 2, 1) # [bsz, vocab_size, sent_length]
        logits_all = self.lsm(logits_all)

        pred_sent_tokenids = self.cls(last_hidden_state).max(2)[1].cpu().tolist()
        return pred_sent_tokenids, (logits_all)

def infer(sentence, template):#, tokenizer, model
    '''
    input: sentence, template
    find the tgt word from two input, feed to the model and return the logit
    '''
    from transformers import BertTokenizer as Tokenizer
    tokenizer = Tokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    vocab = tokenizer.get_vocab()
    vocab = list(vocab.keys())

    sent = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(sent)

    tmp = ['[CLS]'] + tokenizer.tokenize(template) + ['[SEP]']
    tmp_ids = tokenizer.convert_tokens_to_ids(tmp)

    for i, (elem1, elem2) in enumerate(zip(input_ids, tmp_ids)):
        if elem1 != elem2:
            offset = i

    net = mlm_cls()
    if torch.cuda.is_available():
        net.cuda()
        input_ids = torch.LongTensor([input_ids]).cuda()

    checkpoint = torch.load('last.9.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])

    pred_sent_tokenids, classification_logits = net(input_ids)
    tgt_logit = classification_logits[0][offset].cpu().tolist()
    # pred_words = tokenizer.decode(pred_sent_tokenids[0][offset])
    # print(pred_words)
    topk1 = pred_sent_tokenids[0][offset]
    tgt_logit_remove_topk1 = tgt_logit[topk1]
    vocab_remove_topk1 = vocab[topk1]
    tgt_logit.pop(topk1)
    vocab_remove_topk1 = vocab.copy()
    vocab_remove_topk1.pop(topk1)

    return tgt_logit

if __name__ == '__main__':

    sentence = 'today is a good day, hopefully i can get a nice paper soon'
    template = 'today is a good day, hopefully i can get a excellent paper soon'

    infer(sentence, template)
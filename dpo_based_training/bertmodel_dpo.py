import json
import time
import nltk
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Optional, Tuple, Union

from transformers.models.bert.modeling_bert import *
import numpy as np
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.train_layer = nn.ModuleList([BertLayer(config) for _ in range(12)])
        # self.train_layer._name = "trainable_layer0"
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if i == len(self.layer)-1:
                    layer_outputs_train = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                    hidden_states_train = layer_outputs_train[0]

                    if output_attentions:
                        all_self_attentions_train = all_self_attentions + (layer_outputs_train[1],)
                        if self.config.add_cross_attention:
                            all_cross_attentions_train = all_cross_attentions + (layer_outputs_train[2],)

                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                if i >= len(self.layer)-len(self.train_layer):
                    if i==len(self.layer)-len(self.train_layer):hidden_states_train=hidden_states
                    layer_outputs_train = self.train_layer[i-(len(self.layer)-len(self.train_layer))](
                        hidden_states_train,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                    hidden_states_train = layer_outputs_train[0]

                    if output_attentions:
                        all_self_attentions_train = all_self_attentions + (layer_outputs_train[1],)
                        if self.config.add_cross_attention:
                            all_cross_attentions_train = all_cross_attentions + (layer_outputs_train[2],)


            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_hidden_states_train = all_hidden_states + (hidden_states_train,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ),BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states_train,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states_train,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertModel_dpo(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.pooler_train = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs,encoder_outputs_train = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        sequence_output_train = encoder_outputs_train[0]
        pooled_output_train = self.pooler_train(sequence_output_train) if self.pooler_train is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ),BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output_train,
            pooler_output=pooled_output_train,
            past_key_values=encoder_outputs_train.past_key_values,
            hidden_states=encoder_outputs_train.hidden_states,
            attentions=encoder_outputs_train.attentions,
            cross_attentions=encoder_outputs_train.cross_attentions,
        )

class BertForMaskedLM_dropout(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel_dpo(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.sftmx = nn.Softmax(-1)

# from transformers import BertConfig
#
# # Load the default BERT base uncased configuration
# config = BertConfig.from_pretrained('bert-base-uncased')
# config.num_hidden_layers=3
#
# # model = BertForMaskedLM_dropout.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM_dropout(config)
# # for name, module in model.named_modules():
# #     print(f"Module name: {name}\nModule: {module}\n")
#
# for name, module in model.named_modules():
#     print(name, module.__class__.__name__)


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
            self.bert = BertModel_dpo.from_pretrained(args.bert_path, output_hidden_states=True)


            config = BertConfig.from_pretrained(args.bert_path)
            self.cls = BertOnlyMLMHead(config) # Linear -> GELU -> LayerNorm -> Linear

            self.cls_train = BertOnlyMLMHead(config)  # Linear -> GELU -> LayerNorm -> Linear
        for name, param in self.bert.named_parameters():
            if 'train_layer' in name:
                param.requires_grad = True
                param.training = True
            elif 'pooler_train' in name:
                param.requires_grad = True
                param.training = True
            else:
                param.requires_grad = False
                param.training = False
        # 冻结所有层
        for param in self.cls.parameters():
            param.requires_grad = False
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

    def activate_layer(self):
        # freeze all the layer
        for name, param in self.bert.named_parameters():
            if 'train_layer' in name:
                param.requires_grad = True
                param.training = True
            elif 'pooler_train' in name:
                param.requires_grad = True
                param.training = True
            else:
                param.requires_grad = False
                param.training = False
        # freeze all the layer
        for param in self.cls.parameters():
            param.requires_grad = False
            # param.training = False

        self.bert.encoder.train_layer.training=True
        # self.bert.embeddings.training=False
        self.bert.pooler_train.training = True
        self.cls_train.training=True
    def initial_trainlayer_weight(self):
        # copy cls weight to the cls_train
        cls_state_dict = self.cls.state_dict()
        cls_train_state_dict = self.cls_train.state_dict()

        # update cls_train's state_dict，cope cls's weight
        cls_train_state_dict.update(cls_state_dict)

        # load updated state_dict to cls_train
        self.cls_train.load_state_dict(cls_train_state_dict)

        # obtain weight
        for idx in range(len(self.bert.encoder.train_layer)):
            exact_layer = len(self.bert.encoder.train_layer)-idx
            source_layer = self.bert.encoder.layer[-exact_layer]
            target_layer = self.bert.encoder.train_layer[idx]

            source_weights = source_layer.state_dict()
            target_weights = target_layer.state_dict()

            # copy weight
            for name, param in source_weights.items():
                if name in target_weights:
                    target_weights[name].copy_(param)

            # make sure the gradient could be updated
            for param in target_weights.values():
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

    def topk_filter_special_tokens(self,logit_target_ref, k=5):
        sorted_values, sorted_indices = torch.sort(logit_target_ref, descending=True)
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

        return sorted_values[topk_to_take], topk_idx

    def calculate_loss(self, input_id, logit_ref, logit_train, mask_id, labels=None, margin=1, option='target', k=5, alpha=1):

        batch_size = logit_ref.size(0)

        oris = []
        subs = []
        normal_oris = []
        normal_cutoff = []#store group cutoff for normalize

        logit_targets_train = []
        logit_targets_ref = []
        for batch_index in range(batch_size):

            positions = torch.nonzero(mask_id[batch_index]).squeeze(1)
            sent_id = input_id[batch_index].tolist()
            cutoff = sent_id.index(102)
            sent_id = sent_id[1:cutoff]
            oris.extend(
                [self.tokenizer.decode(sent_id)] * (k) * len(positions))  # create original sentence

            normal_oris.append(self.tokenizer.decode(sent_id))
            normal_cutoff.append(k*len(positions))

            for pos_index in positions.tolist():

                logit_target_ref = logit_ref[batch_index, pos_index]#pos_index because start token
                logit_target_train = logit_train[batch_index, pos_index]

                result_model_train, topk_idx=self.topk_filter_special_tokens(logit_target_train,k)
                result_model_ref = logit_target_ref[topk_idx]

                logit_targets_train.append(result_model_train)
                logit_targets_ref.append(result_model_ref)

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

        for cutoff, norm_value in zip(normal_cutoff, ori_scores):
            group = data_list[index:index + cutoff]
            normalized_group = np.exp(group - norm_value)
            normalized_score.extend(normalized_group)
            index += cutoff


        logit_targets_train = torch.stack(logit_targets_train) #[sub size,top k]
        logit_targets_ref = torch.stack(logit_targets_ref)  # [sub size,top k]
        logit_targets_ref.requires_grad_(False) #not trainable

        # if labels!=None:
        #     logit_labels = torch.stack(logit_labels)
        device = logit_targets_train.device # Get the device of logit_targets
        result_scores = torch.tensor(result_scores, device=device).reshape(logit_targets_train.size(0), -1)
        result_ratios = torch.tensor(normalized_score, device=device).reshape(logit_targets_train.size(0), -1)

        #total_sum = torch.sum(torch.tensor(normalized_score, device=device).reshape(logit_targets.size(0), -1)).item()/logit_targets.size(0)
        total_sum = torch.sum(result_ratios).item() / logit_targets_train.size(0)
        print(total_sum)

        cosine_correlation = torch.sum(nn.functional.cosine_similarity(logit_targets_train, result_scores, dim=1)).item()/logit_targets_train.size(0)
        print(cosine_correlation)

        if option == 'dpo':
            sorted_train = self.custom_sort(logit_targets_train, result_scores)
            sorted_ref = self.custom_sort(logit_targets_ref, result_scores)

            loss = self.dpo_loss(sorted_train, sorted_ref, beta=0.1)/logit_targets_train.size(0)

        return loss,total_sum,cosine_correlation

    def dpo_loss(self, sorted_train, sorted_ref, beta=1):
        # numerator = torch.exp(beta * (sorted_train - sorted_ref))[:, :-1]
        # denominator = torch.exp(beta * (sorted_train - sorted_ref))[:, 1:]
        # loss = -torch.sum(numerator - denominator)

        # numerator = (sorted_train - sorted_ref)[:, :-1] #dpo-sigma
        # denominator = (sorted_train - sorted_ref)[:, 1:]
        # loss = -torch.sum(torch.nn.functional.logsigmoid(numerator - denominator))

        numerator = (sorted_train - sorted_ref)[:, :-1] #dpo*
        denominator = (sorted_train - sorted_ref)[:, 1:]
        loss = -torch.sum(numerator - denominator)

        return loss

    def custom_sort(self, tensor1, tensor2):
        concatenated = torch.stack([tensor1, tensor2], dim=2)
        sorted_concatenated, indices = torch.sort(concatenated[:, :, 1], descending=True)
        sorted_tensor1 = torch.gather(concatenated[:, :, 0], 1, indices)
        return sorted_tensor1


    def forward(self, input_ids, labels, mask_pos, is_training=True):

        outputs, outputs_train = self.bert(input_ids)
        last_hidden_state = outputs.last_hidden_state # [bsz, sent_length, hidden_size]
        last_hidden_state_train = outputs_train.last_hidden_state  # [bsz, sent_length, hidden_size]

        logits_ref = self.cls(last_hidden_state) # [bsz, sent_length, vocab_size]
        logits_train = self.cls_train(last_hidden_state_train) # [bsz, sent_length, vocab_size]
        #logits_all = logits_all.permute(0, 2, 1) # [bsz, vocab_size, sent_length]
        proba_ref = nn.functional.log_softmax(logits_ref, dim=-1) # [bsz, sent_length, vocab_size]
        proba_train = nn.functional.log_softmax(logits_train, dim=-1)  # [bsz, sent_length, vocab_size]

        if is_training:

            all_mask = torch.ones_like(mask_pos, dtype=torch.int32) # Create a tensor for cross entropy
            random_mask = self.create_unchanged_mask(input_ids, all_mask) #create target word poistion

            dpo_loss,total_sum,cosine_correlation = self.calculate_loss(input_ids, proba_ref, proba_train, random_mask,option='dpo')

            return dpo_loss,total_sum,cosine_correlation, (logits_train.permute(0, 2, 1))
        else:
            pred_sent_tokenids = self.cls(last_hidden_state).max(2)[1].cpu().tolist()
            return pred_sent_tokenids, (logits_train.permute(0, 2, 1))


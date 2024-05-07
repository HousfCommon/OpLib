#--just practice--
import torch
import torch.nn as nn
import math
from transformers.activations import gelu, gelu_new
import nn.GoldenModelOfALBert.Layer as L
import numpy as np
from nn.GoldenModelOfALBert.Trans_Binary_and_Decimal import *


ACT2FN = {"gelu": gelu,
          "relu": torch.nn.functional.relu,
          # "swish": swish,
          "gelu_new": gelu_new,
          # "mish": mish
          }


class AlbertAttention(nn.Module):
    def __init__(self, config, param_list):
        super(AlbertAttention, self).__init__()
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = L.LinearLayer(config.hidden_size, self.all_head_size,
                                   param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight'],
                                   param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias'])
        self.key = L.LinearLayer(config.hidden_size, self.all_head_size,
                                 param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight'],
                                 param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias'])
        self.value = L.LinearLayer(config.hidden_size, self.all_head_size,
                                   param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight'],
                                   param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias'])

        self.dense = L.LinearLayer(config.hidden_size, config.hidden_size,
                                   param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight'],
                                   param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias'])

        # todo replace layernorm
        self.LayerNorm = L.LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                     weight=param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight'],
                                     bias=param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            head_mask=None
    ):
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        # print("input_id", input_ids.shape)
        # print("mixed_query_layer", mixed_query_layer.shape)

        fra_bw = Get_BitWidth_of_Decimal(mixed_query_layer, 'query', 16)
        mixed_query_layer = Transform_D_To_B(mixed_query_layer, int_bit_width=16 - fra_bw, tol_bit_width=16)
        fra_bw = Get_BitWidth_of_Decimal(mixed_key_layer, 'key', 16)
        mixed_key_layer = Transform_D_To_B(mixed_key_layer, int_bit_width=16 - fra_bw, tol_bit_width=16)
        fra_bw = Get_BitWidth_of_Decimal(mixed_value_layer, 'value', 16)
        mixed_value_layer = Transform_D_To_B(mixed_value_layer, int_bit_width=16 - fra_bw, tol_bit_width=16)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # print("trans_size", query_layer.shape, key_layer.transpose(-1, -2).shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print("score", attention_scores.shape)

        fra_bw = Get_BitWidth_of_Decimal(attention_scores, 'score', 16)
        attention_scores = Transform_D_To_B(attention_scores, int_bit_width=16 - fra_bw, tol_bit_width=16)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # print("attention_scores dim -1 is", attention_scores.shape[-1])

        # Normalize the attention scores to probabilities.
        # todo softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # print("attention_probs", attention_probs.shape)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        fra_bw = Get_BitWidth_of_Decimal(attention_probs, 'attention_probs', 16)
        attention_probs = Transform_D_To_B(attention_probs, int_bit_width=16 - fra_bw, tol_bit_width=16)

        context_layer = torch.matmul(attention_probs, value_layer)
        # print("context", context_layer.shape)

        fra_bw = Get_BitWidth_of_Decimal(context_layer, 'context layer', 16)
        context_layer = Transform_D_To_B(context_layer, int_bit_width=16 - fra_bw, tol_bit_width=16)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # print("permute_context", context_layer.shape)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
                .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
                .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)
        # print("w, b", w.shape, b.shape)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        # print("projected_context_layer", projected_context_layer.shape)
        fra_bw = Get_BitWidth_of_Decimal(projected_context_layer, 'projeted context', 16)
        projected_context_layer = Transform_D_To_B(projected_context_layer, int_bit_width=16 - fra_bw, tol_bit_width=16)
        # max_ceil_pj = torch.max(torch.ceil(projected_context_layer))
        # min_floor_pj = torch.min(torch.floor(projected_context_layer))
        # print("range of projected context is (%d , %d) " % (min_floor_pj.numpy(), max_ceil_pj.numpy()),
        #       "Bitwidth of Integer is %d " %(torch.log(torch.max(torch.abs(min_floor_pj), max_ceil_pj)).numpy()/math.log(2) + 1))

        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer)
        print("pre", layernormed_context_layer[0])
        fra_bw = Get_BitWidth_of_Decimal(layernormed_context_layer, 'layernormed_context_layer', 16)
        layernormed_context_layer = Transform_D_To_B(layernormed_context_layer, int_bit_width=16 - fra_bw, tol_bit_width=16)
        print("post", layernormed_context_layer[0])
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):
    def __init__(self, config, param_list):
        super(AlbertLayer, self).__init__()

        self.config = config
        # todo replace layernorm
        self.full_layer_layer_norm = L.LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                                  weight=param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight'],
                                                  bias=param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias']
                                                  )
        self.attention = AlbertAttention(config, param_list)
        # todo layer fusion
        self.ffn = L.LinearLayer(config.hidden_size, config.intermediate_size,
                                 param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight'],
                                 param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.ffn.bias'])
        self.ffn_output = L.LinearLayer(config.intermediate_size, config.hidden_size,
                                        param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight'],
                                        param_list['albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias'])
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])

        Get_BitWidth_of_Decimal(ffn_output, 'ffn1', 16)
        # max_ceil_ffn1 = torch.max(torch.ceil(ffn_output))
        # min_floor_ffn1 = torch.min(torch.floor(ffn_output))
        # print("range of ffn1 is (%d , %d) " % (min_floor_ffn1.numpy(), max_ceil_ffn1.numpy()),
        #       "Bitwidth of Integer is %d " %(torch.log(torch.max(torch.abs(min_floor_ffn1), max_ceil_ffn1)).numpy()/math.log(2) + 1))
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        Get_BitWidth_of_Decimal(ffn_output, 'ffn2', 16)
        # max_ceil_ffn2 = torch.max(torch.ceil(ffn_output))
        # min_floor_ffn2 = torch.min(torch.floor(ffn_output))
        # print("range of ffn2 is (%d , %d) " % (min_floor_ffn2.numpy(), max_ceil_ffn2.numpy()),
        #       "Bitwidth of Integer is %d " %(torch.log(torch.max(torch.abs(min_floor_ffn2), max_ceil_ffn2)).numpy()/math.log(2) + 1))
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        Get_BitWidth_of_Decimal(hidden_states, 'layernormed_output', 16)
        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class AlbertLayerGroup(nn.Module):
    def __init__(self, config, param_list):
        super(AlbertLayerGroup, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([AlbertLayer(config, param_list) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            if head_mask is None:
                layer_output = albert_layer(hidden_states, attention_mask, head_mask)
            else:
                layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(nn.Module):
    def __init__(self, config, param_list):
        super(AlbertTransformer, self).__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = L.LinearLayer(config.embedding_size, config.hidden_size,
                                                         param_list['albert.encoder.embedding_hidden_mapping_in.weight'],
                                                         param_list['albert.encoder.embedding_hidden_mapping_in.bias'])
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config, param_list) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            if head_mask is None:
                layer_group_output = self.albert_layer_groups[group_idx](
                    hidden_states,
                    attention_mask,
                    head_mask
                )
            else:
                layer_group_output = self.albert_layer_groups[group_idx](
                    hidden_states,
                    attention_mask,
                    head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group]
                )

            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # no use
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        if current_layer == 0:
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        else:
            hidden_states = hidden_states[0]

        layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

        # Index of the hidden group
        group_idx = int(current_layer / (self.config.num_hidden_layers / self.config.num_hidden_groups))

        # Index of the layer inside the group
        layer_idx = int(current_layer - group_idx * layers_per_group)

        layer_group_output = self.albert_layer_groups[group_idx](hidden_states, attention_mask, head_mask[group_idx * layers_per_group:(group_idx + 1) * layers_per_group])
        hidden_states = layer_group_output[0]

        return (hidden_states,)


class AlbertPooler():
    def __init__(self, config, param_list):
        super(AlbertPooler, self).__init__()
        self.pooler = L.LinearLayer(config.hidden_size, config.hidden_size,
                                    param_list['albert.pooler.weight'],
                                    param_list['albert.pooler.bias'])
        # todo tanh()
        self.pooler_activation = nn.Tanh()

    def forward(self, inputs):
        pooled_output = self.pooler_activation(self.pooler(inputs))
        return pooled_output


def main():
    y = torch.randn(20, 64)
    y_r = y.view(20, 8, 8)
    y_r = y_r.permute(1, 0, 2).contiguous()

    net_1 = nn.Softmax(dim=0)
    net_2 = nn.Softmax(dim=1)
    net_3 = nn.Softmax(dim=2)

    return 0


if __name__ == '__main__':
    main()
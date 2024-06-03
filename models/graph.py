import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os
import math
from torch_geometric.nn import GCNConv, GATConv


class GraphAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, graph_type):
        super(GraphLayer, self).__init__()
        self.config = config

        self.graph_type = graph_type
        if self.graph_type == 'graphormer':
            self.graph = GraphAttention(config.hidden_size, config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        elif self.graph_type == 'GCN':
            self.graph = GCNConv(config.hidden_size, config.hidden_size)
        elif self.graph_type == 'GAT':
            self.graph = GATConv(config.hidden_size, config.hidden_size, 1)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.hidden_dropout_prob
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, label_emb, extra_attn):
        residual = label_emb
        if self.graph_type == 'graphormer':
            label_emb, attn_weights, _ = self.graph(
                hidden_states=label_emb, attention_mask=None, output_attentions=False,
                extra_attn=extra_attn,
            )
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)

            residual = label_emb
            label_emb = self.activation_fn(self.fc1(label_emb))
            label_emb = nn.functional.dropout(label_emb, p=self.activation_dropout, training=self.training)
            label_emb = self.fc2(label_emb)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.final_layer_norm(label_emb)
        elif self.graph_type == 'GCN' or self.graph_type == 'GAT':
            label_emb = self.graph(label_emb.squeeze(0), edge_index=extra_attn)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)
        else:
            raise NotImplementedError
        return label_emb


class GraphEncoder(nn.Module):
    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.hir_layers = nn.ModuleList([GraphLayer(config, graph_type) for _ in range(layer)])

        self.label_num = config.num_labels - 3
        self.graph_type = graph_type

        self.label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        if self.graph_type == 'graphormer':
            self.inverse_label_list = {}

            def get_root(path_list, n):
                ret = []
                while path_list[n] != n:
                    ret.append(n)
                    n = path_list[n]
                ret.append(n)
                return ret

            for i in range(self.label_num):
                self.inverse_label_list.update({i: get_root(path_list, i)})
            label_range = torch.arange(len(self.inverse_label_list))
            self.label_id = label_range
            node_list = {}

            def get_distance(node1, node2):
                p = 0
                q = 0
                node_list[(node1, node2)] = a = []
                node1 = self.inverse_label_list[node1]
                node2 = self.inverse_label_list[node2]
                while p < len(node1) and q < len(node2):
                    if node1[p] > node2[q]:
                        a.append(node1[p])
                        p += 1

                    elif node1[p] < node2[q]:
                        a.append(node2[q])
                        q += 1

                    else:
                        break
                return p + q

            self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1)
            hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0))
            self.distance_mat.map_(hier_mat_t, get_distance)
            self.distance_mat = self.distance_mat.view(1, -1)
            self.edge_mat = torch.zeros(self.label_num, self.label_num, 15,
                                        dtype=torch.long)
            for i in range(self.label_num):
                for j in range(self.label_num):
                    self.edge_mat[i, j, :len(node_list[(i, j)])] = torch.tensor(node_list[(i, j)])
            self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1))

            self.id_embedding = nn.Embedding(self.label_num, config.hidden_size, 0)
            self.distance_embedding = nn.Embedding(20, 1, 0)
            self.edge_embedding = nn.Embedding(self.label_num, 1, 0)
            self.label_id = nn.Parameter(self.label_id, requires_grad=False)
            self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
            self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
            self.label_name = []
            for i in range(len(self.label_dict)):
                self.label_name.append(self.label_dict[i])
            self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
            self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        else:
            self.path_list = nn.Parameter(torch.tensor(path_list).transpose(0, 1), requires_grad=False)

    def forward(self, label_emb, embeddings):
        extra_attn = None

        if self.graph_type == 'graphormer':
            label_mask = self.label_name != self.tokenizer.pad_token_id
            # full name
            label_name_emb = embeddings(self.label_name)
            label_emb = label_emb + (label_name_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)

            label_emb = label_emb + self.id_embedding(self.label_id[:, None]).view(-1,
                                                                        self.config.hidden_size)
            extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                dim=1) / (self.distance_mat.view(-1, 1) + 1e-8)
            extra_attn = extra_attn.view(self.label_num, self.label_num)
        elif self.graph_type == 'GCN' or self.graph_type == 'GAT':
            extra_attn = self.path_list

        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb.unsqueeze(0), extra_attn)

        return label_emb.squeeze(0)


class GraphConvolution(nn.Module):
    def __init__(self, in_features_dim, out_features_dim, activation=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.activation = activation
        self.weight = nn.Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # nn.init.zeros_(self.bias)

    def forward(self, infeatn, adj):
        '''
        infeatn: init feature(H，上一层的feature)
        adj: A
        '''
        hidden = torch.matmul(infeatn, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom

        # support = torch.matmul(infeatn, self.weight)  # H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        # output = torch.matmul(adj, support)  # A*H*W  # (in_feat_dim, in_feat_dim) * (in_feat_dim, out_dim)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_layers, activation="", dropout=0.1, nclass=0):

        super(GCN, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConvolution(nfeat, nhid, activation=self.activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConvolution(nhid, nhid, activation=self.activation))
        # output layer
        # self.layers.append(GraphConvolution(nhid, nclass))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        h = x
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(h, adj)
        return h


class GraphAttentionHead(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh = F.dropout(Wh, self.dropout, training=self.training)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e).cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        # print(self.a.data)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (b, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (b, N, 1)
        # e.shape (b, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer2(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nheads, alpha=0.2, concat=True):
        """Dense version of GAT."""
        super(GATLayer2, self).__init__()

        self.concat = concat
        self.attentions = [GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha, concat=concat) for _ in
                           range(nheads)]

    # for i, attention in enumerate(self.attentions):
    #     self.add_module('attention_{}'.format(i), attention)

    # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        out = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        return out


class GATLayer(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nheads, alpha=0.2, concat=True):
        """Dense version of GAT."""
        super(GATLayer, self).__init__()

        self.concat = concat
        self.attentions = [GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha, concat=concat) for _ in
                           range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        out = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        return out


class MultiLayerGAT(nn.Module):
    def __init__(self, nfeat, nhid, nlayers, nheads, dropout=0.6, residual=True):
        """Dense version of GAT."""
        super(MultiLayerGAT, self).__init__()

        assert nhid % nheads == 0
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATLayer(nfeat, nhid // nheads, dropout, nheads))
        # hidden layers
        for i in range(nlayers - 1):
            self.layers.append(GATLayer(nhid, nhid // nheads, dropout, nheads))

        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual

    # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, h, adj):
        # h = self.dropout(x)
        for i, layer in enumerate(self.layers):
            hi = layer(h, adj)
            if self.residual:
                hi = h + hi
            h = self.dropout(hi)
        return h


class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h = self.dropout(x)
        Wh = torch.matmul(h, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh = self.dropout(Wh)

        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e).cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            out = F.elu(h_prime)
        else:
            out = h_prime

        out = h + out
        out = self.dropout(out)

        return out, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (b, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (b, N, 1)
        # e.shape (b, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 1)
        return self.leakyrelu(e)


class RGraphAttentionHead2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout):
        super(RGraphAttentionHead2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h1, h2, adj):
        Wh1 = torch.matmul(h1, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh1 = F.dropout(Wh1, self.dropout, training=self.training)

        Wh2 = torch.matmul(h2, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh2 = F.dropout(Wh2, self.dropout, training=self.training)

        e = self._prepare_attentional_mechanism_input(Wh1, Wh2)

        zero_vec = -9e15 * torch.ones_like(e).cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh2)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh1, Wh2):
        # Wh1.shape (b, N1, out_feature)
        # Wh2.shape (b, N2, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (b, N1, 1), (b, N2, 1)
        # e.shape (b, N, N)
        Wh1 = torch.matmul(Wh1, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh2, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGraphAttentionHead(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout):
        super(RGraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)).cuda())
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, h1, h2, adj):
        Wh1 = torch.matmul(h1, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh1 = F.dropout(Wh1, self.dropout, training=self.training)

        Wh2 = torch.matmul(h2, self.W)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh2 = F.dropout(Wh2, self.dropout, training=self.training)

        e = self._prepare_attentional_mechanism_input(Wh1, Wh2)

        zero_vec = -9e15 * torch.ones_like(e).cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        # print("attention", attention.dtype)
        attention = F.softmax(attention, dim=-1) * adj
        attention = F.dropout(attention, self.dropout, training=self.training)
        # print("adj", adj.dtype)
        # print("attention", attention.dtype)
        # print("Wh2", Wh2.dtype)
        h_prime = torch.matmul(attention, Wh2)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh1, Wh2):
        # Wh1.shape (b, N1, out_feature)
        # Wh2.shape (b, N2, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (b, N1, 1), (b, N2, 1)
        # e.shape (b, N, N)
        Wh1 = torch.matmul(Wh1, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh2, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGraphAttentionHead_V2(nn.Module):
    """
    Relation GATV2 layer, similar to https://arxiv.org/pdf/2105.14491.pdf
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.2, share_weights=False):
        super(RGraphAttentionHead_V2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.linear_l = nn.Linear(in_features, out_features, bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, out_features, bias=False)

        self.a = nn.Linear(out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h1, h2, adj):
        Wh1 = self.linear_l(h1)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh1 = F.dropout(Wh1, self.dropout, training=self.training)

        Wh2 =self.linear_r(h2)  # h.shape: (b, N, in_features), Wh.shape: (b, N, out_features)
        Wh2 = F.dropout(Wh2, self.dropout, training=self.training)

        e = self._prepare_attentional_mechanism_input(Wh1, Wh2)

        zero_vec = -9e15 * torch.ones_like(e).cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1) * adj
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh2)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh1, Wh2):
        # Wh1.shape (b, N1, out_feature)
        # Wh2.shape (b, N2, out_feature)
        # self.a.shape (out_feature, 1)
        # e.shape (b, N1, N2, 1)
        # e.shape (b, N1, N2)

        batch_size, max_len1, dim = Wh1.size()
        _, max_len2, _ = Wh2.size()
        Wh1_repeat = Wh1.repeat(1, max_len2, 1)
        Wh2_repeat_interleave = Wh2.repeat_interleave(max_len1, dim=1)

        e = Wh1_repeat + Wh2_repeat_interleave

        e = e.view(-1, max_len1, max_len2, dim)

        return self.a(self.leakyrelu(e)).squeeze(-1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MHRGraphAttention(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nheads):
        """Dense version of GAT."""
        super(MHRGraphAttention, self).__init__()

        assert nhid % nheads == 0
        # attentions = [RGraphAttentionHead(nfeat, nhid / nheads, dropout=dropout) for _ in range(nheads)]
        #
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        # self.attentions = nn.ModuleList(*attentions)

        self.rattentions = nn.ModuleList()
        for _ in range(nheads):
            self.rattentions.append(RGraphAttentionHead(nfeat, nhid // nheads, dropout=dropout))

    def forward(self, x1, x2, adj):
        out = torch.cat([ratt(x1, x2, adj) for ratt in self.rattentions], dim=-1)
        return out


class RGATLayer(nn.Module):
    def __init__(self, rel_num, nfeat, nhid, dropout, nheads=1, gate=True, edge_norm=True, residual=True):
        """Dense version of RGAT."""
        super(RGATLayer, self).__init__()

        self.rel_num = rel_num
        self.edge_norm = edge_norm

        self.rel_gat = nn.ModuleList()
        for i in range(rel_num):
            self.rel_gat.append(MHRGraphAttention(nfeat, nhid, dropout=dropout, nheads=nheads))

        self.residual = residual

    def forward(self, x, adjs):
        h = x
        rel_out = 0
        for i in range(self.rel_num):
            if self.edge_norm:
                norm_i = torch.sum(adjs[:, i], -1, keepdim=True)
                norm_i = (norm_i == 0) + norm_i
            else:
                norm_i = 1
            temp_out = self.rel_gat[i](x, x, adjs[:, i])
            temp_out = temp_out * (torch.sum(adjs[:, i], -1, keepdim=True) != 0)

            rel_out += temp_out / norm_i


        if self.residual:
            out = F.relu(rel_out + h)
        else:
            out = F.relu(rel_out)

        return out


class RGAT(nn.Module):
    def __init__(self, rel_num, in_feat_dim, out_feat_dim, layer_num, dropout=0.1, num_heads=1, edge_norm=True, residual=True):
        """ Attetion module with vectorized version
        Args:
            label_num: numer of edge labels
            dir_num: number of edge directions
            feat_dim: dimension of roi_feat
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(RGAT, self).__init__()
        self.rel_num = rel_num
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.layer_num = layer_num

        self.dropout = nn.Dropout(dropout)
        self.rel_gat = nn.ModuleList()
        for i in range(layer_num):
            rel_gat_layer = RGATLayer(rel_num, in_feat_dim, in_feat_dim, dropout, num_heads,
                                      edge_norm=edge_norm, residual=residual)
            self.rel_gat.append(rel_gat_layer)


    def forward(self, x, adj_matrices):
        """
        Args:
            input1: [batch_size, seq_len1, dim]
            input2: [batch_size, seq_len2, dim]
            adj_matrix: [num_labels, batch_size, seq_len1, seq_len2]
        Returns:
            output1: [batch_size, seq_len1, dim]
            output2: [batch_size, seq_len2, dim]
        """

        batch_size, seq_len, dim = x.size()

        graph_embs = []
        for i in range(self.layer_num):
            x = self.rel_gat[i](x, adj_matrices)
            graph_embs.append(x) # [batch_size, seq_len, out_feat_dim]
        # graph pooling

        return graph_embs[-1]


class RGCNLayer(nn.Module):
    def __init__(self, rel_num, nfeat, nhid, dropout, nheads=1, gate=True, edge_norm=True, residual=True):
        """Dense version of RGAT."""
        super(RGCNLayer, self).__init__()

        self.rel_num = rel_num
        self.edge_norm = edge_norm

        self.rel_gcn = nn.ModuleList()
        for i in range(rel_num):
            self.rel_gcn.append(GraphConvolution(nfeat, nhid))

        self.residual = residual

    def forward(self, x, adjs):
        h = x
        rel_out = 0
        for i in range(self.rel_num):
            if self.edge_norm:
                norm_i = torch.sum(adjs[:, i], -1, keepdim=True)
                norm_i = (norm_i == 0) + norm_i
            else:
                norm_i = 1
            temp_out = self.rel_gcn[i](x, adjs[:, i])
            temp_out = temp_out * (torch.sum(adjs[:, i], -1, keepdim=True) != 0)

            rel_out += temp_out / norm_i

        if self.residual:
            out = F.relu(rel_out + h)
        else:
            out = F.relu(rel_out)

        return out


class RGCN(nn.Module):
    def __init__(self, rel_num, in_feat_dim, out_feat_dim, layer_num, dropout=0.1, num_heads=1, edge_norm=False, residual=True):
        """ Attetion module with vectorized version
        Args:
            label_num: numer of edge labels
            dir_num: number of edge directions
            feat_dim: dimension of roi_feat
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(RGCN, self).__init__()
        self.rel_num = rel_num
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.layer_num = layer_num

        self.dropout = nn.Dropout(dropout)
        self.rel_gcn = nn.ModuleList()
        for i in range(layer_num):
            rel_gcn_layer = RGCNLayer(rel_num, in_feat_dim, in_feat_dim, dropout, num_heads,
                                      edge_norm=edge_norm, residual=residual)
            self.rel_gcn.append(rel_gcn_layer)


    def forward(self, x, adj_matrices):
        """
        Args:
            input1: [batch_size, seq_len1, dim]
            input2: [batch_size, seq_len2, dim]
            adj_matrix: [batch_size, num_labels, seq_len1, seq_len2, ]
            
        Returns:
            output1: [batch_size, seq_len1, dim]
            output2: [batch_size, seq_len2, dim]
        """

        batch_size, seq_len, dim = x.size()

        graph_embs = []
        for i in range(self.layer_num):
            x = self.rel_gcn[i](x, adj_matrices)
            graph_embs.append(x) # [batch_size, seq_len, out_feat_dim]
        # graph pooling

        return graph_embs[-1]
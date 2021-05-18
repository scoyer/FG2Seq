import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from torch.nn.parameter import Parameter
from utils.utils_general import _cuda
from utils.utils_general import sequence_mask

#############################Adding for RP Model
# For summarizing a set of vectors into a single vector
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.projector = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias = False)

    def forward(self, x, x_mask=None):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        proj = torch.tanh(self.projector(x))
        scores = self.v(proj).squeeze(2)
         
        if x_mask is None:
            scores.data.masked_fill_(x_mask.data, -float('inf'))
        
        weights = F.softmax(scores, dim=1)

        weights_x = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return weights_x

class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp",
                 return_attn_only=False,
                 project=False):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp"]), (
            "Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode
        self.return_attn_only = return_attn_only
        self.project = project

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(
                self.memory_size, self.hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def __repr__(self):
        main_string = "Attention({}, {}".format(self.query_size, self.memory_size)
        if self.mode == "mlp":
            main_string += ", {}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, memory, value=None, mask=None, return_weights=False):
        """
        query: Tensor(batch_size, query_length, query_size)
        memory: Tensor(batch_size, memory_length, memory_size)
        mask: Tensor(batch_size, memory_length)
        """
        if self.mode == "dot":
            assert query.size(-1) == memory.size(-1)
            # (batch_size, query_length, memory_length)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            # (batch_size, query_length, memory_size)
            key = self.linear_query(query)
            # (batch_size, query_length, memory_length)
            attn = torch.bmm(key, memory.transpose(1, 2))
        else:
            # (batch_size, query_length, memory_length, hidden_size)
            hidden = self.linear_query(query).unsqueeze(
                2) + self.linear_memory(memory).unsqueeze(1)
            key = self.tanh(hidden)
            # (batch_size, query_length, memory_length)
            attn = self.v(key).squeeze(-1)
            
        if value is None:
            value = memory

        if mask is not None:
            # (batch_size, query_length, memory_length)
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn.masked_fill_(mask, -float("inf"))

        if self.return_attn_only:
            return attn

        # (batch_size, query_length, memory_length)
        weights = self.softmax(attn)

        # (batch_size, query_length, memory_size)
        weighted_memory = torch.bmm(weights, value)

        if self.project:
            project_output = self.linear_project(
                torch.cat([weighted_memory, query], dim=-1))
            return project_output, weights
        else:
            if return_weights:
                return weighted_memory, attn
            else:
                return weighted_memory

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class RNNEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        rnn_hidden_size = hidden_size // num_directions

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)

    def forward(self, inputs, hidden=None):
        """
        forward
        """
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.hidden_size)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    def _bridge_bidirectional_hidden(self, hidden):
        """
        the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * hidden_size)
        """
        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)


class HRNNEncoder(nn.Module):
    """
    HRNNEncoder
    """
    def __init__(self,
                 sub_encoder,
                 hiera_encoder,
                 is_word_attn=True):
        super(HRNNEncoder, self).__init__()
        self.sub_encoder = sub_encoder
        self.hiera_encoder = hiera_encoder
        self.is_word_attn = is_word_attn
        
        #add self-attention
        if self.is_word_attn:
            self.word_attention = SelfAttention(self.sub_encoder.hidden_size, self.sub_encoder.hidden_size // 2)

    def forward(self, inputs, mask, features=None, sub_hidden=None, hiera_hidden=None):
        """
        inputs: Tuple[Tensor(batch_size, max_hiera_len, max_sub_len, hidden_size), 
                Tensor(batch_size, max_hiera_len)]
        """
        indices, lengths = inputs
        batch_size, max_hiera_len, max_sub_len, hidden_size = indices.size()
        hiera_lengths = lengths.gt(0).long().sum(dim=1)

        # Forward of sub encoder
        indices = indices.view(-1, max_sub_len, hidden_size)
        sub_lengths = lengths.view(-1)
        sub_enc_inputs = (indices, sub_lengths)
        sub_outputs, sub_hidden = self.sub_encoder(sub_enc_inputs, sub_hidden)

        #word-attention
        if self.is_word_attn:
            mask = mask.view(-1, max_sub_len)
            sub_hidden = self.word_attention(sub_outputs, mask).view(batch_size, max_hiera_len, -1)

        if features is not None:
            sub_hidden = torch.cat([sub_hidden, features], dim=-1)

        # Forward of hiera encoder
        hiera_enc_inputs = (sub_hidden, hiera_lengths)
        hiera_outputs, hiera_hidden = self.hiera_encoder(hiera_enc_inputs, hiera_hidden)

        sub_outputs = sub_outputs.view(
            batch_size, max_hiera_len, max_sub_len, -1)

        last_sub_outputs = torch.stack(
            [sub_outputs[b, l - 1] for b, l in enumerate(hiera_lengths)])
        last_sub_lengths = torch.stack(
            [lengths[b, l - 1] for b, l in enumerate(hiera_lengths)])

        max_len = last_sub_lengths.max()
        last_sub_outputs = last_sub_outputs[:, :max_len]
        return hiera_outputs, hiera_hidden, sub_outputs, sub_hidden, last_sub_outputs, last_sub_lengths
            #return hiera_outputs, hiera_hidden, (last_sub_outputs, last_sub_lengths)

class GCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, relation_size, dropout, B=0):
        super(GCNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.relation_size = relation_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.B = B
        
        if self.B == 0:
            self.W = nn.Parameter(torch.empty(self.relation_size, self.input_size, self.hidden_size, dtype=torch.float))
            nn.init.xavier_uniform_(self.W)
        else:
            self.W1 = nn.Parameter(torch.empty(self.B, self.input_size, self.hidden_size, dtype=torch.float))
            self.W2 = nn.Parameter(torch.empty(self.relation_size, self.B, dtype=torch.float))
            nn.init.xavier_uniform_(self.W1)
            nn.init.xavier_uniform_(self.W2)

        self.W0 = nn.Linear(self.input_size, self.hidden_size, bias=False)

    def forward(self, x, graph):
        # x     : [batch * m * h1]
        # graph : [batch * r * m * m]

        x_transform = torch.matmul(graph, x.unsqueeze(1))   

        if self.B == 0:
            W = self.W
        else:
            W = torch.matmul(self.W2, self.W1.permute(1, 0, 2)).permute(1, 0, 2)
        
        x_transform = x_transform.contiguous().view(x_transform.size(0), graph.size(1), graph.size(2), -1)
        x_transform = x_transform.transpose(0, 1).contiguous().view(x_transform.size(1), -1, x_transform.size(-1))
        x_transform = torch.sum(torch.bmm(x_transform, W), dim=0).view(x.size(0), x.size(1), -1)
        
        x_self = self.W0(x)
        
        out = self.relu(x_transform + x_self)

        return out

    def forward_sparse(self, x, graph):
        pass


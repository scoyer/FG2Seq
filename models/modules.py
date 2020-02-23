import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from torch.nn.parameter import Parameter
from utils.utils_general import _cuda
from utils.utils_general import sequence_mask
from models.layers import SelfAttention, Attention, RNNEncoder, HRNNEncoder, GCNEncoder


class DualAttentionDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, dropout):
        super(DualAttentionDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) 
        self.C = shared_emb 
        self.softmax = nn.Softmax(dim=1)
        self.gru = nn.GRU(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(4*embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim = 1)
        self.knowledge_attention = Attention(embedding_dim, embedding_dim*2, embedding_dim, mode='mlp')
        self.context_attention = Attention(embedding_dim, embedding_dim*2, embedding_dim, mode='mlp')
        self.concat = nn.Linear(5*embedding_dim, embedding_dim)
        self.entity_ranking = Attention(embedding_dim, embedding_dim*2, embedding_dim, mode='mlp', return_attn_only=True)

    def forward(self, extKnow, extKnow_mask, context, context_mask, story_size, story_lengths, copy_list, encode_hidden, target_batches, max_target_length, schedule_sampling, get_decoded_words):
        batch_size = len(copy_list)

        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []
        
        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        
        # Start to generate word-by-word
        for t in range(max_target_length):
            rnn_input_list, concat_input_list = [], []

            embed_q = self.dropout_layer(self.C(decoder_input)) # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            rnn_input_list.append(embed_q)

            rnn_input = torch.cat(rnn_input_list, dim=1)
            _, hidden = self.gru(rnn_input.unsqueeze(0), hidden)
            concat_input_list.append(hidden.squeeze(0))

            #get knowledge attention
            knowledge_outputs, _ = self.knowledge_attention(hidden.transpose(0,1), extKnow, mask=extKnow_mask, return_weights=True)
            concat_input_list.append(knowledge_outputs.squeeze(1))

            #get context attention
            context_outputs = self.context_attention(hidden.transpose(0,1), context, mask=context_mask)
            concat_input_list.append(context_outputs.squeeze(1))

            #concat_input = torch.cat((hidden.squeeze(0), context_outputs.squeeze(1), knowledge_outputs.squeeze(1)), dim=1)
            concat_input = torch.cat(concat_input_list, dim=1)
            concat_output = torch.tanh(self.concat(concat_input))

            p_vocab = self.attend_vocab(self.C.weight, concat_output)
            p_entity = self.entity_ranking(concat_output.unsqueeze(1), extKnow, mask=extKnow_mask).squeeze(1)

            prob_soft = self.softmax(p_entity)

            all_decoder_outputs_vocab[t] = p_vocab
            all_decoder_outputs_ptr[t] = p_entity

            use_teacher_forcing = random.random() < schedule_sampling
            if use_teacher_forcing:
                decoder_input = target_batches[:,t] 
            else:
                _, topvi = p_vocab.data.topk(1)
                decoder_input = topvi.squeeze()
            
            if get_decoded_words:

                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []
                
                for bi in range(batch_size):
                    token = topvi[bi].item() #topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])
                    
                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:,i][bi] < story_lengths[bi]-1: 
                                cw = copy_list[bi][toppi[:,i][bi].item()]            
                                break
                        temp_f.append(cw)
                        
                        if args['record']:
                            memory_mask_for_step[bi, toppi[:,i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        # scores = F.softmax(scores_, dim=1)
        return scores_

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class ContextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout):
        super(ContextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)

        #define two RNNEncoders and one HRNNEncoder
        self.question_rnn1 = RNNEncoder(
                input_size=embedding_dim,
                hidden_size=embedding_dim * 2,
                embedder=None,
                num_layers=1,
                bidirectional=True,
                dropout=dropout)
        self.question_rnn2 = RNNEncoder(
                input_size=embedding_dim * 2,
                hidden_size=embedding_dim * 2,
                embedder=None,
                num_layers=1,
                bidirectional=False,
                dropout=dropout)
        self.hier_question_rnn = HRNNEncoder(self.question_rnn1, self.question_rnn2)

    def forward(self, x2, x2_lengths, x2_mask):
        x2_embed = self.embedding(x2.contiguous())

        #add dropout
        x2_embed = self.dropout_layer(x2_embed)

        hiera_outputs, hiera_hidden, sub_outputs, sub_hidden, last_sub_outputs, last_sub_lengths = self.hier_question_rnn((x2_embed, x2_lengths), x2_mask)

        # Get the question mask
        question_len = x2_lengths.gt(0).long().sum(dim=1)
        question_mask = torch.stack(
                [x2_mask[b, l - 1] for b, l in enumerate(question_len)])
        max_len = last_sub_lengths.max()
        question_mask = question_mask[:, :max_len]

        return x2_embed, sub_outputs, sub_hidden, hiera_outputs, hiera_hidden, last_sub_outputs, last_sub_lengths, question_mask


class KnowledgeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, relation_size, dropout, B):
        super(KnowledgeEncoder, self).__init__()
        #Embedding parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.relation_size = relation_size
        self.relu = nn.ReLU()
        self.B = B
        
        #get C_i_1
        self.question_attn1 = Attention(embedding_dim, embedding_dim, embedding_dim, mode='mlp')
        self.dialog_flow1 = RNNEncoder(embedding_dim * 2 + 1, embedding_dim, embedder=None, num_layers=1, bidirectional=False)
        self.gcn1 = GCNEncoder(embedding_dim, embedding_dim, self.relation_size, dropout, B=self.B)

        #get C_i_2
        self.question_attn2 = Attention(embedding_dim * 2, embedding_dim * 2, embedding_dim, mode='mlp')
        self.dialog_flow2 = RNNEncoder(embedding_dim * 4, embedding_dim, embedder=None, num_layers=1, bidirectional=False)
        self.gcn2 = GCNEncoder(embedding_dim, embedding_dim, self.relation_size, dropout, B=self.B)

        #self-attention
        self.entity_attention = SelfAttention(embedding_dim * 2, embedding_dim)

    def graph_norm(self, graph):
        graph = graph.to_dense()
        batch_size = graph.size(0)
        
        degree = torch.sum(graph, dim=-1, keepdim=True).clamp(min=1)
        graph = graph / degree  
        return graph
        
    def forward(self, x1, x1_f, x1_mask, x1_lengths, x2, x2_mask, x2_lengths, x2_embed, x2_outputs, x2_hidden, graph):
        """
        x1 :                [batch * len_k * MEM_TOKEN_SIZE]
        x1_f :              [batch * q_num * len_k * n_feat(1)]
        x1_mask :           [batch * len_k]
        x1_lengths :        [batch]
        x2 :                [batch * q_num * len_c * MEM_TOKEN_SIZE]
        x2_mask :           [batch * q_num * len_c]
        x2_embed :          [batch * q_num * len_c * h1]
        x2_outputs :        [batch * q_num * len_c * h]
        x2_lengths :        [batch * q_num]
        """

        #print("x1 size:", x1.size())
        #print("x1_f size:", x1_f.size())
        #print("x1_mask size:", x1_mask.size())
        #print("x2 size:", x2.size())
        #print("x2_mask size:", x2_mask.size())

        batch_size, len_k = x1.size(0), x1.size(1)
        q_num, len_c = x2.size(1), x2.size(2)
        
        def expansion_for_doc(z):
            return z.unsqueeze(1).expand(z.size(0), q_num, z.size(1), z.size(2)).contiguous().view(-1, z.size(1), z.size(2))
        
        #embedding
        x1_embed = self.embedding(x1)
        #add dropout
        x1_embed = self.dropout_layer(x1_embed)
        x1_embed_expand = expansion_for_doc(x1_embed) #(b * q_num) * len_k * em1
        x1_mask_expand = x1_mask.unsqueeze(1).expand(x1.size(0), x2.size(1), x1.size(1)).contiguous().view(-1, x1_mask.size(-1)) #(b * q_num) * len_k
        graph = self.graph_norm(graph)
        graph_expand = graph.unsqueeze(1).expand(graph.size(0), q_num, graph.size(1), graph.size(2), graph.size(3))
        graph_expand = graph_expand.contiguous().view(-1, graph.size(1), graph.size(2), graph.size(3))

        x2_embed = x2_embed.contiguous().view(-1, x2_embed.size(-2), x2_embed.size(-1))
        x2_mask = x2_mask.view(-1, x2_mask.size(-1))

        #question Encoding
        questions_hiddens = x2_outputs.view(batch_size * q_num, len_c, -1)

        def flow_operation(cur_h, flow):
            flow_in = cur_h.transpose(0, 1).view(len_k, batch_size, q_num, -1)
            flow_in = flow_in.transpose(0, 2).contiguous().view(q_num, batch_size * len_k, -1).transpose(0, 1)
            # [bsz * context_length, max_qa_pair, hidden_state]
            flow_out,_ = flow(flow_in)
            # [bsz * context_length, max_qa_pair, flow_hidden_state_dim (hidden_state/2)]

            flow_out = flow_out.transpose(0, 1).view(q_num, batch_size, len_k, -1).transpose(0, 2).contiguous()
            flow_out = flow_out.view(len_k, batch_size * q_num, -1).transpose(0, 1)
            # [bsz * max_qa_pair, context_length, flow_hidden_state_dim]
            return flow_out

        #get C_i_1
        x1_attn = self.question_attn1(x1_embed_expand, x2_embed, mask=x2_mask) #(b * q_num) * len_k * em2
        C1_input = torch.cat([x1_embed_expand, x1_attn, x1_f.view(batch_size*q_num, len_k, 1)], dim=2) #(b * q_num) * len_k * (em1 + em2 + n_feat)
        C1_1 = flow_operation(C1_input, self.dialog_flow1) #(b * q_num) * len_k * em
        C1_2 = self.gcn1(C1_1, graph_expand)


        #get C_i_2
        x1_attn2 = self.question_attn2(torch.cat((C1_1, C1_2), dim=2), questions_hiddens, mask=x2_mask)
        C2_input = torch.cat((C1_1, C1_2, x1_attn2), dim=2)
        C2_1 = flow_operation(C2_input, self.dialog_flow2)
        C2_2 = self.gcn2(C2_1, graph_expand)
        
        C_final = torch.cat((C2_1, C2_2), dim=2)
        C_final = C_final.contiguous().view(batch_size, q_num, len_k, -1)
        
        #get the last question representation
        qid = x2_lengths.gt(0).long().sum(dim=1)
        outputs = torch.stack(
                [C_final[b, l - 1] for b, l in enumerate(qid)]) #batch_size * len_k * h

        hidden = self.entity_attention(outputs, x_mask = x1_mask).unsqueeze(1)
        return outputs, hidden

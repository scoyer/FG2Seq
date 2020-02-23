import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *

from itertools import chain

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask

def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, story, trg=False):
        if trg:
            for word in story:
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, relation_size):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.relation_size = relation_size
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        context_arr = self.data_info['context_arr'][index]
        context_arr = [self.preprocess(seq, self.src_word2id, trg=False) for seq in context_arr]
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        indicator = self.data_info['indicator'][index]
        indicator = [torch.Tensor(seq) for seq in indicator]
        graph = self.data_info['graph'][index]
        graph = torch.Tensor(graph)
        
        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['response_plain'] = " ".join(self.data_info['response'][index])
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence]+ [EOS_token]
        else:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence]
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences):
            lengths = torch.tensor([len(seq) for seq in sequences]).long()
            max_len = 1 if max(lengths)==0 else max(lengths)
            mask = torch.ones(len(sequences), max_len).byte()
            padded_seqs = torch.ones(len(sequences), max_len).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
                mask[i,:end] = torch.zeros(end)
            return padded_seqs, lengths, mask

        def merge_index(sequences):
            lengths = torch.tensor([len(seq) for seq in sequences]).long()
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            mask = torch.ones(len(sequences), max(lengths)).byte()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
                mask[i, :end] = torch.zeros(end)
            return padded_seqs, lengths, mask

        def merge_conversation(sequences):
            dim1 = max([len(seq) for seq in sequences])
            dim2 = max([len(s) for seq in sequences for s in seq])
            padded_seqs = torch.ones(len(sequences), dim1, dim2).long()
            lengths = torch.zeros(len(sequences), dim1).long()
            mask = torch.ones(len(sequences), dim1, dim2).byte()
            for i, seq in enumerate(sequences):
                for j, s in enumerate(seq):
                    end = len(s)
                    lengths[i, j] = end
                    padded_seqs[i, j, :end] = s[:end]
                    mask[i, j, :end] = torch.zeros(end)

                for j in range(len(seq), dim1):
                    padded_seqs[i, j, :2] = torch.LongTensor([SOS_token, EOS_token])
                    mask[i, j, :2] = torch.zeros(2)

            return padded_seqs, lengths, mask

        def merge_indicator(sequences):
            dim1 = max([len(seq) for seq in sequences])
            dim2 = max([len(s) for seq in sequences for s in seq])
            padded_seqs = torch.zeros(len(sequences), dim1, dim2).float()
            for i, seq in enumerate(sequences):
                for j, s in enumerate(seq):
                    end = len(s)
                    padded_seqs[i, j, :end] = s[:end]
            return padded_seqs

        def merge_graph(sequences, edge_num, node_num):
            #dim1 = max([len(seq) for seq in sequences])
            #dim2 = max([len(s) for seq in sequences for s in seq])
            #padded_seqs = torch.zeros(len(sequences), dim1, dim2, dim2).float()
            #for i, seq in enumerate(sequences):
            #    for j, s in enumerate(seq):
            #        end = len(s)
            #        padded_seqs[i, j, :end] = s[:end]
            #padded_seqs.to_sparse()

            all_indices = torch.cat(sequences, dim=0)
            i = torch.zeros(all_indices.size(0), 4).long()
            i[:, 1:] = all_indices
            
            idx = 0
            for seq_id, seq in enumerate(sequences):
                i[idx:idx+seq.size(0), 0] = torch.LongTensor([seq_id] * seq.size(0))
                idx = idx + seq.size(0)

            v = torch.ones(i.size(0)).float()
            padded_seqs = torch.sparse.FloatTensor(i.t(), v, torch.Size([len(sequences), edge_num, node_num, node_num]))

            return padded_seqs

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        response, response_lengths, response_mask = merge(item_info['response'])
        sketch_response, _, _ = merge(item_info['sketch_response'])
        ptr_index, _, _ = merge(item_info['ptr_index'])
        context_arr, context_arr_lengths, context_arr_mask = merge_conversation(item_info['context_arr'])
        kb_arr, kb_arr_lengths, kb_arr_mask = merge(item_info['kb_arr'])
        indicator = merge_indicator(item_info['indicator'])
        graph = merge_graph(item_info['graph'], self.relation_size, kb_arr.size(1))
        

        # convert to contiguous and cuda
        response = _cuda(response.contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        context_arr = _cuda(context_arr.contiguous())
        kb_arr = _cuda(kb_arr.contiguous())
        indicator = _cuda(indicator.contiguous())
        graph = _cuda(graph)

        #mask to contiguous and cuda
        response_mask = _cuda(response_mask.contiguous())
        context_arr_mask = _cuda(context_arr_mask.contiguous())
        kb_arr_mask = _cuda(kb_arr_mask.contiguous())
        
        #length to contiguous and cuda
        response_lengths = _cuda(response_lengths.contiguous())
        context_arr_lengths = _cuda(context_arr_lengths.contiguous())
        kb_arr_lengths = _cuda(kb_arr_lengths.contiguous())

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        data_info['response_mask'] = response_mask
        data_info['context_arr_mask'] = context_arr_mask
        data_info['kb_arr_mask'] = kb_arr_mask


        data_info['response_lengths'] = response_lengths
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths
        return data_info

def get_seq(pairs, lang, batch_size, type, relation_size):   
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []
    
    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if(type):
            #for seq in pair['context_arr']:
            #    lang.index_words(seq, trg=True)
            lang.index_words(sum(pair['context_arr'], []), trg=True)
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['kb_arr'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)
    
    dataset = Dataset(data_info, lang.word2index, lang.word2index, relation_size)
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size = batch_size,
                                              shuffle = type,
                                              collate_fn = dataset.collate_fn)
    return data_loader

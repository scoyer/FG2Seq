import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
import datetime
from utils.measures import wer,moses_multi_bleu
from sklearn.metrics import f1_score
import math

class LuongSeqToSeq(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, lr, dropout, relation_size, B):
        super(LuongSeqToSeq, self).__init__()
        self.name = "LuongSeqToSeq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.relation_size = relation_size
        self.B = B

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = EncoderRNN(lang.n_words, hidden_size, dropout=dropout)
            self.decoder = LuongAttnDecoderRNN(hidden_size, lang.n_words, dropout=dropout)

        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()
            
    def save_model(self, dec_type):
        name_data = "KVR/" if self.task=='kvr' else "CAM/"
        directory = 'save/GF2Seq-'+args["addName"]+name_data+(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'lr'+str(self.lr)+'RS'+str(self.B)+str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')
        
    def reset(self):
        self.loss, self.print_every, = 0, 1
    
    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0, ss=1.0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        # Encode and Decode
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, _ = self.encode_and_decode(data, max_target_length, ss, False)
        
        # Loss calculation and backpropagation
        loss = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), 
            data['sketch_response'].contiguous(), 
            data['response_lengths'])
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        self.loss += loss.item()
    
    def encode_and_decode(self, data, max_target_length, schedule_sampling, get_decoded_words):
        # Encode dialog history to vectors
        enc_outputs, enc_hidden = self.encoder(data['context_arr'], data['context_arr_lengths'])
        
        outputs_vocab, decoded_fine = self.decoder(max_target_length, data['response'], enc_hidden, enc_outputs, get_decoded_words=get_decoded_words)

        return outputs_vocab, decoded_fine

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.extKnow.train(False)
        self.decoder.train(False)
        
        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev),total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if self.task == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar: 
            # Encode and Decode
            _, _, decoded_fine, decoded_coarse = self.encode_and_decode(data_dev, self.max_resp_len, -1.0, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS': break
                    else: st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS': break
                    else: st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                
                if self.task == 'kvr': 
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                elif self.task == 'cam':
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), [], data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.extKnow.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t"+str(acc_score))

        if self.task == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("\tCAL F1:\t{}".format(F1_cal_pred/float(F1_cal_count))) 
            print("\tWET F1:\t{}".format(F1_wet_pred/float(F1_wet_count))) 
            print("\tNAV F1:\t{}".format(F1_nav_pred/float(F1_nav_count))) 
            print("BLEU SCORE:\t"+str(bleu_score))
        elif self.task == 'cam':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            print("BLEU SCORE:\t"+str(bleu_score))
        
        if (early_stop == 'BLEU'):
            if (bleu_score >= matric_best):
                self.save_model('BLEU-'+str(bleu_score))
                print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        context_plain = data['kb_arr_plain'][batch_idx]
        conv_plain = sum(data['context_arr_plain'][batch_idx], [])

        kb_len = len(context_plain) - len(conv_plain) - 1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len): 
            kb_temp = [w for w in context_plain[i] if w!='PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(conv_plain[kb_len:]):
            if word_arr[1]==flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr,': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()      
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout       
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout) 
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout)
        if USE_CUDA:
            self.lstm = self.lstm.cuda() 
            self.embedding_dropout = self.embedding_dropout.cuda()
            self.embedding = self.embedding.cuda() 

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(1)
        c0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))  
        h0_encoder = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)) ### * self.num_directions = 2 if bi
        if USE_CUDA:
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda() 
        return (h0_encoder, c0_encoder)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.embedding_dropout(embedded)
        hidden = self.get_state(input_seqs)
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)    
        outputs, hidden = self.lstm(embedded, hidden)
        if input_lengths:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        return outputs, hidden


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.W1 = nn.Linear(2*hidden_size, hidden_size)
        # self.v = nn.Linear(hidden_size, 1)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)   
        if USE_CUDA:
            self.embedding = self.embedding.cuda()
            self.dropout = self.dropout.cuda()
            self.lstm = self.lstm.cuda()
            self.out = self.out.cuda() 
            self.W1 = self.W1.cuda() 
            self.v = self.v.cuda() 
            self.concat = self.concat.cuda()

    def decode(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        max_len = encoder_outputs.size(0)
        encoder_outputs = encoder_outputs.transpose(0,1)
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.lstm(embedded, last_hidden)

        s_t = hidden[0][-1].unsqueeze(0)
        H = s_t.repeat(max_len,1,1).transpose(0,1)

        energy = F.tanh(self.W1(torch.cat([H,encoder_outputs], 2)))
        energy = energy.transpose(2,1)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        a = F.softmax(energy)
        context = a.bmm(encoder_outputs)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden
    
    def forward(self, batch_size, max_target_length, target_batches, encoder_hidden, encoder_outputs, teacher_forcing_ratio=0.5, get_decoded_words=False):
        batch_size = len(target_batches)
    
        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
        decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],encoder_hidden[1][:self.decoder.n_layers])

        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            decoder_input = decoder_input.cuda()
        
        if get_decoded_words:
            use_teacher_forcing = False
        else:
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
        decoded_words = []
        
        if use_teacher_forcing:    
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_vacab, decoder_hidden = self.decoder.decode(decoder_input, decoder_hidden, encoder_outputs)

                all_decoder_outputs_vocab[t] = decoder_vacab
                decoder_input = target_batches[t] # Next input is current target
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                decoder_vacab,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs_vocab[t] = decoder_vacab
                topv, topi = decoder_vacab.data.topk(1)
                decoder_input = Variable(topi.view(-1)) # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
                
                temp = []
                if get_decoded_words:
                    for bi in range(batch_size):
                        token = topi[bi].item() #topvi[:,0][bi].item()
                        temp.append(self.lang.index2word[token])
        
        return all_decoder_outputs_vocab, decoded_words
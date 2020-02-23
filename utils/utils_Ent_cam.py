import json
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *
import ast

from utils.utils_general import *

relation_set = [
        "address",
        "area",
        "food",
        "id",
        "location",
        "name",
        "phone",
        "postcode",
        "pricerange",
        "type"
]


def read_langs(file_name, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, kb_arr, kb_id = [], [], [], []
    max_resp_len = 0
    
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    context_arr.append(u.split(' '))
                
                    # Get gold entity
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_index = list(set(gold_ent))

                    # Get entity set
                    entity_set, entity_type_set = generate_entity_set(kb_arr)

                    # Get local pointer position for each word in system response
                    ptr_index = []
                    for key in r.split():
                        if key in entity_set:
                            index = entity_set.index(key)
                        else: 
                            index = len(entity_set)
                        ptr_index.append(index)
 
                    sketch_response = generate_template(r, ent_index, entity_set, entity_type_set)

                    #add empty token
                    if len(entity_set) == 0:
                        entity_set.append("$$$$")
                        entity_type_set.append("empty_token")
                     
                    entity_set.append("$$$$")
                    entity_type_set.append("empty_token")
                    
                    #generate indicator
                    indicator = generate_indicator(context_arr, entity_set)
                    
                    #generate graph
                    graph = generate_graph(entity_set, relation_set, kb_arr)
                    
                    data_detail = {
                        'context_arr':list(context_arr),
                        'kb_arr':list(entity_set),
                        'response':r.split(' '),
                        'sketch_response':sketch_response.split(' '),
                        'ptr_index':ptr_index+[len(entity_set) - 1],
                        'indicator':indicator,
                        'ent_index':ent_index,
                        'id':int(sample_counter),
                        'ID':int(cnt_lin),
                        'graph':graph}
                    data.append(data_detail)
                    
                    context_arr.append(r.split(' '))

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    kb_id.append(nid)
                    kb_info = line.split(' ')
                    kb_arr.append(kb_info)
                    if len(kb_info) != 5:
                       print(kb_info)
            else:
                cnt_lin += 1
                context_arr, kb_arr, kb_id = [], [], []
                if(max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len



def generate_indicator(context_arr, entity_set):
    """
    generate a list with the same size of context_arr, indicating whether each element of context_arr appears in kb_arr
    """

    indicator = []
    for s_id, question in enumerate(context_arr):
        indicator.append([1 if entity in question else 0 for entity in entity_set])

    return indicator


def generate_entity_from_context(context_arr, global_entity, entity_set, entity_type_set):
    for sent in context_arr:
        for entity in sent:
            if entity in entity_set:
                continue
            for k, v in global_entity.items():
                if entity in v:
                    entity_set.append(entity)
                    entity_type_set.append(k)
                    break
    return entity_set, entity_type_set


def generate_graph(entity_set, relation_set, kb_arr):
    node_num = len(entity_set)
    edge_num = len(relation_set)

    #check validity
    for kb in kb_arr:
        assert kb[1] in relation_set, kb[1]

    #graph = [[[0 for _ in range(node_num)] for _ in range(node_num)] for _ in range(edge_num)]

    #for kb in kb_arr:
    #    entity_id1 = entity_set.index(kb[0])
    #    relation_id= relation_set.index(kb[1])
    #    entity_id2 = entity_set.index(kb[2])
    #    graph[relation_id][entity_id1][entity_id2] = 1
    #    graph[relation_id][entity_id2][entity_id1] = 1

    graph = []
    for kb in kb_arr:
        entity_id1 = entity_set.index(kb[0])
        relation_id= relation_set.index(kb[1])
        entity_id2 = entity_set.index(kb[2])
        graph.append([relation_id, entity_id1, entity_id2])
        graph.append([relation_id, entity_id2, entity_id1])
   
    if len(graph) == 0:
        graph.append([0,0,0])
    #for i in range(len(entity_set)):
    #    graph.append([0, i, i])

    return graph


def generate_template(sentence, sent_ent, entity_set, entity_type_set):
    """
    Based on the system response and the provided entity table, the output is the sketch response. 
    """
    sketch_response = [] 
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else: 
                ent_type = None
                for entity_id, entity in enumerate(entity_set):
                    if word == entity:
                        ent_type = entity_type_set[entity_id]
                        break

                    #for key in global_entity.keys():
                    #    if key!='poi':
                    #        global_entity[key] = [x.lower() for x in global_entity[key]]
                    #        if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                    #            ent_type = key
                    #            break
                    #    else:
                    #        poi_list = [d['poi'].lower() for d in global_entity['poi']]
                    #        if word in poi_list or word.replace('_', ' ') in poi_list:
                    #            ent_type = key
                    #            break
                sketch_response.append('@'+ent_type)        
    sketch_response = " ".join(sketch_response)
    return sketch_response

def  generate_entity_set(kb_arr):
    entity_set, entity_type_set = [], []
    for kb in kb_arr:
        if kb[0] not in entity_set:
            entity_set.append(kb[0])
            entity_type_set.append(kb[3])
        if kb[2] not in entity_set:
            entity_set.append(kb[2])
            entity_type_set.append(kb[4])

    return entity_set, entity_type_set


def prepare_data_seq(batch_size=100):
    file_train = 'data/CamRest/train.txt'
    file_dev = 'data/CamRest/dev.txt'
    file_test = 'data/CamRest/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    
    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True, len(relation_set))
    dev   = get_seq(pair_dev, lang, batch_size, False, len(relation_set))
    test  = get_seq(pair_test, lang, batch_size, False, len(relation_set))
    
    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, lang, max_resp_len, len(relation_set)

def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d

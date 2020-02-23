from utils.config import *
from models.FG2Seq import *

directory = args['path'].split("/")
DS = directory[1].split('DS')[1].split('HDD')[0]
HDD = directory[1].split('HDD')[1].split('BSZ')[0]
BSZ =  int(directory[1].split('BSZ')[1].split('DR')[0])
B = directory[1].split('RS')[1].split('BLEU')[0]

if DS=='kvr': 
    from utils.utils_Ent_kvr import *
elif DS=='cam':
    from utils.utils_Ent_cam import *
else: 
    print("You need to provide the --dataset information")

train, dev, test, lang, max_resp_len, relation_size = prepare_data_seq(batch_size=BSZ)

model = FG2Seq(
	int(HDD), 
	lang, 
	max_resp_len, 
	args['path'], 
	DS, 
	lr=0.0, 
	dropout=0.0,
    relation_size=relation_size,
    B=int(B))

acc_test = model.evaluate(test, 1e7) 

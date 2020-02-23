import math
from tqdm import tqdm

from utils.config import *
from models.FG2Seq import *


early_stop = args['earlyStop']
if args['dataset']=='kvr':
    from utils.utils_Ent_kvr import *
    early_stop = 'BLEU'
elif args['dataset']=='cam':
    from utils.utils_Ent_cam import *
    early_stop = 'BLEU'
else:
    print("[ERROR] You need to provide the --dataset information")

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, lang, max_resp_len, relation_size = prepare_data_seq(batch_size=int(args['batch']))

model = FG2Seq(
    int(args['hidden']), 
    lang, 
    max_resp_len, 
    args['path'], 
    args['dataset'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    relation_size=relation_size,
    B=int(args['relation_size_reduce']))

for epoch in range(200):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    schedule_sampling = args['schedule_sampling_ratio'] / (args['schedule_sampling_ratio'] + math.exp(epoch / args['schedule_sampling_ratio']) - 1)
    print("schedule_sampling_ratio: ", schedule_sampling)

    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), reset=(i==0), ss=schedule_sampling)
        pbar.set_description(model.print_loss())
        # break
    if((epoch+1) % int(args['evalp']) == 0):    
        acc = model.evaluate(test, avg_best, early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if(cnt == 10 or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 



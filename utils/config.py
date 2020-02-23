import os
import argparse
from tqdm import tqdm

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0 

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
parser.add_argument('-ds','--dataset', help='dataset, babi or kvr', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False)
parser.add_argument('-dr','--drop', help='Drop Out', required=False)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10)
parser.add_argument('-ss','--schedule_sampling_ratio', help='schedule_sampling_ratio', type=float, required=False, default=10.0)

parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-gs','--genSample', help='Generate Sample', required=False, default=0)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-rec','--record', help='use record function during inference', type=int, required=False, default=0)

parser.add_argument('-B', '--relation_size_reduce', help='reduce the size of relation set', type=int, required=False,default=0)

args = vars(parser.parse_args())
print(str(args))
print("USE_CUDA: "+str(USE_CUDA))

LIMIT = int(args["limit"])

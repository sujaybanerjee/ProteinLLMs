import argparse
import os
import random

import pickle  # Importing pickle for loading precomputed embeddings

import numpy as np
import torch
from torch import nn

import esm


def exists(fileOrFolderPath):
    return os.path.exists(fileOrFolderPath)


def set_seed(seed=-1):
    if seed == -1:
        seed = random.randint(1, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def createFolder(folder):
    if not exists(folder):
        os.makedirs(folder)


def print_namespace(anamespace, ignore_none=True):
    for key in anamespace.__dict__:
        if ignore_none and anamespace.__dict__[key] is None:
            continue
        print("{}: {}".format(key, anamespace.__dict__[key]))


def parsePredProbs(outs):
    # Same as your original function
    __type = 1
    if outs.size(-1) == 2:
        __type = 2
        outs = outs.view(-1, 2)
    else:
        outs = outs.view(-1, 1)

    sam_num = outs.size(0)
    outs = outs.tolist()
    pred_probs = []
    for j in range(sam_num):
        out = outs[j]
        if 2 == __type:
            prob_posi = out[1]
            prob_nega = out[0]
        else:
            prob_posi = out[0]
            prob_nega = 1.0 - prob_posi

        sum_probs = prob_posi + prob_nega

        if sum_probs < 1e-99:
            pred_probs.append(0.)
        else:
            pred_probs.append(prob_posi / sum_probs)

    return pred_probs


class CoreModel(nn.Module):
    def __init__(self, in_dim=20, out_dim=2, body_num=10, dr=0.1):
        super(CoreModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(128)
        )

        self.body_num = body_num
        self.body1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.GELU(),
                nn.BatchNorm1d(128),
            ) for _ in range(body_num)
        ])
        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(128) for _ in range(body_num)
        ])
        self.dropout = nn.Dropout(p=dr)

        self.tail = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, out_dim, kernel_size=1)
        )

    def forward(self, x):
        # Same as your original function
        x = x.transpose(-1, -2).contiguous()
        x = self.conv(x)
        for bind in range(self.body_num):
            x1 = self.body1[bind](x)
            x = self.bn_list[bind](x + x1)
        x = self.dropout(x)
        x = self.tail(x).transpose(-1, -2).contiguous()
        return torch.softmax(x, dim=-1)


class JModel(nn.Module):
    def __init__(self, model):
        super(JModel, self).__init__()
        self.model = model
        self.useless = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.to(self.useless.device)
        model = self.model.to(self.useless.device)
        return model(x)


def load_model(emb_dim=1280, body_num=5):
    model = CoreModel(
        in_dim=emb_dim,
        out_dim=2,
        body_num=body_num,
        dr=0.5
    )
    return model


def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if len(name) > 1:
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if len(seq_list) > 0:
        ans[name] = "".join(seq_list)
    return ans

'''
def dateTag():
    time_tuple = time.localtime(time.time())
    yy = time_tuple.tm_year
    mm = f"{time_tuple.tm_mon:02d}"
    dd = f"{time_tuple.tm_mday:02d}"
    date_tag = f"{yy}{mm}{dd}"
    return date_tag'''


'''def timeTag():
    time_tuple = time.localtime(time.time())
    hour = f"{time_tuple.tm_hour:02d}"
    minuse = f"{time_tuple.tm_min:02d}"
    second = f"{time_tuple.tm_sec:02d}"
    time_tag = f"{hour}:{minuse}:{second}"
    return time_tag'''


'''def timeRecord(time_log, content):
    date_tag = dateTag()
    time_tag = timeTag()
    with open(time_log, 'a') as file_object:
        file_object.write("{} {} says: {}\n".format(date_tag, time_tag, content))'''


def load_precomputed_embeddings(pickle_file):
    with open(pickle_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


if __name__ == '__main__':

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-outfolder", "--outfolder")
    parser.add_argument("-seq_fa", "--seq_fa")
    parser.add_argument("-embeddings", "--embeddings_file")  # New argument for embeddings
    #parser.add_argument("-cutoff", "--prob_cutoff", type=float, default=0.48)
    parser.add_argument("-dv", "--device", default='cuda:0')
    args = parser.parse_args()

    if args.outfolder is None or args.seq_fa is None or args.embeddings_file is None:
        parser.print_help()
        exit("PLEASE INPUT YOUR PARAMETERS CORRECTLY")

    print_namespace(args)
    set_seed(2023)
    seq_fa = args.seq_fa
    outfolder = args.outfolder
    e2eatpm = "{}/e2eatpm/e2eatpm.pkl".format(os.path.abspath('.'))

    device = args.device if torch.cuda.is_available() else 'cpu'

    core_model = load_model()
    if os.path.exists(e2eatpm):
        checkpoint = torch.load(e2eatpm, map_location=device)
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if key.startswith('body2') or key.startswith('body3') or key.startswith('weights'):
                del state_dict[key]
        core_model.load_state_dict(state_dict)

    model = JModel(core_model).to(device)
    model.eval()

    seq_dict = loadFasta(seq_fa)

    # Load precomputed embeddings
    precomputed_embeddings = load_precomputed_embeddings(args.embeddings_file)

    '''start_index = args.start_index
    end_index = args.end_index
    if end_index <= start_index:
        end_index = len(seq_dict)'''

    keys = list(seq_dict.keys())
    tot_seq_num = len(seq_dict)
    for ind in range(tot_seq_num):

        key = keys[ind].split('|')[1]
        seq = seq_dict[keys[ind]]

        if ind % 1 == 0:
            print("The {}/{}-th {}({}) is predicting...".format(ind, tot_seq_num, key, len(seq)))

        # Use precomputed embeddings instead of embedding the sequence
        if key in precomputed_embeddings:
            emb = precomputed_embeddings[key]

            #emb_tensor = torch.tensor(emb).unsqueeze(0).to(device)  # Add batch dimension and move to device
            out = model(emb)
        else:
            print(f"Embedding for {key} not found.")
            continue

        probs = parsePredProbs(out)
        filepath = "{}/{}.tsv".format(outfolder, key)
        with open(filepath, 'w') as file_object:
            length = len(probs)
            file_object.write("Index\tAA\tProb\n")
            for i in range(length):
                aa = seq[i]
                prob = probs[i]
                file_object.write("{:5d}\t{}\t{:.3f}\n".format(i, aa, probs[i]))
                

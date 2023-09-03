from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,BertTokenizer,RobertaTokenizer
import os.path as osp
import json
import re
import copy
import random
import torch
import numpy as np
from utils import construct_graph





class DeepMathData(Dataset):
    def __init__(self,args) -> None:
        super().__init__()

        if args.model_name in ["bert-base-uncased","bert-large-uncased"]:
            self.tokenizer = BertTokenizer.from_pretrained(
                args.model_path, model_max_length=args.max_len)
            self.bos = self.tokenizer.cls_token + ' '
            self.sep = ' ' + self.tokenizer.sep_token + ' '
            self.eos = ' ' + self.tokenizer.sep_token


        elif args.model_name in ["roberta-base","roberta-large"]:

            self.tokenizer = RobertaTokenizer.from_pretrained(
                args.model_path, model_max_length=args.max_len)
            self.bos = self.tokenizer.bos_token + ' '
            self.sep = ' ' + self.tokenizer.sep_token + ' '
            self.eos = ' ' + self.tokenizer.eos_token

        else:
            raise NotImplementedError
        
        self.max_input_len = args.max_len
        self.max_output_len = args.max_len

        self.path = osp.join(args.data_path, "raw_data.jsonl")
        self.data = self.read_data(args)
    

    def read_data(self,args):
        datas = []

        #flag = 0
        for line in open(self.path):
            ex = json.loads(line)

            # flag+=1

            conclusion = [self.bos] + ex["hypothesis"] + [self.eos]
            conclu_str = ' '.join(str(i) for i in conclusion)
            if len(ex["premise"]) > args.max_len:
                ex["premise"] = ex["premise"][:args.max_len]
            prem_str = ' '.join(str(i) for i in ex["premise"])
            # print(prem_str)
            # print("len of prem_str",len(prem_str))
            # print(prem_str)

            # inputs = [self.bos]
            # inputs.extend(ex["hypothesis"])
            # inputs.append(self.sep)
            # inputs.extend(ex["premise"])
            # inputs.append(self.eos)


            #adj = construct_graph(ex["premise"])
            # print(len(ex["premise"]))
            adj = construct_graph(ex["premise"])
            

            # print("inputs_str",inputs_str)
            # if flag >10:
            #     break

            # # inputs = self.tokenizer.convert_tokens_to_ids(inputs_str) 

            # print("inputs",inputs)
            prems = self.tokenizer.convert_tokens_to_ids(prem_str.split(" "))
            # print(prems)
            # print("len of prems",len(prems))

            datas.append({
                "premise" : prems ,
                "conclusion" : conclu_str,
                "adj":adj,
                "label": ex["label"]
            })

            # if len(datas) > 100000:
            #     break

        random.shuffle(datas)
        return datas

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx:int) :
        return self.data[idx]
    
    def collate(self,batch):
        #print("-------------collate data------------------")

        conclu_inputs = [example["conclusion"] for example in batch]
        prem_inputs = [example["premise"] for example in batch]
        adjs = [example["adj"] for example in batch]


        node_pad_seqs = []
        node_mask = []
        lens = [len(s) for s in prem_inputs]
        #print(lens)
        max_len = np.array(lens).max()
        for s in prem_inputs:
            l = len(s)
            
            if l < max_len:
                
                # print("需要进行padding")
                # print(s)
                s = s + [0] * (max_len - l)
                mask = [1] * l + [0] * (max_len - l)
                assert len(s) == len(mask)
            else:
                mask = [1] *max_len
            node_pad_seqs.append(s)
            node_mask.append(mask)
        
        conclu_input = self.tokenizer(
            conclu_inputs, 
            padding=True,
            truncation=True, 
            add_special_tokens = False,
            return_tensors='pt')
        

        # prem_input = self.tokenizer(
        #     prem_inputs, 
        #     padding=True,
        #     truncation=True, 
        #     add_special_tokens = False,
        #     return_tensors='pt')

        conclu = conclu_input['input_ids']
        conclu_mask = conclu_input['attention_mask']
        # prem = prem_input['input_ids']
        # prem_mask = prem_input['attention_mask']

        label = torch.tensor([example["label"] for example in batch], dtype = torch.float32)

        adj_pad = []
        lens = [len(a) for a in adjs]
        # print(lens)

        max_len = np.array(lens).max()
        
        for a in adjs:
            pad = np.zeros((max_len,max_len))
            for i in range(len(a)):
                for j in range(len(a)):
                    pad[i,j] = a[i,j]
            adj_pad.append(pad)

        f = torch.LongTensor

        # print("adj_size",f(adj_pad).shape)
        # print("prem",f(node_pad_seqs).shape)


        return f(node_pad_seqs),f(node_mask),conclu,conclu_mask,f(adj_pad),label


if __name__ == '__main__':


    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default = "bert-base-uncased")
    parser.add_argument("--model_path", type=str, default = "/data/webw5/exp/BackChain/ProofWriter/local_model/bert-base")

    parser.add_argument("--data_path", type=str, default = "../processed_data")
    parser.add_argument("--task_type", type=str, default = "task2")

    parser.add_argument("--max_len", type=int, default = 512)
    parser.add_argument("--neg_sample_num", type=int, default = -1)

    args = parser.parse_args()


    dataset = DeepMathData(args)

    

    # for i in range(30):
    #     print(dataset.__getitem__(i))
    # print(len(dataset))
    # print(len(dataset))

    data_loader = DataLoader(
            dataset,
            2,
            shuffle = True,
            num_workers = 1,
            collate_fn = dataset.collate,
            pin_memory = True,
            drop_last = True,
        )
    
    


    f = 0 

    for i,d in enumerate(data_loader):
        # print("[[[[[[[[[[]]]]]]]]]]")
        print("-------------------------------")
        f += 1
        if f> 10:
            break










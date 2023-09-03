
from transformers import RobertaTokenizer,BertTokenizer
from torch.utils import data
import numpy as np
import torch
import json 

from config import set_config
from utils import extract,adj_concat
import copy
import random

import os.path as osp

class GnnDataset(data.Dataset):
    def __init__(self,args,mode):


        if args.model_name in ["bert-base-uncased","bert-large-uncased"]:
            self.tokenizer = BertTokenizer.from_pretrained(
                args.model_path, model_max_length = args.max_len)
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
        
        self.data = []
        self.max_len =  args.max_len

        self.path = osp.join(args.data_path, 'v0.0',"folio-" + mode + ".jsonl")

        with open(self.path,'r',encoding = 'utf-8') as f:
            for line in f.readlines():                
                sample = json.loads(line)   #将字符串转化为字典
                #print(sample.type)
                self.data.append(sample)

        self.data = []
        self.max_len =  args.max_len
        self.mode = mode
        self.args = args

        with open(self.path,'r',encoding = 'utf-8') as f:
            for line in f.readlines():                
                sample = json.loads(line)   #将字符串转化为字典
                self.data.append(sample)


    # def negative_sampling(self,sample):
    #     s = copy.deepcopy(sample) 
    #     if s['label'] == 'True':
    #     # print('true')
            
    #         prems = s['premises-FOL']
    #         operate_id = random.randint(0,len(prems))
    #         if operate_id == len(prems):
    #             s['conclusion'] =  'Not ' + s['conclusion']
    #         else:
    #             prems.remove(prems[operate_id])
    #         s['label'] = 'False'
    #         return s
        
    #print(s)

        
     #   if s['label'] == 'False':
    #    # print('false')
          #  s['conclusion'] =  'Not ' + s['conclusion']
#
 #           s['label'] = 'True'
  #          return s


    def __len__(self):
        return len(self.data)    

    def __getitem__(self,idx):
        result = self.data[idx]
        #print(result)
        labels = result['label']
        #train集和valid集差异 + label处理
        if self.mode == 'train':
            story_id = result['story_id']
            example_id = result['example_id']
            if labels == 'True':
                label = 0
            elif labels == 'False':
                label = 2
            elif labels == 'Unknown':
                label = 1
            else:
                raise ValueError("Unreasonable Error")

        elif self.mode == 'validation':
            conclusion_fol = result['conclusion-FOL']
            if labels == 'True':
                label = 0
            elif labels == 'False':
                label = 2
            elif labels == 'Uncertain':
                label = 1
            else:
                raise ValueError("Error Label")
        else:
            raise ValueError("Error mode ")

        #conclusion处理
        conclusion = result["conclusion"]
        conclu_seq = conclusion.split(' ')   #这里的fomula是指一个表达式
        conclu_seq = [p.lower() for p in conclu_seq if len(p) > 0]
        conclu_seq = ['<s>']  + conclu_seq +  ['</s>']
     
        #前提处理，得到节点序列和邻接矩阵
        premises_fol = result['premises-FOL']
        node_seq = ['<s>','symbol','predicate','variable']
        adj = np.eye((4))

        adj_list = []
        node_map_list = ['special'] * 4

        #np.set_printoptions(threshold=np.inf)
        for p in premises_fol:
            node_list,sub_adj,node_map = extract(p)   #处理单个表达式，得到节点序列和连接关系
            node_seq.extend(node_list)
            adj_list.append(sub_adj)
            node_map_list.extend(node_map)
        
        assert len(node_seq) == len(node_map_list)


        #句子内部逻辑连接 ：把adj列表中所有的邻接矩阵按对角线拼接

        for a in adj_list:
            adj = adj_concat(adj,a)
        assert len(node_seq) == len(adj)

        #句子间连接
        mark_sep = [idx for idx,node in enumerate(node_seq) if node == '</s>']
        for k in mark_sep:
            for j in mark_sep:
                adj[k,j] = 1
                adj[j,k] = 1
                adj[0,k] = 1
                adj[k,0] = 1

        #逻辑语义连接
        #print(node_map_list)
        for i,name in enumerate(node_map_list):
            if name == 'symbol':
                adj[i,1] = 1
                adj[1,i] = 1
            if name == 'predicate':
                adj[i,2] = 1
                adj[2,i] = 1
            if name == 'variable':
                adj[i,3] = 1
                adj[3,i] = 1
        
        for i in range(4):
                adj[0,i] = 1
                adj[i,0] = 1
        #CLS的连接(全局信息提取）
        # print(adj)
        #将node和concludion序列转化成id

        tokenizer = BertTokenizer.from_pretrained(self.args.model_path,add_special_tokens = True)
        conclu_id_seq = tokenizer.convert_tokens_to_ids(conclu_seq) 
        node_id_seq = tokenizer.convert_tokens_to_ids(node_seq) 

        #return node_seq,adj,conclu_seq,label
       # print("node_seq",node_id_seq)
       # print("conclu_seq",conclu_id_seq)
       # print("label",label)
        return node_id_seq,adj,conclu_id_seq,label

def padding(batch):
    node_seqs = [sample[0] for sample in batch]
    adjs = [sample[1] for sample in batch]
    conclu_seqs = [sample[2] for sample in batch]
    labels = [sample[3] for sample in batch]

    #conclusion  padding
    con_pad_seqs = []
    con_mask = []
    lens = [len(s) for s in conclu_seqs]
    #print(lens)
    max_len = np.array(lens).max()
    for s in conclu_seqs:
        l = len(s)
        
        if l < max_len:
            s = s + [0] * (max_len - l)
            m = [1] * l + [0] * (max_len - l)
        else:
            m = [1] * max_len
        con_pad_seqs.append(s)
        con_mask.append(m)

    #adj padding
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
    
    # print('adj数目',len(adj_pad))
    # print([len(a) for a in adj_pad])

    #premise_seq处理
    node_pad_seqs = []
    node_mask = []
    lens = [len(s) for s in node_seqs]
    #print(lens)
    max_len = np.array(lens).max()
    for s in node_seqs:
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

    f = torch.LongTensor

    return f(node_pad_seqs),f(node_mask),f(adj_pad),f(con_pad_seqs),f(con_mask),f(labels)

if __name__ == '__main__':
    args = set_config()


    train_set = GnnDataset(args,"train")
    valid_set = GnnDataset(args,"valid")

    # print(len(train_set))
    # print(len(valid_set))
    d = train_set.__getitem__(0)
    a = valid_set.__getitem__(0)

    train_loader = data.DataLoader(train_set,batch_size=args.batch_size,shuffle = True,collate_fn = padding)
    valid_loader = data.DataLoader(valid_set,batch_size=args.batch_size,shuffle = True,collate_fn = padding)

    # device = 'cuda'
    # for d in iter(train_loader):

	# 	#获取训练数据
    #     node_seq,adj,conclu_seq,label = d
    #     node_seq = node_seq.to(device)
    #     adj = adj.to(device)
    #     conclu_seq = conclu_seq.to(device)
    #     label = label.to(device)
    #     print("node_seq",node_seq.shape) #torch.Size([2, 42])
    #     print("adj",adj.shape)   #torch.Size([2, 42, 42])
    #     print("conclu_seq",conclu_seq.shape)  #torch.Size([2, 12])
    #     print("label",label.shape)  #torch.Size([2])
    #     # adj.to(device)




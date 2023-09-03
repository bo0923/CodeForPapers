
#先是一个简单的BERT/RoBERTa + 分类

import torch.nn as nn
import torch
from transformers import RobertaModel,BertModel,BertTokenizer,RobertaTokenizer
import math
import torch.nn.functional as F

from gcn import GCN

class Predict(nn.Module):
    def __init__(self,args):
        super().__init__()
        # 加载并冻结bert模型参数
        self.prediction = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(args.bert_embedding_dim *2 , args.linear_hidden),
            nn.BatchNorm1d( args.linear_hidden),
            nn.Tanh(),
            nn.Linear( args.linear_hidden, 2),
            nn.Softmax(dim = -1)
        )

    def forward(self,context):
  
        logits = self.prediction(context)
        y_hat = logits.argmax(-1)
        return logits,y_hat

#把predict替换成排序模型def Ranker(nn.Module)

#把predict替换成排序模型def Ranker(nn.Module)


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
    # query :batch_size * embedding_dim
    # k/v  : batch_size * seq_len * embedding_dim 
    # k/v_mask :batch_size *seq_len

    def forward(self,query,key,value,attn_mask = None):

        query = query.unsqueeze(1)
        #print("query",query.shape)
        attn = query @ key.transpose(1,2) / math.sqrt(query.size(2)) 
        #attn: b*1*s  attn_mask: b*s
        if attn_mask is not None:
            #print('attn',attn.shape)
            attn_mask = attn_mask.unsqueeze(1)
            # mask = torch.eq(attn_mask,1)
            #print("mask",mask.shape)
            #print(attn_mask)
            attn = attn.masked_fill(attn_mask == 0 ,float('-inf'))
            # print("attn",attn)
            #print(attn.shape)
        weights = F.softmax(attn,dim = -1)  #weights:b*1*s
        #print("weight",weights.shape)
        result = self.dropout(weights) @ value
        #print("result.shape",result.shape)
        return  result

class GraphModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 加载并冻结bert模型参数

        
        if args.model_name in ["bert-base-uncased","bert-large-uncased"]:
            self.embed = BertModel.from_pretrained(args.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(args.model_path)

            self._bos = self.tokenizer.cls_token + ' '
            self._sep = ' ' + self.tokenizer.sep_token + ' '
            self._eos = ' ' + self.tokenizer.sep_token
            self.output_dim = 768
            
        elif args.model_name in ["roberta-base","roberta-large"]:
            self.embed = RobertaModel.from_pretrained(args.model_path)
            self.tokenizer =  RobertaTokenizer.from_pretrained(args.model_path)

            self._bos = self.tokenizer.bos_token + ' '
            self._sep = ' ' + self.tokenizer.sep_token + ' '
            self._eos = ' ' + self.tokenizer.eos_token
            self.output_dim = 1024


        else:
            raise NotImplementedError
        
        self.gcn = GCN(args)
        self.attention = AttentionLayer()
        self.prediction = Predict(args)

        
    def forward(self,premise,prem_mask,conclu,conclu_mask,adj):
        # premise_seq =  premise_seq.to("cuda")

        conclu_embedded = self.embed(input_ids = conclu,attention_mask = conclu_mask,return_dict = True)["pooler_output"]

        # print(premise.shape)
        # print(prem_mask.shape)
        # print("conclu_embedded",conclu_embedded.shape)
        assert premise.shape == prem_mask.shape
        premise_embed = self.embed(input_ids = premise,attention_mask = prem_mask,return_dict = True)
        # print(premise_embed)
        premise_embedded = premise_embed.last_hidden_state
        # print("premie_embedded",premise_embedded.shape)

        node_hidden = self.gcn( premise_embedded,prem_mask,adj)
       # print("c_cls",c_cls.shape)
       # print("node_embedding",node_embedding.shape)
       # print("node_emedding[0]",node_embedding[0].shape)
       # print("node_embedding[:,0]",node_embedding[:,0].shape)
        # c_cls :batch_size * embedding_dim
        # p_hidden  : batch_size * seq_len * embedding_dim 
        # prem_mask :batch_size *seq_len
        # mask机制：希望prem_mask = 0 的位置对应的权重应该 = 0
        #result:batch*1 *embedding
        context_1 = self.attention(conclu_embedded,node_hidden,node_hidden,prem_mask)  #attn_mask应该是指k-v对中不需要关注的部分
    #    # print(c_cls.shape)
    #    # print(context_1.shape)
    #     context = torch.cat([c_cls,context_1.squeeze(1)], dim=1)
        context = torch.cat([conclu_embedded,context_1.squeeze(1)],dim = 1)

        # print("----------------------------")

        # print(context.shape)

        # print("----------------------------")
      
        logits,y_hat = self.prediction(context)
        return logits,y_hat








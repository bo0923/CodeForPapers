
import torch
import math
from transformers.trainer_pt_utils import get_parameter_names

from transformers.optimization import Adafactor,AdamW,get_scheduler

def create_optimizer(model,args):
    # decay if not LayerNorm or bias
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps":1e-6,
        }
    optimizer_kwargs["lr"] = args.learning_rate
    
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(optimizer, args):
    warmup_steps = math.ceil(args.num_training_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    return lr_scheduler



#variable,predicate,constant（一位且大写）

S_CONNECT = ['∧','∨','→','⊕',"="]
F_CONNECT = ['∀','∃','¬']



from typing import OrderedDict



def find_predicate(formula_map,formula,idx,direct):


    if direct == 'forward':
        for i,w in enumerate(reversed(formula[0:idx])):
            if formula_map[w] == "predicate":
                return len(formula[0:idx]) - i
    if direct == 'backward':
        for i,w in enumerate(formula[idx:]):
            if formula_map[w] == "predicate":
                return i
        
import numpy as np


def construct_graph(formula):    #input:一个序列（句子）(包括/不包括conclu)  
    #print(formula)

    formula_map = dict()   #语义字典
    adj_map = dict()      #连接关系字典
    formula = [w[0] if isinstance(w,list) else w for w in formula ]

    # print("fomula")
    # print(formula)
    # print("len",len(formula))
    # print("-----------------")

    #建立map:
    for w in formula:

        # if isinstance(w,list):
        #     w = w[0]

        if w in F_CONNECT or w in S_CONNECT:
            formula_map[w] = 'symbol'
        elif len(w) == 1:
            formula_map[w] = 'variable' 
        else:
            formula_map[w] = 'predicate'    #字典和谓词会有重复，所以需要更合理的方法，需要再修改一下

    # print(formula_map)

    #处理一元符和二元符之间的连接 和语义角色（fomula_map处理完）

    # #node_seq:返回的节点序列
    adj_map = dict()

    for idx,w in enumerate(formula):
        if w in F_CONNECT:
            adj_map[idx] = [idx+1]

        elif w in S_CONNECT:
            adj_map[idx] = []

            adj_map[idx].extend([find_predicate(formula_map,formula,idx,"forward"),find_predicate(formula_map,formula,idx,"backward")])

        elif formula_map[w] == 'predicate':
            adj_map[idx] = []
            adj_map[idx].extend([i for i in range(idx, idx + find_predicate(formula_map,formula,idx,"backward"))])

    # print(adj_map)
    # #node_seq:返回的节点序列

    adj = np.eye(len(formula))
    # print("adj.size",adj.shape)

    for i,w in enumerate(adj_map):
            
            for idx in adj_map[w]:
                if idx is not None:

                    adj[w,idx-1] = 1
                    adj[idx-1,w] = 1

    return adj

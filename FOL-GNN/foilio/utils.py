

S_CONNECT = ['∧','∨','→','⊕']
F_CONNECT = ['∀','∃','¬']
VARIABLE = ['x','y','z']

#extract函数的作用：输入一个逻辑表达式，返回他的图形式（节点列表+节点间的连接关系）
#其实是数据组织形式/方法上创新的关键模块
# for p in premises_fol:
#     node_list,node_map = extract(p)   #这个函数里面把分词直接做出来算了，然后根据分词和map都搞一下
#     node_list += ['SEP']
#     node_seq.extend(node_seq)

import re
import numpy as np


def extract(fomula):  
    """给定一个一阶逻辑表达式，提取出节点序列，每个节点对应的语义信
    息，并给出表示节点间连接关系的邻接矩阵  return :node_seq,adj,node_semantic_map"""
    #分词
    fomula = fomula.replace('(',' ')
    fomula = fomula.replace(')',' ')
    fomula = fomula.replace(',',' ')
    fomula = fomula.replace('∀','∀ ')
    fomula = fomula.replace('∃','∃ ')
    fomula = fomula.replace('¬','¬ ')

    fomula = fomula.split(' ')  
    fomula = [p.lower() for p in fomula if len(p) > 0]
    
    
    fomula_map = dict()   #语义字典
    adj_map = dict()      #连接关系字典
    for w in fomula:
        fomula_map[w] = 'predicate'
        adj_map[w] = []

    #pick出常量并成立成谓词 + 变量 + 析取符的形式
    constants = set(fomula[i+1] for i in range(len(fomula)-1) if len(fomula[i])>2 and len(fomula[i+1])>2)

    if  constants:        
        # print("句子中存在常量")
        fomula_map['x'] = 'variable'
        fomula_map['∧'] = 'symbol'
        adj_map['x'] = []
        adj_map['∧'] = [c for c in constants]

    #处理一元符和二元符之间的连接 和语义角色（fomula_map处理完）
    for idx,w in enumerate(fomula):
        if w in F_CONNECT:
            fomula_map[w] = 'symbol'
            adj_map[w].append(fomula[idx+1])
        if w in S_CONNECT:
            fomula_map[w] = 'symbol'
            adj_map[w].extend([fomula[idx-2],fomula[idx +1]])
        if w in VARIABLE:
            fomula_map[w] = 'variable' 
    
    #node_seq:返回的节点序列
    node_seq = [k for k,v in fomula_map.items()]
    # node_map：用于建立邻接矩阵，将连接的目标词转化为其对应的索引
    node_map = dict()
    for n in node_seq:
        node_map[n] = []
    

    #返回连接的序号
    for k,v in node_map.items():
        if fomula_map[k] == 'symbol':
            node_map[k].append([node_seq.index(w) for w in adj_map[k]])
        if fomula_map[k] == 'variable':
            node_map[k].append([node_seq.index(k) for k,v in fomula_map.items() if v == 'predicate'])

    node_seq += ['</s>']
    #node_map['SEP'] = [i for i in range(len(node_seq))]  #sep节点和句子中的每个单词相连接
    node_map['</s>'] = [node_seq.index(k) for k,v in fomula_map.items() if v == 'variable']

    # print(node_seq)
    # print(node_map)  #连接关系

    #根据连接关系(node_seq)建立邻接矩阵
    adj = np.eye(len(node_seq))
    for i,word in enumerate(node_seq):
        if node_map[word]:
            for idx in node_map[word]:
                adj[i,idx] = 1
                adj[idx,i] = 1
    # print(adj)


    map_list  = [v for k,v in fomula_map.items()] + ['</s>']
    # print(map_list)
    return node_seq,adj,map_list


#把b拼接到a上，然后返回a
def adj_concat(a,b):
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a,np.zeros((lenb,lena))))
    right = np.row_stack((np.zeros((lena,lenb)),b))
    m = np.hstack((left,right))
    return m





if __name__ == '__main__':

    # a = np.eye(5)
    # b = np.eye(3)
    # m = adj_concat(a,b)
    # print(m)

    f1 = "∀x (Drinks(x) ⊕ Jokes(x))"
    f2 = "(Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina))"
    extract(f1)
    extract(f2)
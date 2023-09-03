

#把preprocess好的数据全部存到一个文件里，然后直接随机划分（按照之前一贯的划分方式）

from parse_tptp import parse,get_label
import os
import json



path = './nndata'

save_path = './processed_data/raw_data.jsonl'  #需要根据label组成句子对

# print(os.walk(path))



datas = []

#需要针对每个文件的不同进行组织
for home,_,files in os.walk(path):
    print(home)
    for file in files:
        with open(os.path.join(path,file)) as f:

            hypos = []
            pos = []
            negs = []

            for data in f.readlines():

                data_label = get_label(data)
                if data_label == "hypo":
                    hypos.append(parse(data))
                if data_label == "pos":
                    pos.append(parse(data))
                if data_label == "neg":
                    negs.append(parse(data))

            for p in pos:
                data_dict = {
                    "hypothesis" :  hypos[0],
                    "premise": p,
                    "label" : True
                }

                with open(save_path, 'a') as write_f:  #train,test,dev分别做一遍
                    json_data = json.dumps(data_dict, sort_keys= True) #直接将字典数据写入字符串
                    
                    write_f.write(json_data)
                    write_f.write('\n')

            for n in negs:
                data_dict = {
                    "hypothesis" :  hypos[0],
                    "premise": n,
                    "label" : False
                }

                with open(save_path, 'a') as write_f:  #train,test,dev分别做一遍
                    json_data = json.dumps(data_dict, sort_keys= True) #直接将字典数据写入字符串
                    
                    write_f.write(json_data)
                    write_f.write('\n')




        


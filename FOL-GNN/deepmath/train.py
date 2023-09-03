




from config import set_config
import logging
import os.path as osp
import json

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepMathData

from utils import create_optimizer,create_scheduler

from model import  GraphModel

import os
#改变一下训练策略？看看能不能得到更高的accuracy
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
def run(args):

    torch.backends.cudnn.deterministic = True  #固定cudnn卷积算法，+ torch.seed，保证每次输出的结果一致
  
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Dataset


    #划分数据集

    full_dataset = DeepMathData(args)      #不用mode了

    train_size = int(args.trainset_rate * len(full_dataset))
    val_size = int(args.valset_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_set, val_set,test_set = torch.utils.data.random_split(full_dataset, [train_size, val_size,test_size])


    train_loader = DataLoader(
            train_set,
            args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
            collate_fn = full_dataset.collate,
            pin_memory = True,
            drop_last = True,
        )
    
    val_loader = DataLoader(
            val_set,
            args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
            collate_fn = full_dataset.collate,
            pin_memory = True,
            drop_last = True,
        )
    
    test_loader = DataLoader(
            test_set,
            args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
            collate_fn = full_dataset.collate,
            pin_memory = True,
            drop_last = True,
        )


    args.num_training_steps = args.epoch * len(train_loader)

    logging.info(f"{len(train_loader)*args.batch_size} Training Samples loaded")

    logging.info(f"{len(val_loader)*args.batch_size} Val Samples loaded")
    logging.info(f"{len(test_loader)*args.batch_size} Test Samples loaded")

    logging.info("loading model")

    model = GraphModel(args)

    if args.resume_path:
        # state_dict = torch.load(args.resume_path, map_location='cuda:0')
        state_dict = torch.load(args.resume_path)
        model.load_state_dict(state_dict)
        logging.info(f"Resume model parameters form {args.resume_path}")

    model = model.to(device)
    logging.info("model loaded")

    with open (osp.join(args.exp_dir, 'model.config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=False, indent=4)


    #这里能不能设置warm up:从utils来初始化optimizer和learning schedule
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr,weight_decay = args.weight_decay)

    optimizer = create_optimizer(model, args)
    lr_scheduler = create_scheduler(optimizer, args)
    
    # optimizer = create_optimizer(model, args)
    # lr_scheduler = create_scheduler(optimizer, args)

    crit = torch.nn.CrossEntropyLoss()

    import time
    exp_time = time.strftime('%Y-%m-%d', time.localtime())

    logging.info("start training")

    logger = {"best_epoch": -1, "val_acc": -1}


    for i in range(1,args.epoch+1):

        logging.info(f"---------------- Epoch {i} Start ----------------")

        model.train()

        acc_count = 0
        total_num = 0
        total_loss = 0

        for batch in tqdm(train_loader):
            #这里要根据数据和model有不同的形式了

            prem,prem_mask,conclu,conclu_mask,adj,label = batch
 
            f = lambda x : x.to(device)
            prem_mask = f(prem_mask)
            prem = f(prem)
            conclu = f(conclu)
            conclu_mask = f(conclu_mask)
            adj = f(adj)
            label = f(label)

            logits,y_pred= model(prem,prem_mask,conclu,conclu_mask,adj) #(logits)

            correct_num = torch.sum(y_pred == label).item()
            acc_count += correct_num

            #计算总样本数
            total_num += label.shape[0]
            loss = crit(logits,label.long())   #将logits和batch进行比较，但是torch自带的损失函数会将离散的label转化为one-hot的
            
            #contrast_loss = SupConLoss(context,batch_labels)
            #total_loss =  total_loss + 0.5 * loss.item() + 0.5 * contrast_loss
            total_loss +=  loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            
            optimizer.step()
            lr_scheduler.step()
            
            optimizer.zero_grad()  

            # batch_compute_time = time.time() - compute_start
            # compute_time += batch_compute_time
            #nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)  clip和weight_decay的数据都大约设置成多少


        train_acc = acc_count / total_num
        train_loss = total_loss / total_num
        

        logging.info("Epoch:{}  train loss:{} train accuracy: {}% ".format(i,train_loss,train_acc * 100))
        

        logging.info("------Start Validation----------------")

        val_correct_num = 0
        total_num = 0
 
        with torch.no_grad():

            for batch in tqdm(val_loader):

                prem,prem_mask,conclu,conclu_mask,adj,label = batch
 
                f = lambda x : x.to(device)
                prem_mask = f(prem_mask)
                prem = f(prem)
                conclu = f(conclu)
                conclu_mask = f(conclu_mask)
                adj = f(adj)
                label = f(label)

                logits,y_pred= model(prem,prem_mask,conclu,conclu_mask,adj) #(logits)
                #这里要根据数据和model有不同的形式了

                correct_num = torch.sum(y_pred == label).item()
                val_correct_num += correct_num

                #计算总样本数
                total_num += label.shape[0]
                

            val_acc = val_correct_num / total_num


        logging.info("Epoch:{}   dev accuracy: {}% ".format(i,val_acc * 100))
        
        logging.info("------Start Testing----------------")

        test_correct_num = 0
        total_num = 0
 
        with torch.no_grad():

            for batch in tqdm(test_loader):
                #这里要根据数据和model有不同的形式了


                prem,prem_mask,conclu,conclu_mask,adj,label = batch
 
                f = lambda x : x.to(device)
                prem_mask = f(prem_mask)
                prem = f(prem)
                conclu = f(conclu)
                conclu_mask = f(conclu_mask)
                adj = f(adj)
                label = f(label)

                logits,y_pred= model(prem,prem_mask,conclu,conclu_mask,adj) #(logits)


                correct_num = torch.sum(y_pred == label).item()
                test_correct_num += correct_num

                #计算总样本数
                total_num += label.shape[0]
                

            test_acc = test_correct_num / total_num

        logging.info("Epoch:{}  test accuracy: {}% ".format(i,test_acc * 100))

        if test_acc > logger["val_acc"]:
            logger["val_acc"] = test_acc
            logger["best_epoch"] = i

            # save_path = osp.join(args.exp_dir,'saved_model',args.task_type,'epoch'+ str(i) +'.pth')


            save_path = osp.join(args.exp_dir,'saved_model_2e',exp_time+'best_model_1.pth')
            torch.save(model.state_dict(),save_path)

        logging.info("best eopoch is EPOCH {},test acc is {}%".format(logger["best_epoch"],logger["val_acc"]))

        logging.info(f"Epoch {i} finished")



if __name__ == '__main__':

    args = set_config()
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    with open(osp.join(args.exp_dir,'config.json'),'w') as f:
        json.dump(vars(args),f,sort_keys = True,indent = 4)

    run(args)

    open(osp.join(args.exp_dir, 'done'), 'a').close()
    import sys
    #logging.info('Python info: {}'.format(os.popen('which python').read().strip()))
    logging.info('Command line is: {}'.format(' '.join(sys.argv)))
    logging.info('Called with args:')


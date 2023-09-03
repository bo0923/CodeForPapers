

from config import set_config
import torch
import time


from torch.utils import data

from dataset import GnnDataset,padding
from model import ReasonModel
# from con_loss import SupConLoss
#import visdom

def train(args,model,train_loader,valid_loader,device,optimizer ,crit):
    
    logger = {"best_epoch": -1, "val_acc": -1}
    #contrast_crit = SupConLoss()
    for epoch in range(args.epoch):
        #begin_time = time.time()
        model.train()
        acc_count = 0
        total_num = 0
        compute_time = 0
        total_loss = 0
        loss_list = []
        for idx,data in enumerate(train_loader):
            batch_node,node_mask,batch_adj,batch_conclus,conclu_mask,batch_labels = data
            f = lambda x: x.to(device)
            batch_node = f(batch_node)
            batch_adj = f(batch_adj)
            batch_conclus = f(batch_conclus)
            batch_labels = f(batch_labels)
            node_mask = f(node_mask)
            conclu_mask = f(conclu_mask)
            
        
            model.zero_grad() #一个batch更新一次参数，每次计算更新参数前需要将梯度清零
            #compute_start = time.time()
            logits,y_pred= model(batch_node,node_mask,batch_adj,batch_conclus,conclu_mask) #(logits)
            #y_pred = torch.argmax(logits,dim = 1)
            correct_num = torch.sum(y_pred == batch_labels).item()
            acc_count += correct_num

            #计算总样本数
            total_num += batch_labels.shape[0]
            loss = crit(logits,batch_labels)   #将logits和batch进行比较，但是torch自带的损失函数会将离散的label转化为one-hot的
            
            #contrast_loss = SupConLoss(context,batch_labels)
            #total_loss =  total_loss + 0.5 * loss.item() + 0.5 * contrast_loss
            total_loss =  total_loss + 10 * loss.item()
            loss.backward()

            # batch_compute_time = time.time() - compute_start
            # compute_time += batch_compute_time
            #nn.utils.clip_grad_norm_(model.parameters(), prog_args.clip)  clip和weight_decay的数据都大约设置成多少
            optimizer.step()

        train_acc = acc_count / total_num
        train_loss = total_loss / total_num
        



        # vis.line(X=[epoch], Y=[train_loss], win=loss_window, opts=opt, update='append')
        # vis.line(X=[epoch], Y=[train_acc], win=acc_window, opts=opt_acc, update='append')


        print("Epoch:{}   loss:{}  accuracy: {} %".format(epoch,train_loss,train_acc * 100))

        

        #evaluate
        model.eval()
        correct_num = 0
        total = 0
        with torch.no_grad():
            for idx,batch_data in enumerate(valid_loader):
                batch_seq,node_mask,batch_adj,batch_conclu_seq,conclu_mask,batch_labels = batch_data
                f = lambda x: x.to(device)
                batch_seq = f(batch_seq)
                batch_adj = f(batch_adj)
                batch_labels = f(batch_labels)
                batch_conclu_seq = f(batch_conclu_seq)
                node_mask = f(node_mask)
                conclu_mask = f(conclu_mask)
         #       print('node_mask.shape',node_mask.shape)
          #      print('batch_node.shape',batch_node.shape)
           #     assert node_mask.shape == batch_node.shape

                logits,y_hat = model(batch_seq,node_mask,batch_adj,batch_conclu_seq,conclu_mask)

                val_correct = torch.sum(y_hat ==  batch_labels)
                correct_num += val_correct.item()
                total += batch_labels.shape[0]
        result = correct_num / total

        if result > logger["val_acc"] and epoch > 10:
            logger["val_acc"] = result
            logger["best_epoch"] = epoch
            torch.save(model.state_dict(),args.save_path + '_epoch' +str(epoch) +'.pth')
        
        loss_list.append(result)
        #vis.line(X=[epoch], Y=[result], win=val_acc_window, opts=val_opt_acc, update='append')
        
        print("The test accuracy is {}".format(result))
    print(loss_list)
    print("best eopoch is EPOCH {},acc is {}%".format(logger["best_epoch"],logger["val_acc"]))
        

#测试和训练的区别：（1）不进行参数更新  （2）不进行多个epoch的训练，完成一次的数据遍历和结果统计即可

def evaluate(args,model,data_loader,path,device):
    model.load_state_dict(torch.load(path))
    model.eval()
    correct_num = 0 
    with torch.no_grad:
        for idx,batch_data in enumerate(data_loader):
            batch_seq,batch_adj,batch_labels = batch_data
            f = lambda x: x.to(device)
            f = lambda x: x.to(device)
            batch_seq = f(batch_seq)
            batch_adj = f(batch_adj)
            batch_labels = f(batch_labels)

            y_logits = model(batch_seq,batch_adj)
            y_pred = torch.argmax(y_logits,dim = 1)
            correct = torch.sum(y_pred ==  batch_labels)
            correct_num += correct.item()
    result = correct_num / (len(data_loader) * args.batch_size)
    return result
            


torch.manual_seed(4077)
args = set_config()

device  = torch.device('cuda' if torch.cuda.is_available() else "cpu")


train_set = GnnDataset(args,"train")
valid_set = GnnDataset(args,'validation')

train_loader = data.DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle = False,collate_fn = padding)
valid_loader = data.DataLoader(valid_set,batch_size=args.batch_size,shuffle = True,collate_fn = padding)

#for i,data in enumerate(valid_loader):
#    n,n_mask,a,c,c_mask,l = data
#    print("n.shape",n.shape)
#    print("n.mask",n_mask.shape)
#    assert n.shape == n_mask.shape
model = ReasonModel(args)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(),lr = args.learning_rate,weight_decay = 1e-4)
crit = torch.nn.CrossEntropyLoss()


print("start training")

train(args,model,train_loader,valid_loader,device,optimizer = optimizer,crit= crit)




# vis = visdom.Visdom(env = 'main')
# opt= {
#     'xlabel':'epoch',
#     'ylabel':'loss',
#     'titel':'loss曲线'
# }

# loss_window = vis.line(
#     X=[0],
#     Y=[0],
#     opts=opt
# )


# opt_acc= {
#     'xlabel':'epoch',
#     'ylabel':'acc',
#     'titel':'准确率曲线'
# }

# acc_window = vis.line(
#     X=[0],
#     Y=[0],
#     opts=opt_acc
# )



# val_opt_acc= {
#     'xlabel':'epoch',
#     'ylabel':'val_acc',
#     'titel':'验证集准确率曲线'
# }

# val_acc_window = vis.line(
#     X=[0],
#     Y=[0],
#     opts=opt_acc
# )
# print("start training")

# train(args,model,train_loader,valid_loader,device,optimizer = optimizer,crit= crit)

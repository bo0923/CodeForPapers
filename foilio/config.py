import argparse


def set_config():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_path", type=str, required=True)


    parser.add_argument("--model_name", type=str, default='bert-base-uncased') 
    parser.add_argument("--model_path", type=str, default='/root/BackChain/EntailmentBank/selector/bert-base')
    #数据路径
    parser.add_argument("--data_path", type=str, default='../data')
    
    #网络参数(超参)
    parser.add_argument("--gcn_num", type=int, default=2)
    parser.add_argument("--bert_embedding_dim", type=int, default=768)

    #优化参数
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=75)

    #保存路径
    parser.add_argument("--save_path", type=str, default='exp/')
    

    args = parser.parse_args()
    return args

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

from cbert_utils import set_seed, get_scheduler, get_optimizer, request_logger
from cbert_preprocess import pre_data

def set_model(args):
    model = BertForMaskedLM.from_pretrained(args.model_name)
    if args.data_path == 'trec':
        model.bert.embeddings.token_type_embeddings =  nn.Embedding(6, 768) # 레이블 클래스에 맞춰 차원 변경
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02) # 이거 필요한 듯
    elif args.data_path == 'SetFit/sst5':
        model.bert.embeddings.token_type_embeddings = nn.Embedding(5, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    return model 

def train(model, train_loader, val_loader, args):
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, train_loader, args)
    model.to(args.device)
    # criterion = nn.CrossEntropyLoss()
    best_epoch_loss = np.inf
    for epoch in range(args.epochs):
        print("=======Training========")
        logger.info(f"=======Training========")
        model.train()
        avg_loss = 0
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, data in bar:
            item = {key: val.to(args.device) for key, val in data.items()}
            
            # print(item)
            optimizer.zero_grad()
            outputs = model(**item)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()

            if (step+1) % 50 == 0:
                bar.set_description(f"epoch: {epoch+1}/{args.epochs} avg_loss: {avg_loss / 50:.5f}")
                print('')
                logger.info(f"epoch: {epoch+1}/{args.epochs} avg_loss: {avg_loss / 50:.5f}")
                avg_loss = 0

        print("=======Validation========")
        logger.info(f"=======Validation========")
        with torch.no_grad():
            avg_loss = 0
            total_val_loss = []
            bar = tqdm(enumerate(val_loader), total=len(val_loader))
            for step, data in bar:
                model.eval()
                item = {key: val.to(args.device) for key, val in data.items()}

                outputs = model(**item)
                loss = outputs.loss

                total_val_loss.append(loss.item())
                avg_loss += loss.item()
                if (step+1) % 50 == 0:
                    bar.set_description(f"epoch: {epoch+1}/{args.epochs} avg_loss: {avg_loss / 50:.5f}")
                 #    print("avg_loss: {}".format(avg_loss / 50))
                    print('')
                    logger.info(f"epoch: {epoch+1}/{args.epochs} avg_loss: {avg_loss / 50:.5f}")
                    avg_loss = 0
            if best_epoch_loss > sum(total_val_loss)/len(total_val_loss):
                # best_epoch_loss = sum(total_val_loss)/len(total_val_loss)
                best_epoch_loss = sum(total_val_loss)/len(total_val_loss)
                # torch.save(model.state_dict(), os.path.join(config['save_path'], '{}_best_model.pt'.format(config['data_name'])))
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'best_model.pt'))
                # torch.save(model, os.path.join(config['save_path'], '{}_best_model'.format(config['data_name'])))
                print(f"best model loss is {best_epoch_loss}, saving at {args.model_save_dir}")
                logger.info(f"best model loss is {best_epoch_loss}, saving at {args.model_save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    # parser.add_argument('--load_model', type=str, default='linear', help='linear, linearlstm, lstm')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--data_path', type=str, default='SetFit/sst2')
    parser.add_argument('--model_save_dir', type=str, default='./results')
    parser.add_argument('--data_save_dir', type=str, default='./datasets')
 
    parser.add_argument('--mask_ratio', type=float, default=2)
    parser.add_argument('--sample_size', type=float, default=1)

    parser.add_argument('--logger_path', type=str, default='./log/train_log2.log')
    parser.add_argument('--logger', type=str, default='train')

    args = parser.parse_args()
    # args.data_save_dir = os.path.join('./{}_data'.format(os.path.basename(os.path.normpath(args.data_path))))
    
    # if not os.path.exists('./log'):
    #     os.makedirs('./log', exist_ok=True)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)
    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir, exist_ok=True)
    if os.path.exists(args.logger_path):
        with open(args.logger_path, "w") as file:
            file.truncate(0)

    print(args.logger_path)
    logger = request_logger(f'{args.logger}', args)
    set_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = set_model(args)

    train_loader= pre_data(tokenizer, args, mode='train')   
    valid_loader= pre_data(tokenizer, args, mode='valid')

    
    logger.info(":::datetime:::{}:::".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(":::model:::{}:::".format(args.model_name))
    logger.info("epochs:{}, optimizer:{}".format(args.epochs, args.optimizer))
    logger.info("lr:{}, weight_decay:{}, num_warmup_steps:{}".format(args.lr, args.weight_decay, args.num_warmup_steps))
    train(model, train_loader, valid_loader, args)

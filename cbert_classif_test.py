import os
import numpy as np
import pandas as pd
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

import transformers
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertForSequenceClassification
from transformers import AdamW
from transformers import DataCollatorForLanguageModeling # for masking

import datasets
from datasets import load_dataset
from datasets import DatasetDict

import argparse
from datetime import datetime
from cbert_utils import set_seed, get_scheduler, get_optimizer, request_logger, set_num_labels
from cbert_classif_preprocess import pre_data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def test(model, test_loader, args):
    
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    total_trues = []
    total_preds = []
    batch_trues = []
    batch_preds = []
    
    print("=======Test========")
    logger.info(f"=======Test========")
    model.to(args.device)
    

                
    with torch.no_grad():
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for step, data in bar:
            num_labels = set_num_labels(args)
            model.eval()
            # item = {key: val.to(args.device) for key, val in data.items() if key != 'label'}
            item = {key: val.to(args.device) for key, val in data.items()}
            outputs = model(**item)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            trues = item['labels']

            total_trues.extend(trues.tolist()) 
            total_preds.extend(preds.tolist()) 
            batch_trues.extend(trues.tolist()) 
            batch_preds.extend(preds.tolist())

            if (step+1) % 20 == 0:
                acc = accuracy_score(batch_trues, batch_preds)
                if num_labels == 2:
                    precision, recall, f1, _ = precision_recall_fscore_support(batch_trues, batch_preds, average='binary')
                else: 
                    precision, recall, f1, _ = precision_recall_fscore_support(batch_trues, batch_preds, average='macro')

                bar.set_description(f"accuuracy: {acc:.5f} precision: {precision:.5f} recall: {recall:.5f} f1: {f1:.5f}")
                print('')
                logger.info(f"accuuracy: {acc:.5f} precision: {precision:.5f} recall: {recall:.5f} f1: {f1:.5f}")
                batch_trues = []
                batch_preds = []
        print('')
        total_acc = accuracy_score(total_trues, total_preds)
        total_precision, total_recall, total_f1, _ = precision_recall_fscore_support(total_trues, total_preds, average='macro')
        print(f"total_accuuracy: {total_acc:.5f} \ntotal_precision: {total_precision:.5f} \ntotal_recall: {total_recall:.5f} \ntotal_f1: {total_f1:.5f}")
        logger.info(f"total_accuuracy: {total_acc:.5f} \ntotal_precision: {total_precision:.5f} \ntotal_recall: {total_recall:.5f} \ntotal_f1: {total_f1:.5f}")
            


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

    parser.add_argument('--logger_path', type=str, default='./log/classif_train_log.log')
    parser.add_argument('--logger', type=str, default='train')

    parser.add_argument('--train_version', type=str, default='train')

    args = parser.parse_args()
    # args.data_save_dir = os.path.join('./{}_data'.format(os.path.basename(os.path.normpath(args.data_path))))
    args.model_save_path = os.path.join(args.model_save_dir, './classif_{}_best_model.pt'.format(args.train_version))
    
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
    
    num_labels = set_num_labels(args)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(args.model_save_path))
    # train_loader= pre_data(tokenizer, args, mode='train')   
    test_loader= pre_data(tokenizer, args, mode='test')

    
    logger.info(":::datetime:::{}:::".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(":::model:::{}:::".format(args.model_name))
    logger.info(":::dataset:::{}:::".format(args.data_path))
    test(model, test_loader, args)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import csv
import argparse
import shutil

from cbert_utils import set_seed
from cbert_preprocess import pre_data

def set_model(args):
    model_path = os.path.join(args.model_save_dir, 'best_model.pt')
    model = BertForMaskedLM.from_pretrained(args.model_name)

    if args.data_path == 'trec':
        model.bert.embeddings.token_type_embeddings =  nn.Embedding(6, 768) # 레이블 클래스에 맞춰 차원 변경
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02) # 이거 필요한 듯
    elif args.data_path == 'SetFit/sst5':
        model.bert.embeddings.token_type_embeddings = nn.Embedding(5, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    
    model.load_state_dict(torch.load(model_path))    
    return model

def augment(model, tokenizer, loader, args, writer):
    # padding_token_id = tokenizer.pad_token_id #0
    masking_token_id = tokenizer.mask_token_id #103

    for epoch in range(args.epochs):
        with torch.no_grad():
            model.to(args.device)
            model.eval()

            bar = tqdm(enumerate(loader), total=len(loader))
            for step, data in bar:
                item = {key: val.to(args.device) for key, val in data.items()}
                token_lens = [sent.tolist().count(1) for sent in item['attention_mask']]
                mask_id = [np.random.choice(range(1, l-1), size=max((l-1)//args.mask_ratio, 1), replace=False) for l in token_lens] # special token 제외, 중복 제외, 0과 l-1 제외
                for input, m_id in zip(item['input_ids'], mask_id):
                    input[m_id] = masking_token_id
                outputs = model(**item)
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                labels = [t[0] for t in item['token_type_ids'].tolist()]

                for input, m_id, pred, label in zip(item['input_ids'], mask_id, preds, labels):
                   pred = torch.multinomial(pred, num_samples=args.sample_size, replacement=True)[m_id] # max_len 중 mask_id에 해당하는 것만 뽑기
                   pred = pred.reshape(-1, len(m_id))
                   for p in pred:
                      input[m_id] = p
                      sent = tokenizer.decode(input, skip_special_tokens=True)
                      writer.writerow([sent, label])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=1)
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
    parser.add_argument('--sample_size', type=int, default=1)

    parser.add_argument('--logger_file', type=str, default='')
    parser.add_argument('--logger', type=str, default='')

    args = parser.parse_args()
    
    # logger = request_logger(f'{args.logger}', args)
    set_seed(args.seed) 
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # args.data_save_dir = os.path.join('./{}_data'.format(os.path.basename(os.path.normpath(args.data_path))))

    origin_path = os.path.join(args.data_save_dir, 'train.tsv')
    aug_path = os.path.join(args.data_save_dir, 'aug.tsv')
    # trian file copy
    if os.path.exists(aug_path):
        os.remove(aug_path)
    shutil.copy(origin_path, aug_path)

    file = open(aug_path, 'a', encoding='utf-8', newline='')
    # 파일 내용 지우기
    # file.truncate(0)
    writer = csv.writer(file, delimiter='\t')

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = set_model(args)

    train_loader = pre_data(tokenizer, args, mode='train')

    augment(model, tokenizer, train_loader, args, writer)
    file.close()
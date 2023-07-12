import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
  def __init__(self, texts, labels, args):
    self.texts = texts
    self.labels = labels
    self.args = args
    # self.isTest = isTest

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.texts.items()}
    item['labels'] = self.labels[idx]
    return item
  
  def __len__(self):
    return len(self.labels)
  
def pre_data(tokenizer, args, mode='train'):
    if mode == 'train':
       df = pd.read_csv(f'{args.data_save_dir}/train.tsv', sep="\t", header=None)
    elif mode == 'valid':
       df = pd.read_csv(f'{args.data_save_dir}/valid.tsv', sep="\t", header=None)
    elif mode == 'test':
        df = pd.read_csv(f'{args.data_save_dir}/test.tsv', sep="\t", header=None)
    elif mode == 'aug':
        df = pd.read_csv(f'{args.data_save_dir}/aug.tsv', sep="\t", header=None)
    
    df.rename(columns={0: 'text', 1: 'label'}, inplace=True)
    print('Data loaded!')
    
    enc = tokenizer(df['text'].tolist(), 
                   truncation=True, 
                   padding='max_length', 
                   max_length=args.max_len, 
                   return_tensors='pt')

    dataset = MyDataset(enc, df['label'].tolist(), args)


    if mode == 'train' or mode == 'aug':
       dataloader = DataLoader(dataset, 
                               batch_size=args.batch_size, 
                               sampler=RandomSampler(dataset), 
                               drop_last=True)
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                sampler=SequentialSampler(dataset),
                                drop_last=True)
    return dataloader


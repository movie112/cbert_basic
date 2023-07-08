from transformers import DataCollatorForLanguageModeling # for masking
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer
import pandas as pd
import os
 
class MyDataset(Dataset):
  def __init__(self, texts, labels, args):
    self.texts = texts
    self.labels = labels
    self.args = args
    # self.isTest = isTest

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.texts.items()}
    
    # token_type_ids -> segment_embeddings : 1 -> label
    for i in range(self.args.max_len):
        if item['input_ids'][i] == 0:
              item['token_type_ids'][i] = 0
        else:
              item['token_type_ids'][i] = self.labels[idx]
    # item['labels'] = self.labels[idx]
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

    collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer,
       mlm=True,
       mlm_probability=0.15,
       )
    if mode == 'train':
       dataloader = DataLoader(dataset, 
                               batch_size=args.batch_size, 
                               sampler=RandomSampler(dataset), 
                               collate_fn=collator, 
                               drop_last=True)
    elif mode == 'valid':
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                sampler=SequentialSampler(dataset),
                                collate_fn=collator, 
                                drop_last=True)
    elif mode == 'test' or mode == 'aug':
        dataloader = DataLoader(dataset, 
                                batch_size=args.batch_size, 
                                sampler=SequentialSampler(dataset),
                                # collate_fn=collator, 
                                drop_last=True)
    return dataloader


def load_split(data_path, save_dir, seed=42):
  from datasets import load_dataset
  ''' Dataset Load '''
  if data_path in ['trec', 'SetFit/subj', 'SetFit/sst5', 'SetFit/sst2']:
    data = load_dataset(data_path)
    print(data)
    # train, valid data split
    if data_path in ['trec', 'SetFit/subj']:
        data_train_val = data['train'].train_test_split(test_size=0.1, seed=seed)
        data =  DatasetDict({
            'train' : data_train_val['train'],
            'validation' : data_train_val['test'],
            'test' : data['test']
        })
    if data_path in ['trec']:
        data = data.remove_columns('fine_label')
        data = data.rename_column('coarse_label', 'labels')
    elif data_path in ['SetFit/sst5', 'SetFit/sst2', 'SetFit/subj']:
        data = data.remove_columns('label_text')
        data = data.rename_column('label', 'labels')
  else:
    # import pyarrow as pa
    data = pd.read_csv(data_path, sep="\t", header=None)
    data.rename(columns={0: 'text', 1: 'labels'}, inplace=True)
    data = Dataset.from_pandas(data)
    print(data)
    data_train_val = data.train_test_split(test_size=0.2, seed=seed)
    data_val_test = data_train_val['test'].train_test_split(test_size=0.5, seed=seed)
    data =  DatasetDict({
            'train' : data_train_val['train'],
            'validation' : data_val_test['train'],
            'test' : data_val_test['test']
    })

  train_df = pd.DataFrame(data['train'])
  valid_df = pd.DataFrame(data['validation'])
  test_df = pd.DataFrame(data['test'])
  train_df.to_csv(f'{save_dir}' + '/train.tsv', sep='\t', index=False, header=False)
  valid_df.to_csv(f'{save_dir}' + '/valid.tsv', sep='\t', index=False, header=False)
  test_df.to_csv(f'{save_dir}' + '/test.tsv', sep='\t', index=False, header=False)


if __name__ == '__main__':
  # os.makedirs('./sst2_data', exist_ok=True)
  # load_split('SetFit/sst2', './sst2_data') 
  os.makedirs('./sst5_data', exist_ok=True)
  load_split('SetFit/sst5', './sst5_data') 


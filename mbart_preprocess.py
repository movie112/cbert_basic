import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
  def __init__(self, source, target, tokenizer, max_len):
    self.source = source
    self.target = target
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.source)
  
  def __getitem__(self, idx):
    source_enc = self.tokenizer(self.source[idx], 
                                truncation=True, 
                                padding='max_length', 
                                max_length=self.max_len, 
                                return_tensors='pt')
    target_enc = self.tokenizer(self.target[idx],
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_len,
                                return_tensors='pt')
    return {
        'input_ids': source_enc['input_ids'].squeeze(),  # [max_length]
        'attention_mask': source_enc['attention_mask'].squeeze(),  # [max_length]
        'labels': target_enc['input_ids'].squeeze(),  # [max_length]
    }
  
# dataloader
def pre_data(tokenizer, args, mode='train'):
    if mode == 'train':
       df = pd.read_csv(f'{args.data_save_dir}/train.tsv', sep="\t", header=None)
    elif mode == 'val':
       df = pd.read_csv(f'{args.data_save_dir}/val.tsv', sep="\t", header=None)
    elif mode == 'test':
        df = pd.read_csv(f'{args.data_save_dir}/test.tsv', sep="\t", header=None)
    df.rename(columns={0: 'SRC', 1: 'TRG'}, inplace=True)
    print('Data loaded!')

    dataset = MyDataset(df['SRC'].tolist(), df['TRG'].tolist(), tokenizer, args.max_len)
    if mode == 'train':
       dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=RandomSampler(dataset), drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset), drop_last=True)
    return dataloader


####
def split_data(data_path, save_dir): ##
    print('Loading data...')
    tmp_df = pd.read_csv(data_path, sep="\t", header=None)
    total_df = tmp_df.iloc[:, :2]
    total_df.rename(columns={0: 'SRC', 1: 'TRG'}, inplace=True)

    train_df, temp_df = train_test_split(total_df, test_size=0.2, random_state=42)  # 전체 데이터를 학습/검증용으로 8:2 비율로 분할
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 남은 데이터를 검증/테스트용으로 1:1 비율로 분할
    print(len(train_df), len(val_df), len(test_df))
    #saving datasets
    print('Saving datasets...')
    train_df.to_csv(f'{save_dir}/train.tsv', index=False, sep='\t')
    val_df.to_csv(f'{save_dir}/val.tsv', index=False, sep='\t')
    test_df.to_csv(f'{save_dir}/test.tsv', index=False, sep='\t')

def make_tsv(dir, save_dir, names):
   lst = ['train', 'val', 'test']

   os.makedirs(save_dir, exist_ok=True)
   for l in lst:
      en_df = pd.read_csv(f'{dir}/{l}.{names[0]}', sep="\t", header=None)
      de_df = pd.read_csv(f'{dir}/{l}.{names[1]}', sep="\t", header=None)
      en_df[1] = de_df[0]
      en_df.to_csv(f'{save_dir}/{l}.tsv', index=False, sep='\t', header=None)

if __name__ == '__main__':
   make_tsv('/HDD/dataset/WMT/2016/multi_modal', './en_de_data', ['en', 'de'])


import os
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
# from transformers import MBartForConditionalGeneration, MBartConfig, MBartTokenizer # MBart50TokenizerFast, 

from transformers import MBartConfig, MBart50Tokenizer
from mbart_model import MBartForConditionalGeneration

from mbart_utils import set_seed, get_scheduler, get_optimizer, request_logger
from mbart_preprocess import pre_data

def train(model, train_loader, valid_loader, args, logger):
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, train_loader, args)

    # criterion = nn.CrossEntropyLoss()
    best_epoch_loss = np.inf
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0
        print("=======Training========")
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch in bar:
            item = {key: val.to(args.device) for key, val in batch.items()}

            optimizer.zero_grad()
            outputs = model(**item)
            loss = outputs[0]

            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()

            if (step+1) % 100 == 0:
                bar.set_description(f"epoch: {epoch+1}/{args.epochs} avg_loss: {avg_loss / 100:.5f}")
                print('')
                logger.info(f"epoch: {epoch+1}/{args.epochs} avg_loss: {avg_loss / 100:.5f}")
                avg_loss = 0

        print("=======Validation========")
        with torch.no_grad():
            model.eval()
            bar = tqdm(enumerate(valid_loader), total=len(valid_loader))    
            total_val_loss = []
            avg_loss = 0
            for step, batch in bar:
                item = {key: val.to(args.device) for key, val in batch.items()}
                outputs = model(**item) 
                loss = outputs[0]
                total_val_loss.append(loss.item())      
                avg_loss += loss.item()
                if (step + 1) % 100 == 0:
                    bar.set_description(f"loss: {avg_loss/100:.5f}, epoch: {epoch + 1}")
                    print('')
                    avg_loss =0 
            print(f"epoch_loss: {sum(total_val_loss)/len(total_val_loss)}")
            logger.info(f"epoch_loss: {sum(total_val_loss)/len(total_val_loss)}")
            if best_epoch_loss > sum(total_val_loss)/len(total_val_loss):
                best_epoch_loss = sum(total_val_loss)/len(total_val_loss)
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'best_model.pt'))#.format(os.path.basename(os.path.normpath(args.model_save_dir))))) # en_de_data
                print(f"best model loss is {best_epoch_loss}, saving at {args.model_save_dir}")
                logger.info(f"best model loss is {best_epoch_loss}, saving at {args.model_save_dir}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--num_warmup_steps', type=int, default=100)
    # parser.add_argument('--load_model', type=str, default='linear', help='linear, linearlstm, lstm')
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--model_name', type=str, default="facebook/mbart-large-50")
    # parser.add_argument('--origin_data_dir', type=str, default='/HDD/dataset/WMT/2016/multi_modal/') ###./datasets/fra.txt
    parser.add_argument('--data_save_dir', type=str, default='./en_de_data')
    parser.add_argument('--model_save_dir', type=str, default='./results')

    parser.add_argument('--src_lang', type=str, default='en_XX')
    parser.add_argument('--tgt_lang', type=str, default='de_DE')

    parser.add_argument('--logger_path', type=str, default='./log/log_mbart_train.log')
    parser.add_argument('--logger', type=str, default='log_mbart')

    args = parser.parse_args()

    # if not os.path.exists('./log'):
    #     os.makedirs('./log', exist_ok=True)
    if not os.path.exists(args.data_save_dir):
        os.makedirs(args.data_save_dir, exist_ok=True)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)
    if os.path.exists(args.logger_path):
        with open(args.logger_path, "w") as file:
            file.truncate(0)
    print(args.logger_path)
    logger = request_logger(f'{args.logger}', args)
    set_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = MBartConfig()
    config.vocab_size = 250027
    model = MBartForConditionalGeneration(config)
    model = model.to(args.device)

    tokenizer = MBart50Tokenizer.from_pretrained(args.model_name, src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    train_loader = pre_data(tokenizer, args, mode='train')
    valid_loader = pre_data(tokenizer, args, mode='val')

    logger.info(":::datetime:::{}:::".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(":::model:::{}:::".format(args.model_name))
    logger.info("epochs:{}, optimizer:{}".format(args.epochs, args.optimizer))
    logger.info("lr:{}, weight_decay:{}, num_warmup_steps:{}".format(args.lr, args.weight_decay, args.num_warmup_steps))
    
    train(model, train_loader, valid_loader, args, logger)
    

    
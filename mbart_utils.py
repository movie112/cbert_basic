import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import transformers

# setting seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_scheduler(optimizer, train_dataloader, args, name='linear'):
    if name == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    elif name == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    return scheduler
# optimizer
def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
     
    return optimizer
def request_logger(logger_name, args):

    # os.remove(args.logger_file)
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) > 0:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(args.logger_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
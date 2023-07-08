import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
# from transformers import MBartForConditionalGeneration, MBartConfig, MBartTokenizer # MBart50TokenizerFast, 

from transformers import MBartConfig, MBart50Tokenizer
from mbart_model import MBartForConditionalGeneration

from nlgeval import NLGEval

from mbart_utils import set_seed, request_logger
from mbart_preprocess import pre_data

def test(model, test_loader, args, logger):
    print("=======Testing========")
    with torch.no_grad():

            ref_list = []
            hyp_list = [] # for nlg-eval
            total_correct = 0
            total_samples = 0

            Eval = NLGEval(metrics_to_omit=['CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
            model.eval()

            bar = tqdm(enumerate(test_loader), total=len(test_loader))
            for step, batch in bar:
                item = {key: val.to(args.device) for key, val in batch.items()}
                labels = item['labels']
                non_pad_mask = labels.ne(tokenizer.pad_token_id)
                
                generated_ids = model.generate(item['input_ids'], num_beams = args.beam_size, max_length=args.max_len, min_length=args.max_len) ##args로 변경

                decoded = [tokenizer.decode(row, skip_special_tokens=True) for row in labels]
                decoded = ['<s>'+row for row in decoded]
                encoded = [tokenizer.encode(row, add_special_tokens=False, truncation=True, padding='max_length', max_length=args.max_len) for row in decoded]
                encoded_labels = torch.tensor(encoded).to(args.device)


                correct = (generated_ids[:, 1:] == encoded_labels[:, 1:]) & non_pad_mask[:, 1:]
                total_correct += correct.sum().item()
                total_samples += non_pad_mask[:, 1:].sum().item()
                accuracy = total_correct / total_samples

                bar.set_postfix({'Accuracy': accuracy})

                generated_texts = tokenizer.batch_decode(generated_ids[:, 1:], skip_special_tokens=True)
                label_texts = tokenizer.batch_decode(encoded_labels[:, 1:], skip_special_tokens=True)


                for r, h in zip(label_texts, generated_texts):
                    if '</s>' in h:
                        h = h[:h.index('</s>')]
                    ref_list.append(r)
                    hyp_list.append(h)
                    total_samples  += 1

                if step % 50 == 0 or step == len(test_loader) - 1:
                    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
                    logger.info(":::step:::{}:::test_acc:::{}:::".format(step, overall_accuracy))
                    print(":::step:::{}:::test_acc:::{}:::".format(step, overall_accuracy))

            total_correct /= total_samples

            metrics_dict = Eval.compute_metrics([ref_list], hyp_list)
            # Final - End of testing
            print(f"TEST - Acc: {total_correct:.4f}")
            print(f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
            print(f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
            print(f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
            print(f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
            print(f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
            print(f"TEST - Rouge_L: {metrics_dict['ROUGE_L']:.4f}")
            print(f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")
            # Final - End of testing
            logger.info(f"TEST - Acc: {total_correct:.4f}")
            logger.info(f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
            logger.info(f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
            logger.info(f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
            logger.info(f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
            logger.info(f"TEST - Bleu_avg: {(metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4:.4f}")
            logger.info(f"TEST - Rouge_L: {metrics_dict['ROUGE_L']:.4f}")
            logger.info(f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=5e-5)
    # parser.add_argument('--weight_decay', type=float, default=1e-2)
    # parser.add_argument('--scheduler', type=str, default='linear')
    # parser.add_argument('--num_warmup_steps', type=int, default=100)
    # parser.add_argument('--load_model', type=str, default='linear', help='linear, linearlstm, lstm')
    parser.add_argument('--device', type=str, default='cuda:1')

    parser.add_argument('--model_name', type=str, default="facebook/mbart-large-50")
#    parser.add_argument('--origin_data_dir', type=str, default='/HDD/dataset/WMT/2016/multi_modal/') ###./datasets/fra.txt
    parser.add_argument('--data_save_dir', type=str, default='./en_de_data')
    parser.add_argument('--model_save_dir', type=str, default='./results')

    parser.add_argument('--src_lang', type=str, default='en_XX')
    parser.add_argument('--tgt_lang', type=str, default='de_DE')

    parser.add_argument('--logger_path', type=str, default='./log/log_mbart_test.log')
    parser.add_argument('--logger', type=str, default='log_mbart')

    parser.add_argument('--beam_size', type=int, default=1) # 'beam', 'greedy' : 1
    
    args = parser.parse_args()

    args.model_save_path = os.path.join(args.model_save_dir, 'best_model.pt')#.format(os.path.basename(os.path.normpath(args.data_save_dir))))
    if os.path.exists(args.logger_path):
        with open(args.logger_path, "w") as file:
            file.truncate(0)

    print(args.logger_path)
    logger = request_logger(f'{args.logger}', args)
    set_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    config = MBartConfig()
    config.vocab_size = 250027
    print('model_path', args.model_save_path)
    model = MBartForConditionalGeneration(config)
    model.load_state_dict(torch.load(args.model_save_path))
    model = model.to(args.device)

    tokenizer = MBart50Tokenizer.from_pretrained(args.model_name, src_lang=args.src_lang, tgt_lang=args.tgt_lang)

    test_loader= pre_data(tokenizer, args, mode='test')

    logger.info(":::datetime:::{}:::".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(":::model:::{}:::".format(args.model_name))
    test(model, test_loader, args, logger)



            # result_df = pd.DataFrame(columns=['caption', 'reference', 'generated',
            #                           'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
            #                           'bleu_avg', 'rouge_l', 'meteor'])
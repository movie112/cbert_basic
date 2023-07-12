import os

if __name__ == '__main__':
    os.system("python mbart_train.py --epochs 10 \
              --batch_size 16 \
              --max_len 128 \
              --lr 5e-5 \
              --data_save_dir ./en_de_data \
              --model_save_dir ./en_de_data \
              --src_lang en_XX \
              --tgt_lang de_DE \
              --logger mbart_train \
              --logger_path ./en_de_data/mbart_train.log \
              --model_name facebook/mbart-large-cc25 \
              --device cuda:1")
    
    os.system("python mbart_test.py \
              --batch_size 16 \
              --max_len 128 \
              --data_save_dir ./en_de_data \
              --model_save_dir ./en_de_data \
              --src_lang en_XX \
              --tgt_lang de_DE \
              --logger mbart_test \
              --logger_path ./en_de_data/mbart_test.log \
              --model_name facebook/mbart-large-cc25 \
              --device cuda:1")

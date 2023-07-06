import os

if __name__ == '__main__':
    os.system("python mbart_train.py --epochs 10 \
              --batch_size 8 \
              --max_len 128 \
              --lr 5e-5 \
              --data_save_dir ./en_de_data \
              --model_save_dir ./en_de_data \
              --src_lang en_XX \
              --tgt_lang de_DE \
              --logger log_mbart \
              --logger_path ./en_de_data/mbart_train.log \
              --device cuda:0")
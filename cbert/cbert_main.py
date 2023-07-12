import os

if __name__ == '__main__':
    os.system("python cbert_train.py --epochs 10 \
              --batch_size 16 \
              --max_len 128 \
              --lr 5e-5 \
              --data_path SetFit/sst2 \
              --data_save_dir ./sst2_data \
              --model_save_dir ./sst2_data \
              --logger cbert\
              --logger_path ./sst2_data/cbert_train.log \
              --device cuda:3") 

    os.system("python cbert_aug.py --epochs 1 \
              --batch_size 16 \
              --max_len 128 \
              --data_path SetFit/sst5 \
              --data_save_dir ./sst5_data \
              --model_save_dir ./sst5_data \
              --mask_ratio 2 \
              --sample_size 1 \
              --device cuda:1") 
    
    os.system("python cbert_classif_train.py --epochs 20 \
              --batch_size 16 \
              --max_len 128 \
              --lr 1e-7 \
              --data_path SetFit/sst5 \
              --data_save_dir ./sst5_data \
              --model_save_dir ./sst5_data \
              --logger cbert\
              --logger_path ./sst5_data/cbert_classif_train.log \
              --train_version train \
              --device cuda:1") 
    
    os.system("python cbert_classif_train.py --epochs 20 \
              --batch_size 16 \
              --max_len 128 \
              --lr 1e-7 \
              --data_path SetFit/sst5 \
              --data_save_dir ./sst5_data \
              --model_save_dir ./sst5_data \
              --logger cbert\
              --logger_path ./sst5_data/cbert_classif_train.log \
              --train_version aug \
              --device cuda:1") 

    os.system("python cbert_classif_test.py \
              --batch_size 16 \
              --max_len 128 \
              --data_path SetFit/sst2 \
              --data_save_dir ./sst2_data \
              --model_save_dir ./sst2_data \
              --logger cbert\
              --logger_path ./sst2_data/cbert_classif_train_test.log \
              --train_version train \
              --device cuda:0") 
    
    os.system("python cbert_classif_test.py \
              --batch_size 16 \
              --max_len 128 \
              --data_path SetFit/sst2 \
              --data_save_dir ./sst2_data \
              --model_save_dir ./sst2_data \
              --logger cbert\
              --logger_path ./sst2_data/cbert_classif_aug_test.log \
              --train_version aug \
              --device cuda:0") 

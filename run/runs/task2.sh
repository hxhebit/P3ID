#!/usr/bin/env bash
data_dir=./dataset # dataset path
PORT=29502
CONFIG=run/args/task2.yaml # config file
save_folder=$1  # folder to save

# run
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=$PORT main.py --config $CONFIG --data_dir $data_dir --src_domain A_train --tgt_domain C_train --tgt_domain_valid B_valid --tgt_domain_test B_test --save_folder $save_folder
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=$PORT main.py --config $CONFIG --data_dir $data_dir --src_domain A_train --tgt_domain B_train --tgt_domain_valid C_valid --tgt_domain_test C_test --save_folder $save_folder
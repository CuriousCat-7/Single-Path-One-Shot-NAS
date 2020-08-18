CUDA_VISIBLE_DEVICES=4 python supernet.py \
    --exp_name k50m100_mcucb_spos_cifar10 \
    --data_dir ~/data/torch\
    --classes 10\
    --dataset cifar10\
    --sample_method mcucb\
    --freq_weight 1000\
    --k 50\
    --m 100\
    --mc_sample_num 1000\

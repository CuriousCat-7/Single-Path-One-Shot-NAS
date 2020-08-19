CUDA_VISIBLE_DEVICES=5 python supernet.py \
    --exp_name k1m2_mcucb_spos_cifar10 \
    --data_dir ~/data/torch\
    --classes 10\
    --dataset cifar10\
    --sample_method mcucb\
    --freq_weight 100\
    --k 1\
    --m 2\

CUDA_VISIBLE_DEVICES=6 python supernet.py \
    --exp_name mcucb_spos_cifar10 \
    --data_dir ~/data/torch\
    --classes 10\
    --dataset cifar10\
    --sample_method mcucb\
    --freq_weight 100\

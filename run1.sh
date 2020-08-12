#CUDA_VISIBLE_DEVICES=6 python supernet.py \
#    --exp_name mcucb_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\
#    --sample_method mcucb\
#    --freq_weight 100\

CUDA_VISIBLE_DEVICES=0,1,2,3 python supernet.py \
    --exp_name v2_mcucb_spos_cifar10 \
    --data_dir ~/data/torch\
    --classes 10\
    --dataset cifar10\
    --sample_method mcucb\
    --freq_weight 100\
    --batch_size 768 \
    --sampler_batch_size 96 \

#CUDA_VISIBLE_DEVICES=6 python supernet.py \
#    --exp_name mcucb_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\
#    --sample_method mcucb\
#    --freq_weight 100\
# 开山，效果不错


#CUDA_VISIBLE_DEVICES=0,1,2,3 python supernet.py \
#    --exp_name v2_mcucb_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\
#    --sample_method mcucb\
#    --freq_weight 100\
#    --batch_size 768 \
#    --sampler_batch_size 96 \
# 多GPU效果不好


# v3 使用雨露均沾式Q值更新，效果不好，过拟合严重。


#CUDA_VISIBLE_DEVICES=6 python supernet.py \
#    --exp_name v4_mcucb_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\
#    --sample_method mcucb\
#    --freq_weight 100\
# 重复开山的实验，效果可复现，下面是valid_acc 90 的choice

#CUDA_VISIBLE_DEVICES=6 python choice_model.py\
#    --exp_name v4_mcucb_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\
#    --choice '[3, 2, 3, 3, 1, 1, 2, 0, 0, 3, 2, 2, 3, 2, 3, 2, 0, 2, 2, 1]'

#python flops_counter.py\
#    --exp_name uniform_spos_cifar10 \
#    --classes 10\
#    --dataset cifar10\
#    --choice '[3, 2, 3, 3, 1, 1, 2, 0, 0, 3, 2, 2, 3, 2, 3, 2, 0, 2, 2, 1]'

CUDA_VISIBLE_DEVICES=6 python random_search.py\
    --exp_name  v4_mcucb_spos_cifar10\
    --data_dir ~/data/torch\
    --classes 10\
    --dataset cifar10\

#CUDA_VISIBLE_DEVICES=7 python supernet.py \
#    --exp_name uniform_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\

#CUDA_VISIBLE_DEVICES=7 python choice_model.py\
#    --exp_name uniform_spos_cifar10 \
#    --data_dir ~/data/torch\
#    --classes 10\
#    --dataset cifar10\

python flops_counter.py\
    --exp_name uniform_spos_cifar10 \
    --classes 10\
    --dataset cifar10\

import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
from thop import profile
from torchvision import datasets
from utils import data_transforms
from model import SinglePath_OneShot, train, validate
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sampler import MCUCBSampler, UniformSampler
from torch.nn.parallel import DataParallel
from loguru import logger


def main():
    # args & device
    args = config.get_args()

    # tensorboard, logger
    tag = args.exp_name + '_super'
    writer = SummaryWriter(f"./snapshots/tb/{tag}")
    logger.add(f"snapshots/logs/{tag}.log")

    logger.info(args)

    if torch.cuda.is_available():
        logger.info('Train on GPU!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ## type checking
    assert args.sample_method in ["uniform", "mcucb"]

    # dataset
    assert args.dataset in ['cifar10', 'imagenet']
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=args.num_workers)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=args.num_workers)
    elif args.dataset == 'imagenet':
        train_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'train'), train_transform)
        val_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'valid'), valid_transform)
        train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)

    # SinglePath_OneShot
    model = SinglePath_OneShot(args.dataset, args.resize, args.classes, args.layers)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.sample_method == "uniform":
        sampler = UniformSampler([args.num_choices]*args.layers)
    elif args.sample_method == "mcucb":
        val_iter = utils.DataIterator(
                torch.utils.data.DataLoader(
                    valset, batch_size=args.sampler_batch_size,
                    shuffle=True , pin_memory=True, num_workers=args.num_workers))
        sampler = MCUCBSampler(
                [args.num_choices]*args.layers,
                val_iter,
                criterion,
                init_Q = args.init_Q,
                c = args.freq_weight,
                alpha=args.value_lr
                )
    else:
        raise NotImplementedError
    lr = args.learning_rate * args.batch_size / args.base_batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))

    # flops & params & structure
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),) if args.dataset == 'cifar10'
                            else (torch.randn(1, 3, 224, 224),), verbose=False)
    # print(model)
    logger.info('Random Path of the Supernet: Params: %.2fM, Flops:%.2fM' % ((params / 1e6), (flops / 1e6)))
    model = model.to(device)
    summary(model, (3, 32, 32) if args.dataset == 'cifar10' else (3, 224, 224))
    model = DataParallel(model)

    # train supernet
    start = time.time()
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, device, model, sampler, criterion, optimizer, scheduler, supernet=True, writer=writer)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            rst = validate(args, epoch, val_loader, device, model, sampler, criterion, supernet=True)
            for key, value in rst.items():
                writer.add_scalar(f"Valid/{key}", value, epoch+1)
            utils.save_checkpoint(
                    {
                        'state_dict': model.state_dict(),
                        "sampler_state_dict": sampler.state_dict()},
                    epoch + 1,
                    tag=tag)
    utils.time_record(start)


if __name__ == '__main__':
    main()

import os
import time
import utils
import torch
import torchvision
import torch.nn as nn
from config import get_args
from model import SinglePath_OneShot, validate
from tqdm import tqdm
from loguru import logger


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.add(f"snapshots/logs/{args.exp_name}_random_search")

    # one-shot
    model = SinglePath_OneShot(args.dataset, args.resize, args.classes, args.layers).to(device)
    ckpt_path = os.path.join('snapshots', args.exp_name + "_super" + '_ckpt_' + "{:0>4d}".format(args.epochs) + '.pth.tar')
    logger.info('Load checkpoint from:', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(
            utils.get_net_state_dict(
                checkpoint['state_dict']),
            strict=True)
    criterion = nn.CrossEntropyLoss().to(device)

    # dataset
    _, valid_transform = utils.data_transforms(args)
    valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
                                          download=False, transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=8)

    # random search
    start = time.time()
    best_acc = 0.0
    acc_list = list()
    best_choice = list()
    for epoch in tqdm(range(args.random_search)):
        choice = utils.random_choice(args.num_choices, args.layers)
        top1_acc = validate(args, epoch, val_loader, device, model, None, criterion, supernet=True, choice=choice)["top1_acc"]
        acc_list.append(top1_acc)
        if best_acc < top1_acc:
            best_acc = top1_acc
            best_choice = choice
    logger.info('acc_list:')
    for i in acc_list:
        logger.info(i)
    logger.info('best_acc:{} \nbest_choice:{}'.format(best_acc, best_choice))
    utils.plot_hist(acc_list, name=args.exp_name)
    utils.time_record(start)

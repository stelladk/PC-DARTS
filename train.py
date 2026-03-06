import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

import utils
from dataset import get_dataset, get_val_dataset
from model import NetworkCIFAR as Network

# ───────────────────────────────────────────── argument parsing ─────── #
parser = argparse.ArgumentParser("PC-DARTS evaluation – custom dataset")

# data
parser.add_argument("--data", type=str, default="../data")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="cifar10 | cifar100 | imagefolder | addnist | multnist | …",
)
parser.add_argument("--image_size", type=int, default=None)
parser.add_argument("--num_classes", type=int, default=None)
parser.add_argument(
    "--dataset_mean", type=float, nargs=3, default=None, metavar=("R", "G", "B")
)
parser.add_argument(
    "--dataset_std", type=float, nargs=3, default=None, metavar=("R", "G", "B")
)
parser.add_argument("--grayscale", action="store_true")
parser.add_argument(
    "--no_augment",
    action="store_true",
    default=False,
    help="Disable all training-time augmentation",
)

# NpyWebDataset-specific
parser.add_argument("--npyweb_url", type=str, default=None)
parser.add_argument("--npyweb_name", type=str, default="")
parser.add_argument("--npyweb_root", type=str, default="data/webdatasets/npy")
parser.add_argument("--npyweb_data_key", type=str, default="_x")
parser.add_argument("--npyweb_label_key", type=str, default="_y")
parser.add_argument("--npyweb_preload", action="store_true", default=True)

# training
parser.add_argument("--batch_size", type=int, default=96)
parser.add_argument("--learning_rate", type=float, default=0.025)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=3e-4)
parser.add_argument("--report_freq", type=int, default=50)
parser.add_argument("--epochs", type=int, default=600)
parser.add_argument("--grad_clip", type=float, default=5)

# model
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--init_channels", type=int, default=36)
parser.add_argument("--layers", type=int, default=20)
parser.add_argument("--auxiliary", action="store_true", default=False)
parser.add_argument("--auxiliary_weight", type=float, default=0.4)
parser.add_argument("--cutout", action="store_true", default=False)
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument("--drop_path_prob", type=float, default=0.3)
parser.add_argument(
    "--arch", type=str, default="PCDARTS", help="genotype name from genotypes.py"
)
parser.add_argument("--save", type=str, default="EXP")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
args.save = "eval-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p"
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# ───────────────────────────────────────────────────────── main ─────── #
def main():
    if not torch.cuda.is_available():
        logging.info("No GPU available, exiting.")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d", args.gpu)
    logging.info("args = %s", args)

    # ── dataset ────────────────────────────────────────────────────────
    train_data, n_classes, in_channels = get_dataset(args)
    valid_data = get_val_dataset(args)
    logging.info(
        "Dataset=%s  classes=%d  in_channels=%d", args.dataset, n_classes, in_channels
    )

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    # ── model ──────────────────────────────────────────────────────────
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, n_classes, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    logging.info("param size = %.2f MB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # ── loop ───────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info("epoch %d  lr %e", epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, _ = train_one_epoch(train_queue, model, criterion, optimizer)
        logging.info("train_acc %f", train_acc)

        valid_acc, _ = infer(valid_queue, model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info("valid_acc %f  best_acc %f", valid_acc, best_acc)

        utils.save(model, os.path.join(args.save, "weights.pt"))


def train_one_epoch(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (x, y) in enumerate(train_queue):
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, y)
        if args.auxiliary:
            loss += args.auxiliary_weight * criterion(logits_aux, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        p1, p5 = utils.accuracy(logits, y, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(p1.item(), n)
        top5.update(p5.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "train %03d  loss=%.4f  top1=%.2f  top5=%.2f",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
            )

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (x, y) in enumerate(valid_queue):
            x = Variable(x).cuda()
            y = Variable(y).cuda()

            logits, _ = model(x)
            loss = criterion(logits, y)

            p1, p5 = utils.accuracy(logits, y, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(p1.item(), n)
            top5.update(p5.item(), n)

            if step % args.report_freq == 0:
                logging.info(
                    "valid %03d  loss=%.4f  top1=%.2f  top5=%.2f",
                    step,
                    objs.avg,
                    top1.avg,
                    top5.avg,
                )

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()

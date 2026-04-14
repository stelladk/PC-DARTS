"""
train_search_custom.py  —  drop-in for PC-DARTS' train_search.py
================================================================
Works with any image classification dataset via dataset.py.

Quick-start examples
--------------------
# CIFAR-10  (identical to original)
python custom_train_search.py --data /data/cifar

# CIFAR-100
python custom_train_search.py --dataset cifar100 --data /data/cifar

# Your own folder dataset  (root/train/<class>/img.jpg)
python custom_train_search.py \\
    --dataset imagefolder --data /data/mydata \\
    --image_size 64 --batch_size 128

# STL-10
python custom_train_search.py \\
    --dataset stl10 --data /data/stl10 --image_size 96

# Flowers-102
python custom_train_search.py \\
    --dataset flowers102 --data /data/flowers \\
    --image_size 224 --batch_size 32 --train_portion 0.8

# Custom per-channel stats (compute with dataset.compute_dataset_stats)
python custom_train_search.py \\
    --dataset imagefolder --data /data/mydata \\
    --dataset_mean 0.45 0.42 0.38 \\
    --dataset_std  0.22 0.21 0.23
"""

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
import genotypes as gt
from architect import Architect
from dataset import get_dataset  # <-- the new module
from genotypes import Genotype
from model_search import Network
from model import NetworkCIFAR as NetworkEval
from logger import Logger
from genotypes import PRIMITIVES

# ───────────────────────────────────────────── argument parsing ─────── #
parser = argparse.ArgumentParser("PC-DARTS search – custom dataset")

# data
parser.add_argument("--data", type=str, default="/scratch/sdouka/data")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="cifar10 | cifar100 | imagefolder | stl10 | svhn | flowers102 | food101",
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

# logger
parser.add_argument("--no-logger", action="store_true")
parser.add_argument("--api", type=str, default="wandb")
parser.add_argument("--exp_name", type=str, default="NAS")
parser.add_argument("--port", type=int, default=27028)
parser.add_argument(
    "--log_path",
    type=str,
    default="/data/iceberg_1/titanic_1/experimentslogs_shared/tau_frugal/stella/",
)
parser.add_argument("--tmpdir", type=str, default="temp")

# NpyWebDataset-specific
parser.add_argument(
    "--npyweb_url",
    type=str,
    default=None,
    help="Direct ZIP download URL (required when --dataset npyweb)",
)
parser.add_argument(
    "--npyweb_name", type=str, default="", help="Cache key; SHA-256 of URL if not set"
)
parser.add_argument(
    "--npyweb_root",
    type=str,
    default="data/webdatasets/npy",
    help="Local cache directory for downloaded data",
)
parser.add_argument(
    "--npyweb_data_key",
    type=str,
    default="_x",
    help="Substring in filename identifying data  files",
)
parser.add_argument(
    "--npyweb_label_key",
    type=str,
    default="_y",
    help="Substring in filename identifying label files",
)
parser.add_argument(
    "--npyweb_preload",
    action="store_true",
    default=True,
    help="Load entire dataset into RAM on init (default True)",
)

# training
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--learning_rate_min", type=float, default=0.0)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=3e-4)
parser.add_argument("--report_freq", type=int, default=50)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--train_portion", type=float, default=0.5)
parser.add_argument("--unrolled", action="store_true", default=False)
parser.add_argument("--grad_clip", type=float, default=5)

# model / search
parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
parser.add_argument("--init_channels", type=int, default=16)
parser.add_argument("--layers", type=int, default=8)
parser.add_argument("--cutout", action="store_true", default=False)
parser.add_argument("--cutout_length", type=int, default=16)
parser.add_argument(
    "--no_augment",
    action="store_true",
    default=False,
    help="Disable all training-time augmentation (random crop, flip, cutout)",
)
parser.add_argument("--drop_path_prob", type=float, default=0.3)
parser.add_argument("--arch_learning_rate", type=float, default=6e-4)
parser.add_argument("--arch_weight_decay", type=float, default=1e-3)
parser.add_argument(
    "--arch_warmup_epochs",
    type=int,
    default=15,
    help="Epochs before architecture parameters start updating (default 15)",
)
parser.add_argument("--save", type=str, default="EXP")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument(
    "--init_genotype",
    type=str,
    default=None,
    help="Name of a genotype in genotypes.py to warm-start arch parameters from (e.g. PCDARTS_multnist_ep30)",
)

# evaluation phase (runs after search with the found genotype)
parser.add_argument("--skip_eval", action="store_true", default=False,
                    help="Skip the evaluation training phase after search")
parser.add_argument("--eval_epochs", type=int, default=100,
                    help="Epochs to train the found architecture (default 100; full PC-DARTS uses 600)")
parser.add_argument("--eval_init_channels", type=int, default=36,
                    help="Initial channels for the evaluation model (default 36)")
parser.add_argument("--eval_layers", type=int, default=20,
                    help="Total layers for the evaluation model (default 20)")
parser.add_argument("--eval_batch_size", type=int, default=96)
parser.add_argument("--eval_learning_rate", type=float, default=0.025)
parser.add_argument("--auxiliary", action="store_true", default=False,
                    help="Use auxiliary tower in the evaluation model")
parser.add_argument("--auxiliary_weight", type=float, default=0.4)

args = parser.parse_args()
args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p"
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# ──────────────────────────────────── arch warm-start from genotype ─── #
def _init_alphas_from_genotype(model, genotype, boost=5.0):
    """
    Warm-start the search model's architecture parameters (alphas / betas)
    so that the operations and edges described by *genotype* start with a
    higher logit than the rest.

    The model still explores freely — non-selected ops/edges start at 0
    while selected ones start at *boost* — but the search begins close to
    the given genotype rather than from a random flat prior.
    """
    steps = model._steps

    # Global edge index where step i begins
    step_starts = []
    s = 0
    for i in range(steps):
        step_starts.append(s)
        s += 2 + i

    def _fill(alphas, betas, gene):
        alphas.zero_()
        betas.zero_()
        for step_i in range(steps):
            for op_name, node_idx in (gene[2 * step_i], gene[2 * step_i + 1]):
                global_edge_idx = step_starts[step_i] + node_idx
                op_idx = PRIMITIVES.index(op_name)
                alphas[global_edge_idx][op_idx] += boost
                betas[global_edge_idx] += boost

    with torch.no_grad():
        _fill(model.alphas_normal.data, model.betas_normal.data, genotype.normal)
        _fill(model.alphas_reduce.data, model.betas_reduce.data, genotype.reduce)


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
    train_data, test_data, n_classes, in_channels = get_dataset(args)
    logging.info(
        "Dataset=%s  classes=%d  in_channels=%d", args.dataset, n_classes, in_channels
    )

    num_train = len(train_data)
    split = int(np.floor(args.train_portion * num_train))
    indices = list(range(num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=2,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=2,
    )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    # logger
    logger = Logger(experiment_name=args.exp_name, port=args.port, api=args.api, enabled=not args.no_logger)
    logger.setup_tracking(file_path=args.log_path)

    with logger(group="PC-DARTS"):
        for param, value in args._get_kwargs():
            if param in ("logger", "api", "exp_name", "port", "log_path", "tmpdir"):
                continue
            logger.log_parameter(f"{param}", value)

        # ── model ──────────────────────────────────────────────────────────
        criterion = nn.CrossEntropyLoss().cuda()

        genotype = None
        if genotype is not None:
            _run_eval_phase(
                genotype, n_classes, in_channels, train_data, test_queue,
                criterion, logger,
            )
            return

        model = Network(args.init_channels, n_classes, args.layers, criterion, in_channels=in_channels)
        model = model.cuda()

        if args.init_genotype is not None:
            init_geno = getattr(gt, args.init_genotype)
            _init_alphas_from_genotype(model, init_geno)
            logging.info("Warm-started arch parameters from genotype: %s", args.init_genotype)
            logging.info("  %s", init_geno)

        logging.info("param size = %.2f MB", utils.count_parameters_in_MB(model))
        logger.watch_pytorch_model(model)

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.learning_rate_min
        )

        architect = Architect(model, args)

        # ── loop ───────────────────────────────────────────────────────────
        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logging.info("epoch %d  lr %e", epoch, lr)

            genotype = model.genotype()
            logging.info("genotype = %s", genotype)

            train_acc, train_loss = train_one_epoch(
                train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch
            )
            logging.info("train_acc %f", train_acc)
            logger.log_metric("search/train accuracy", train_acc/100, epoch, "search epoch")
            logger.log_metric("search/train loss", train_loss, epoch, "search epoch")

            valid_acc, valid_loss = infer(valid_queue, model, criterion)
            logging.info("valid_acc %f", valid_acc)
            logger.log_metric("search/val accuracy", valid_acc/100, epoch, "search epoch")
            logger.log_metric("search/val loss", valid_loss, epoch, "search epoch")

            utils.save(model, os.path.join(args.save, "weights.pt"))
            logger.log_pytorch_model(model, f"PC-DARTS_{args.dataset}", x=None, path=args.tmpdir, run_id=False)

        # ── post-search ────────────────────────────────────────────────────
        genotype = model.genotype()
        logging.info("Final genotype = %s", genotype)

        del model, architect, optimizer, scheduler
        torch.cuda.empty_cache()

        if not args.skip_eval:
            _run_eval_phase(
                genotype, n_classes, in_channels, train_data, test_queue,
                criterion, logger,
            )


def train_one_epoch(
    train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (x, y) in enumerate(train_queue):
        model.train()
        n = x.size(0)
        x = Variable(x, requires_grad=False).cuda()
        y = Variable(y, requires_grad=False).cuda()

        # architecture gradient step — warm up weights first (original: epoch>=15)
        if epoch >= args.arch_warmup_epochs:
            x_val, y_val = next(iter(valid_queue))
            x_val = Variable(x_val, requires_grad=False).cuda()
            y_val = Variable(y_val, requires_grad=False).cuda()
            architect.step(x, y, x_val, y_val, lr, optimizer, unrolled=args.unrolled)

        # weight gradient step
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        p1, p5 = utils.accuracy(logits, y, topk=(1, 5))
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
            logits = model(x)
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

def train_eval_epoch(train_queue, model, criterion, optimizer):
    """Train one epoch of the discrete evaluation model (NetworkCIFAR)."""
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (x, y) in enumerate(train_queue):
        x = x.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, y)
        if logits_aux is not None:
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
            logging.info("eval_train %03d  loss=%.4f  top1=%.2f  top5=%.2f",
                         step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer_eval(queue, model, criterion):
    """Inference for NetworkCIFAR, which returns (logits, logits_aux)."""
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(queue):
            x = x.cuda()
            y = y.cuda()
            logits, _ = model(x)
            loss = criterion(logits, y)
            p1, p5 = utils.accuracy(logits, y, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(p1.item(), n)
            top5.update(p5.item(), n)
            if step % args.report_freq == 0:
                logging.info("eval_infer %03d  loss=%.4f  top1=%.2f  top5=%.2f",
                             step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg


def _run_eval_phase(genotype, n_classes, in_channels, train_data, test_queue, criterion, logger):
    """
    Build a fresh NetworkCIFAR from the found genotype, train it for
    args.eval_epochs, then evaluate on the test set.

    This is the standard PC-DARTS evaluation phase (equivalent to train.py),
    run inline immediately after the search.
    """
    logging.info("=== Evaluation phase: training found architecture ===")
    logging.info("  init_channels=%d  layers=%d  epochs=%d  auxiliary=%s",
                 args.eval_init_channels, args.eval_layers,
                 args.eval_epochs, args.auxiliary)

    eval_model = NetworkEval(
        args.eval_init_channels, n_classes, args.eval_layers,
        args.auxiliary, genotype, in_channels=in_channels,
    )
    eval_model = eval_model.cuda()
    eval_model.drop_path_prob = 0.0
    logging.info("eval model param size = %.2f MB",
                 utils.count_parameters_in_MB(eval_model))
    logging.info("eval model nb of parameters = %d", count_parameters(eval_model))
    logger.log_metric("training/nb of parameters", count_parameters(eval_model), 0, "epoch")

    full_train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.eval_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    eval_optimizer = torch.optim.SGD(
        eval_model.parameters(),
        args.eval_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    eval_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        eval_optimizer, args.eval_epochs, eta_min=0.0
    )

    for eval_epoch in range(args.eval_epochs):
        eval_model.drop_path_prob = args.drop_path_prob * eval_epoch / args.eval_epochs
        train_acc, train_loss = train_eval_epoch(full_train_queue, eval_model, criterion, eval_optimizer)
        eval_scheduler.step()

        logging.info("eval epoch %d  train_acc=%.2f  train_loss=%.4f", eval_epoch, train_acc, train_loss)
        logger.log_metric("training/train accuracy", train_acc / 100, eval_epoch, "epoch")
        logger.log_metric("training/train loss", train_loss, eval_epoch, "epoch")

        utils.save(eval_model, os.path.join(args.save, "eval_weights.pt"))

        test_acc, test_loss = infer_eval(test_queue, eval_model, criterion)
        logging.info("eval test_acc=%.2f  test_loss=%.4f", test_acc, test_loss)
        logger.log_metric("training/test accuracy", test_acc / 100, eval_epoch, "epoch")
        logger.log_metric("training/test loss", test_loss, eval_epoch, "epoch")
    logger.log_pytorch_model(eval_model, f"PC-DARTS_{args.dataset}_eval", x=None, path=args.tmpdir, run_id=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()

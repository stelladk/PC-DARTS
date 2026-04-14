"""
Derive the genotype from a saved PC-DARTS search checkpoint and evaluate
the search model on the test set.

weights.pt is saved at the END of each epoch (overwriting the previous one),
so it always contains the most-recent architecture parameters.

Pass --log to a slurm/training log and both the weights.pt path and the model
config are derived automatically:
  - save dir   →  parsed from "args = Namespace(..., save='...', ...)"
  - channels   →  parsed from init_channels=... in the same Namespace line
  - layers     →  parsed from layers=... in the same Namespace line

--weights, --channels, and --layers can still be passed explicitly to override.

Genotype source (choose one):
  - default: load weights.pt and call model.genotype() (uses both alphas+betas)
  - --from_log_genotype: read the last "genotype = Genotype(...)" line from the log

Usage:
    # fully automatic from log:
    python derive_and_eval.py --log slurm-budget_PC-DARTS_multnist-55780_4294967294.out \\
        --search_epoch 49

    # use last logged genotype instead of re-deriving from weights:
    python derive_and_eval.py --log slurm-budget_PC-DARTS_multnist-55780_4294967294.out \\
        --from_log_genotype --search_epoch 49

    # override weights path:
    python derive_and_eval.py --log slurm-budget_PC-DARTS_multnist-55780_4294967294.out \\
        --weights /other/path/weights.pt --search_epoch 49
"""

import sys
import re
import ast
import argparse
import numpy as np
import torch

if "/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/" not in sys.path:
    sys.path.append("/home/sdouka/Documents/Projects/InriaGitlab/experimental_grow/")
if "/home/tau/sdouka/codebase/experimental_grow" not in sys.path:
    sys.path.append("/home/tau/sdouka/codebase/experimental_grow")

from genotypes import Genotype, PRIMITIVES
from model_search import Network
from logger import Logger
import utils
from dataset import get_dataset

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--log',      required=True, help='path to the slurm/training log file')
parser.add_argument('--data',     default="/scratch/sdouka/data",  help='dataset root; parsed from --log if omitted')
parser.add_argument('--weights',  default=None,
                    help='path to weights.pt; derived from --log if omitted')
parser.add_argument('--from_log_genotype', action='store_true', default=False,
                    help='use the last logged genotype instead of re-deriving from weights')
parser.add_argument('--dataset',  default=None,
                    help='dataset name; parsed from --log if omitted')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--gpu',      type=int, default=0)
# search model config - must match the run that produced weights.pt
parser.add_argument('--channels', type=int, default=None,
                    help='init_channels; parsed from --log if omitted')
parser.add_argument('--layers',   type=int, default=None,
                    help='num layers; parsed from --log if omitted')
# dataset options (forwarded to get_dataset)
parser.add_argument('--image_size',     type=int,   default=None)
parser.add_argument('--num_classes',    type=int,   default=None)
parser.add_argument('--dataset_mean',   type=float, nargs=3, default=None, metavar=('R', 'G', 'B'))
parser.add_argument('--dataset_std',    type=float, nargs=3, default=None, metavar=('R', 'G', 'B'))
parser.add_argument('--grayscale',      action='store_true', default=False)
parser.add_argument('--no_augment',     action='store_true', default=True,
                    help='disable training-time augmentation for the test loader')
# NpyWebDataset options
parser.add_argument('--npyweb_url',       type=str, default=None)
parser.add_argument('--npyweb_name',      type=str, default='')
parser.add_argument('--npyweb_root',      type=str, default='data/webdatasets/npy')
parser.add_argument('--npyweb_data_key',  type=str, default='_x')
parser.add_argument('--npyweb_label_key', type=str, default='_y')
parser.add_argument('--npyweb_preload',   action='store_true', default=True)
# logging
parser.add_argument('--experiment_name', type=str, default='Budget')
parser.add_argument('--no-logger', action='store_true', default=False)
parser.add_argument('--logger_api', type=str, default='wandb', choices=['mlflow', 'wandb'])
parser.add_argument('--logger_port', type=int, default=27027)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--search_epoch', type=int, default=0)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)

tracker = Logger(args.experiment_name, port=args.logger_port,
                 api=args.logger_api, enabled=not args.no_logger)
tracker.setup_tracking(file_path=args.log_path)
tracker.start_run(group="PC-DARTS")

# ── helpers ───────────────────────────────────────────────────────────────────
def parse_log(log_path):
    """Parse save directory, model config, dataset, and data root from the
    training log.

    Reads the "args = Namespace(...)" line printed at the start of training.

    Returns (save_dir, channels, layers, dataset, data_root).
    dataset and data_root may be None if not found.
    """
    save_dir = None
    init_channels = None
    layers = None
    dataset = None
    data_root = None

    with open(log_path) as f:
        for line in f:
            m = re.search(r"args = Namespace\((.+)\)", line)
            if m:
                ns = m.group(1)

                mv = re.search(r"save='([^']+)'", ns)
                if mv:
                    save_dir = mv.group(1)

                mv = re.search(r'\binit_channels=(\d+)', ns)
                if mv:
                    init_channels = int(mv.group(1))

                mv = re.search(r'\blayers=(\d+)', ns)
                if mv:
                    layers = int(mv.group(1))

                mv = re.search(r"\bdataset='([^']+)'", ns)
                if mv:
                    dataset = mv.group(1)

                mv = re.search(r"\bdata='([^']+)'", ns)
                if mv:
                    data_root = mv.group(1)

                break  # only the first Namespace line matters

    if save_dir is None:
        raise ValueError(f"Could not find save= in log {log_path}")
    if init_channels is None or layers is None:
        raise ValueError(f"Could not find init_channels= or layers= in log {log_path}")

    return save_dir, init_channels, layers, dataset, data_root


def parse_last_genotype_from_log(log_path):
    """Return the last Genotype logged during training.

    Each epoch logs: "genotype = Genotype(...)".
    We return the last occurrence, which corresponds to the epoch whose
    weights.pt was saved.
    """
    last_genotype = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r'genotype = (Genotype\(.+\))', line)
            if m:
                last_genotype = m.group(1)
    if last_genotype is None:
        raise ValueError(f"No 'genotype = Genotype(...)' line found in {log_path}")
    return eval(last_genotype, {'Genotype': Genotype})


# ── resolve config from log ───────────────────────────────────────────────────
log_save_dir, log_channels, log_layers, log_dataset, log_data = parse_log(args.log)

weights  = args.weights  if args.weights  is not None else f'{log_save_dir}/weights.pt'
channels = args.channels if args.channels is not None else log_channels
layers   = args.layers   if args.layers   is not None else log_layers
dataset  = args.dataset  if args.dataset  is not None else log_dataset
data     = args.data     if args.data     is not None else log_data

if dataset is None:
    parser.error("Could not determine dataset from log. Pass --dataset explicitly.")
if data is None:
    parser.error("Could not determine data root from log. Pass --data explicitly.")

print(f'Log: {args.log}')
print(f'  save dir : {log_save_dir}')
print(f'  weights  : {weights}  {"(from log)" if args.weights is None else "(overridden)"}')
print(f'  channels : {channels}  layers: {layers}  {"(from log)" if args.channels is None and args.layers is None else "(overridden)"}')
print(f'  dataset  : {dataset}  data: {data}  {"(from log)" if args.dataset is None else "(overridden)"}')

# ── build dataset ─────────────────────────────────────────────────────────────
# Override args fields needed by get_dataset
args.dataset  = dataset
args.data     = data
args.cutout   = False

_, test_data, n_classes, in_channels = get_dataset(args)

test_queue = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size,
    shuffle=False, pin_memory=False, num_workers=2,
)

# ── build search model ────────────────────────────────────────────────────────
criterion = torch.nn.CrossEntropyLoss()
search_model = Network(channels, n_classes, layers, criterion, in_channels=in_channels)

# ── load search model weights ─────────────────────────────────────────────────
state = torch.load(weights, map_location='cpu')
# strip DataParallel 'module.' prefix if present
state = {k.replace('module.', ''): v for k, v in state.items()}

# diagnose shape mismatches before they crash load_state_dict
model_state = search_model.state_dict()
mismatches = [
    (k, state[k].shape, model_state[k].shape)
    for k in state if k in model_state and state[k].shape != model_state[k].shape
]
if mismatches:
    print("Shape mismatches between checkpoint and model:")
    for k, ckpt_shape, model_shape in mismatches:
        print(f"  {k}: checkpoint {ckpt_shape}  vs  model {model_shape}")
    raise RuntimeError(
        "Cannot load checkpoint. "
        "Check --channels / --layers match the run that produced weights.pt."
    )

search_model.load_state_dict(state)
search_model = search_model.cuda()
search_model.eval()

print(f"\nLoaded search model  |  channels={channels}  layers={layers}  "
      f"n_classes={n_classes}  in_channels={in_channels}")
for key, value in vars(args).items():
    tracker.log_parameter(key, str(value))

# ── derive genotype ───────────────────────────────────────────────────────────
if args.from_log_genotype:
    genotype = parse_last_genotype_from_log(args.log)
    print("\nGenotype from log (last logged epoch):")
else:
    genotype = search_model.genotype()
    print("\nDerived genotype (from loaded weights, alphas + betas):")
print(genotype)
tracker.log_parameter('genotype', str(genotype))

# ── evaluate search model on test set ────────────────────────────────────────
criterion_cuda = torch.nn.CrossEntropyLoss().cuda()
top1 = utils.AvgrageMeter()
objs = utils.AvgrageMeter()

with torch.no_grad():
    for step, (inputs, targets) in enumerate(test_queue):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = search_model(inputs)
        loss   = criterion_cuda(logits, targets)
        prec1, _ = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        if step % 50 == 0:
            print(f"test {step:03d}  loss={objs.avg:.4f}  acc={top1.avg:.2f}%")

print(f"\nSearch-model test accuracy:  {top1.avg:.2f}%  (loss {objs.avg:.4f})")
tracker.log_metrics(
    {"training/test accuracy": top1.avg / 100., "training/test loss": objs.avg},
    step=args.search_epoch, step_name="search epoch"
)
tracker.end_run()
print("\nTo train the eval model from scratch, add this genotype to genotypes.py")
print("and run custom_train_search.py (eval phase) or train.py with --arch <name>.")

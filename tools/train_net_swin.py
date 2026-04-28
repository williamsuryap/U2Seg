#!/usr/bin/env python
import os
import sys

from detectron2.config import get_cfg

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import train_net as base_train
from swin_helpers import add_swin_config


def setup(args):
    cfg = get_cfg()
    add_swin_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    base_train.default_setup(cfg, args)
    return cfg


def main(args):
    base_train.setup = setup
    return base_train.main(args)


if __name__ == "__main__":
    args = base_train.default_argument_parser().parse_args()
    args.eval_only = False
    print("Command Line Args:", args)
    base_train.launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

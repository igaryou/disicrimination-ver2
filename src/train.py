"""Thin MMEngine training entry point."""

import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train an MMSeg baseline')
    parser.add_argument('config')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume', nargs='?', const='auto', default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume:
        latest = Path(cfg.work_dir) / 'latest.pth'
        cfg.resume = True
        if args.resume == 'auto' and latest.is_file():
            cfg.load_from = str(latest)
        elif args.resume != 'auto':
            cfg.load_from = args.resume
    if args.no_wandb:
        cfg.visualizer.vis_backends = [
            backend for backend in cfg.visualizer.vis_backends
            if backend.get('type') != 'WandbVisBackend'
        ]
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()

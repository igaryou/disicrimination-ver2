"""Evaluate one checkpoint with the exact validation pipeline."""

import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Evaluate an MMSeg baseline')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()

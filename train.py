import argparse
import datetime
import warnings

import torch

from configs import get_cfg
from trainer import get_trainer
from util.net import distribute_bn, get_timepc, init_training
from util.util import able, init_checkpoint, log_msg, run_pre, update_log_term


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Training CLIP anomaly detection models")

    # model / config
    parser.add_argument(
        "-c",
        "--cfg_path",
        type=str,
        default="configs/clip_ad/clip_ad_mtvecad.py",
        help="Config file for the current model.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run training or only test a checkpoint.",
    )
    parser.add_argument(
        "--cssd_type",
        type=str,
        default=None,
        choices=["pdar", "cssd"],
        help="Select PDAR or sequential CSSD; omit to keep the config default.",
    )

    # training
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--batch_size_test", type=int, default=None, help="Override total test batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--test_start_epoch", type=int, default=None, help="First epoch to run real evaluation.")
    parser.add_argument("--test_every", type=int, default=None, help="Evaluate every N epochs.")

    # experiment / checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Folder used to save runs.")
    parser.add_argument("--resume_dir", type=str, default=None, help="Resume an existing run under checkpoint_dir.")
    parser.add_argument("--logdir_sub", type=str, default=None, help="Custom suffix for the new run folder.")

    # runtime, kept compatible with the original distributed launcher
    parser.add_argument("--sleep", type=int, default=-1)
    parser.add_argument("--memory", type=int, default=-1)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--logger_rank", default=0, type=int)

    parser.add_argument(
        "opts",
        help="Advanced config overrides, for example: data.root='data/mtvecad'",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    if args.epochs is not None:
        cfg.epoch_full = args.epochs
        cfg.trainer.epoch_full = args.epochs
        # A one-epoch smoke test previously produced decay_epochs=0, which
        # makes timm StepLRScheduler divide by zero in `t // decay_t`.
        cfg.trainer.scheduler_kwargs["decay_epochs"] = max(1, int(args.epochs * 0.8))
        if args.test_start_epoch is None:
            cfg.trainer.test_start_epoch = args.epochs
        if args.test_every is None:
            cfg.trainer.test_per_epoch = max(1, args.epochs // 10)

    if args.batch_size is not None:
        cfg.trainer.data.batch_size = args.batch_size
    if args.batch_size_test is not None:
        cfg.trainer.data.batch_size_test = args.batch_size_test
    if args.lr is not None:
        cfg.optim.lr = args.lr
        cfg.trainer.scheduler_kwargs["lr_min"] = args.lr / 100.0
        cfg.trainer.scheduler_kwargs["warmup_lr"] = args.lr / 1000.0
    if args.seed is not None:
        cfg.seed = args.seed
    if args.test_start_epoch is not None:
        cfg.trainer.test_start_epoch = args.test_start_epoch
    if args.test_every is not None:
        cfg.trainer.test_per_epoch = args.test_every
    if args.checkpoint_dir is not None:
        cfg.trainer.checkpoint = args.checkpoint_dir
    if args.resume_dir is not None:
        cfg.trainer.resume_dir = args.resume_dir
    if args.logdir_sub is not None:
        cfg.trainer.logdir_sub = args.logdir_sub
    return cfg


def print_args_summary(cfg):
    if not cfg.master:
        return
    log_msg(cfg.logger, "==> Train.py summary")
    log_msg(cfg.logger, f"==> Config: {cfg.cfg_path}")
    log_msg(cfg.logger, f"==> Mode: {cfg.mode}")
    log_msg(cfg.logger, f"==> Trainer: {cfg.trainer.name}")
    log_msg(cfg.logger, f"==> Model: {cfg.model.name}")
    log_msg(cfg.logger, f"==> Data: {cfg.data.type} at {cfg.data.root}")
    log_msg(cfg.logger, f"==> Epochs: {cfg.trainer.epoch_full}")
    log_msg(cfg.logger, f"==> Batch size: {cfg.trainer.data.batch_size}")
    log_msg(cfg.logger, f"==> LR: {cfg.optim.lr}")
    mamba_context_kwargs = cfg.model.kwargs.get("mamba_context_kwargs", {})
    if mamba_context_kwargs:
        log_msg(cfg.logger, f"==> CSSD type: {mamba_context_kwargs.get('cssd_type', 'pdar')}")


@torch.no_grad()
def evaluate_model(trainer):
    trainer.test()


def train_model(trainer):
    cfg = trainer.cfg
    trainer.reset(isTrain=True)
    if cfg.dist:
        trainer.train_loader.sampler.set_epoch(int(trainer.epoch))

    train_length = cfg.data.train_size
    train_loader = iter(trainer.train_loader)

    while trainer.epoch < trainer.epoch_full and trainer.iter < trainer.iter_full:
        trainer.scheduler_step(trainer.iter)

        # data
        t1 = get_timepc()
        trainer.iter += 1
        train_data = next(train_loader)
        trainer.set_input(train_data)
        t2 = get_timepc()
        update_log_term(trainer.log_terms.get("data_t"), t2 - t1, 1, trainer.master)

        # forward + backward
        trainer.optimize_parameters()
        t3 = get_timepc()
        update_log_term(trainer.log_terms.get("optim_t"), t3 - t2, 1, trainer.master)
        update_log_term(trainer.log_terms.get("batch_t"), t3 - t1, 1, trainer.master)

        # log
        if trainer.master and trainer.iter % cfg.logging.train_log_per == 0:
            msg = able(
                trainer.progress.get_msg(
                    trainer.iter,
                    trainer.iter_full,
                    trainer.iter / train_length,
                    trainer.iter_full / train_length,
                ),
                trainer.master,
                None,
            )
            log_msg(trainer.logger, msg)
            if trainer.writer:
                for key, val in trainer.log_terms.items():
                    trainer.writer.add_scalar(f"Train/{key}", val.val, trainer.iter)
                trainer.writer.flush()

        if trainer.iter % cfg.logging.train_reset_log_per == 0:
            trainer.reset(isTrain=True)

        # one epoch finished
        if trainer.iter % train_length == 0:
            trainer.epoch += 1
            if cfg.dist and trainer.dist_BN != "":
                distribute_bn(trainer.net, trainer.world_size, trainer.dist_BN)
            if hasattr(trainer.optim, "sync_lookahead"):
                trainer.optim.sync_lookahead()

            if (
                trainer.epoch >= cfg.trainer.test_start_epoch
                or trainer.epoch % cfg.trainer.test_per_epoch == 0
            ):
                evaluate_model(trainer)
            else:
                trainer.test_ghost()

            cfg.total_time = get_timepc() - cfg.task_start_time
            total_time = str(datetime.timedelta(seconds=int(cfg.total_time)))
            eta_time = str(
                datetime.timedelta(
                    seconds=int(cfg.total_time / trainer.epoch * (trainer.epoch_full - trainer.epoch))
                )
            )
            log_msg(
                trainer.logger,
                f"==> Total time: {total_time}\t Eta: {eta_time} \tLogged in '{cfg.logdir}'",
            )

            trainer.save_checkpoint()
            trainer.reset(isTrain=True)
            if cfg.dist:
                trainer.train_loader.sampler.set_epoch(int(trainer.epoch))
            train_loader = iter(trainer.train_loader)

    trainer._finish()


def main():
    args = parse_args()

    cfg = get_cfg(args)
    cfg = apply_cli_overrides(cfg, args)

    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)

    print_args_summary(cfg)
    trainer = get_trainer(cfg)

    if cfg.mode == "train":
        train_model(trainer)
    elif cfg.mode == "test":
        evaluate_model(trainer)
    else:
        raise NotImplementedError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()

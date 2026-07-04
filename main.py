import os
import random
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import torch
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BatchSizeFinder, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy


def set_deterministic(seed=0):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainBatchSizeFinder(BatchSizeFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        self.scale_batch_size(trainer, pl_module)


import wandb

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import RootCfg, load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.ema import EMA, EMAModelCheckpoint
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.dynamic_model import get_dynamic_model
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.state_adapter import StateAdapter


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def green(text: str) -> str:
    return f"{Fore.GREEN}{text}{Fore.RESET}"


def load_callbacks(cfg: RootCfg, with_wandb: bool, output_dir: Path):
    callbacks = []
    if with_wandb:
        callbacks.append(LearningRateMonitor("step", True))

    # Set up checkpointing.
    print(f"Checkpointing every {cfg.checkpointing.every_n_train_steps} training steps.")
    filename = "epoch_{epoch}-step_{step}-psnr_avg_{val/psnr_val_avg:.3f}-lpips_{val/lpips_val_avg:.3f}-ssim_{val/ssim_val_avg:.3f}"
    if cfg.dataset.n_step_predict > 0:
        filename = "epoch_{epoch}-step_{step}-psnr_avg_{val/psnr_val_avg:.3f}-future_psnr_{val/future_psnr_val_avg:.3f}-lpips_{val/lpips_val_avg:.3f}-ssim_{val/ssim_val_avg:.3f}"
    callbacks.append(
        ModelCheckpoint(
            filename=filename,
            auto_insert_metric_name=False,
            dirpath=output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,  # default is 1
            # monitor="psnr_val_avg",
            monitor="val/psnr_val_avg" if cfg.dataset.n_step_predict == 0 else "val/future_psnr_val_avg",
            mode="max",  # save the lastest k ckpt, can do offline test later
            verbose=True,
        )
    )
    callbacks.append(
        ModelCheckpoint(
            filename=filename,
            auto_insert_metric_name=False,
            dirpath=output_dir / "checkpoints",
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
    )

    if cfg.data_loader.dynamic_train_batch_size:
        callbacks.append(TrainBatchSizeFinder())
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = "_"
        cb.CHECKPOINT_NAME_LAST = "last-" + filename
    return callbacks


def load_logger(cfg_dict: DictConfig, output_dir: Path, prev_exp_dir: Path = None, test_path: Path = None):
    """
    In case output_dir has a wandb directory, we either resume or fork the experiment from there.
    If we fork, we also create a new directory with the current timestamp, to keep the wandb directory clean.
    """

    def get_wandb_id_for_resume(prev_exp_dir: Path):
        latest_wandb_id = None
        wandb_dir = prev_exp_dir / "wandb"
        if wandb_dir.exists():
            list_of_files = list(wandb_dir.glob("run-*"))
            if list_of_files:
                latest_wandb_id = str(max(list_of_files, key=os.path.getctime)).split("-")[-1]
        return latest_wandb_id

    with_wandb = cfg_dict.wandb.mode != "disabled"
    if with_wandb:
        # Defaults for new exp
        wandb_id = wandb.util.generate_id()
        group_id = f"{wandb_id}_group"
        resume = None
        dscp = "New exp"
        wandb_dir = output_dir

        _wandb_id = get_wandb_id_for_resume(prev_exp_dir)
        if _wandb_id is not None:
            # resume from the previous run if not in test mode and if asked for resume, otherwise fork
            resume = "must" if (cfg_dict.mode != "test" and cfg_dict.checkpointing.resume) else False
            wandb_id = _wandb_id if resume else wandb_id
            group_id = f"{wandb_id}_group" if resume else f"{_wandb_id}_group"
            dscp = "Resume" if resume else ("Fork" if cfg_dict.mode != "test" else "Test")
            wandb_dir = (
                output_dir / "wandb" / f"{test_path.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
                if cfg_dict.mode == "test"
                else wandb_dir
            )
            wandb_dir.mkdir(parents=True, exist_ok=True)

        wandb_extra_kwargs = {"id": wandb_id, "group": f"{cfg_dict.wandb.name}_{group_id}", "resume": resume}
        print(green(f"\n\n{dscp}: wandb-id: {wandb_id}, group-id: {cfg_dict.wandb.name}_{group_id}\n\n"))

        extra_name = (
            # f"{output_dir.parent.name}/{output_dir.name}"
            ""
            if cfg_dict.mode != "test"
            else f" (test_{test_path.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')})"
        )
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name}{extra_name}",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=cfg_dict.wandb.get("log_model", False),
            save_dir=wandb_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    return logger, with_wandb


def prepare_outdir(cfg_dict: DictConfig):
    """
    In case the output_dir/checkpoints exists, we either resume or fork the experiment from there.
    If we fork, we also create a new directory with the current timestamp, to keep the checkpoints directory clean.

    """
    if cfg_dict.output_dir is None:
        # to override it: [python main.py ... hydra.run.dir="/path/to/dir"]
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)

    prev_exp_dir = output_dir
    if (output_dir / "checkpoints").exists():
        if cfg_dict.mode == "test":
            print(cyan(f"Testing: "))
        elif not cfg_dict.checkpointing.resume:
            print(cyan("Forking:"))
            output_dir = output_dir / f"fork/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            print(cyan("Resuming:\n Checkpoints will be saved in the same directory."))
        print(cyan(f" Previous experiment directory: {prev_exp_dir}"))
    else:
        print(cyan("Starting a new experiment:"))
    print(cyan(f" Output directory: {output_dir}"))

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, prev_exp_dir


def get_test_output_path(output_dir, output_path: Path | None):
    output_path = str(output_path) if output_path is not None else None
    if output_path is None:
        output_path = output_dir / f"test/{time.strftime('%Y-%m-%d_%H')}"
    elif output_path.startswith("/"):
        output_path = output_path  # absolute path provided
    else:
        output_path = output_dir / output_path  # local paths are relative to output_dir
    print(cyan(f" Test output path: {output_path}"))
    return output_path


def get_checkpoint_from_prev_exp(prev_exp_dir: Path):
    latest_checkpoint = None
    checkpoint_dir = prev_exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        list_of_files = [p for p in checkpoint_dir.glob("*.ckpt") if "last" not in p.stem]
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def get_checkpoint(path: str | None, wandb_cfg: dict, prev_exp_dir: Path):
    checkpoint_path = get_checkpoint_from_prev_exp(prev_exp_dir) or update_checkpoint_path(path, wandb_cfg)
    if checkpoint_path is not None:
        print(cyan(f" Initialize from checkpoint: {checkpoint_path}.\n"))
    return checkpoint_path


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory and check for previous experiment
    cfg.output_dir, prev_exp_dir = prepare_outdir(cfg_dict)
    cfg.test.output_path = get_test_output_path(cfg.output_dir, cfg.test.output_path)

    # Prepare the checkpoint for loading.
    checkpoint_path = get_checkpoint(cfg.checkpointing.load, cfg.wandb, prev_exp_dir)

    # Set up logging with wandb.
    logger, with_wandb = load_logger(cfg_dict, cfg.output_dir, prev_exp_dir, cfg.test.output_path)

    # Set up checkpointing.
    callbacks = load_callbacks(cfg, with_wandb, cfg.output_dir)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    trainer = Trainer(
        detect_anomaly=cfg.trainer.detect_anomaly,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy=ddp_strategy if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=1 if cfg.trainer.val_check_interval < 1.0 else None,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        # enable_checkpointing=False,
    )
    trainer.logger.log_hyperparams(asdict(cfg))
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    # set_deterministic(cfg_dict.seed + trainer.global_rank)

    state_adapter = StateAdapter(cfg.model.state_adapter)
    state_info = state_adapter.get_state_info()
    encoder_info = state_adapter.get_encoder_info()
    decoder_info = state_adapter.get_decoder_info()

    encoder, encoder_vis = get_encoder(cfg.model.encoder, cfg.dataset, state_info, encoder_info)
    dynamic_model = (
        get_dynamic_model(cfg.model.dynamic_model, cfg.dataset, state_info)
        if cfg.dataset.n_step_predict > 0
        else None
    )
    decoder = get_decoder(cfg.model.decoder, cfg.dataset, decoder_info)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_vis,
        "dynamic_model": dynamic_model,
        "decoder": decoder,
        "state_adapter": state_adapter,
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
        "detect_anomaly": cfg.trainer.detect_anomaly,
    }
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = ModelWrapper.load_from_checkpoint(checkpoint_path, **model_kwargs, strict=False)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)
    # model_wrapper.load_state_dict(torch.load("checkpoints/tgs_weights.pth"), strict=False)

    # TODO: REQUIRED because we plan to remove part of the model in encoder_costvolume/depth_predictor
    # but still want to be able to load the old checkpoints for baseline comparison
    model_wrapper.strict_loading = False

    # Update for multi-step training curriculum
    if cfg.train.dyn_model_schedule.n_max_pred > 1 and cfg.dataset.n_step_predict > 0:
        cfg.dataset.n_step_predict *= cfg.train.dyn_model_schedule.n_max_pred

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    if cfg.dataset.test_validity:
        data_module.check_validity()
        return

    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=(checkpoint_path if cfg.checkpointing.resume else None),
        )
        # run test after training
        # trainer.test(model_wrapper, datamodule=data_module)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    # torch.autograd.set_detect_anomaly(True)
    # set_deterministic()

    train()

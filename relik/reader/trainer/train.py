import os
from pathlib import Path
from pprint import pprint
import hydra
import lightning
from hydra.utils import to_absolute_path, get_original_cwd
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
import omegaconf
import torch
from torch.utils.data import DataLoader

from relik.reader.data.relik_reader_data import RelikDataset
from relik.reader.lightning_modules.relik_reader_pl_module import RelikReaderPLModule
from relik.reader.pytorch_modules.optim import LayerWiseLRDecayOptimizer
from relik.reader.utils.special_symbols import get_special_symbols
from relik.reader.utils.strong_matching_eval import ELStrongMatchingCallback

def train(cfg: DictConfig) -> None:

    lightning.seed_everything(cfg.training.seed)
    # check if deterministic algorithms are available
    if "deterministic" in cfg and cfg.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # log the configuration
    pprint(OmegaConf.to_container(cfg, resolve=True))

    special_symbols = get_special_symbols(cfg.model.entities_per_forward)

    # model declaration
    model = RelikReaderPLModule(
        cfg=OmegaConf.to_container(cfg),
        transformer_model=cfg.model.model.transformer_model,
        additional_special_symbols=len(special_symbols),
        training=True,
    )

    # optimizer declaration
    opt_conf = cfg.model.optimizer
    electra_optimizer_factory = LayerWiseLRDecayOptimizer(
        lr=opt_conf.lr,
        warmup_steps=opt_conf.warmup_steps,
        total_steps=opt_conf.total_steps,
        total_reset=opt_conf.total_reset,
        no_decay_params=opt_conf.no_decay_params,
        weight_decay=opt_conf.weight_decay,
        lr_decay=opt_conf.lr_decay,
    )

    model.set_optimizer_factory(electra_optimizer_factory)

    # datasets declaration
    train_dataset: RelikDataset = hydra.utils.instantiate(
        cfg.data.train_dataset,
        dataset_path=to_absolute_path(cfg.data.train_dataset_path),
        special_symbols=special_symbols,
    )

    # update of validation dataset config with special_symbols since they
    #  are required even from the EvaluationCallback dataset_config
    with open_dict(cfg):
        cfg.data.val_dataset.special_symbols = special_symbols

    val_dataset: RelikDataset = hydra.utils.instantiate(
        cfg.data.val_dataset,
        dataset_path=to_absolute_path(cfg.data.val_dataset_path),
    )

    # callbacks declaration
    callbacks = [
        ELStrongMatchingCallback(
            to_absolute_path(cfg.data.val_dataset_path), cfg.data.val_dataset
        ),
        ModelCheckpoint(
            "model",
            filename="{epoch}-{val_core_f1:.2f}",
            monitor="val_core_f1",
            mode="max",
        ),
        LearningRateMonitor(),
    ]

    wandb_logger = WandbLogger(
        cfg.model_name, project=cfg.project_name, offline=cfg.offline
    )

    # trainer declaration
    trainer: Trainer = hydra.utils.instantiate(
        cfg.training.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    model.relik_reader_core_model._tokenizer = train_dataset.tokenizer

    # Trainer fit
    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(train_dataset, batch_size=None, num_workers=0),
        val_dataloaders=DataLoader(val_dataset, batch_size=None, num_workers=0),
    )

    # if cfg.training.save_model_path:
    experiment_path = Path(wandb_logger.experiment.dir)
    model = RelikReaderPLModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    # model.relik_reader_core_model._tokenizer = train_dataset.tokenizer
    model.relik_reader_core_model.save_pretrained(experiment_path)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()

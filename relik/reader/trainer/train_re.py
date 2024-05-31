import hydra
from pathlib import Path
import lightning
from hydra.utils import to_absolute_path
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from relik.reader.data.relik_reader_re_data import RelikREDataset
from relik.reader.lightning_modules.relik_reader_re_pl_module import (
    RelikReaderREPLModule,
)
from relik.reader.pytorch_modules.optim import (
    AdamWWithWarmupOptimizer,
    LayerWiseLRDecayOptimizer,
)
from relik.reader.utils.relation_matching_eval import REStrongMatchingCallback
from relik.reader.utils.special_symbols import get_special_symbols_re
import wandb

@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    lightning.seed_everything(cfg.training.seed)

    special_symbols = get_special_symbols_re(cfg.model.relations_per_forward)

    # datasets declaration
    train_dataset: RelikREDataset = hydra.utils.instantiate(
        cfg.data.train_dataset,
        dataset_path=to_absolute_path(cfg.data.train_dataset_path),
        special_symbols=special_symbols,
    )

    # update of validation dataset config with special_symbols since they
    #  are required even from the EvaluationCallback dataset_config
    with open_dict(cfg):
        cfg.data.val_dataset.special_symbols = special_symbols

    val_dataset: RelikREDataset = hydra.utils.instantiate(
        cfg.data.val_dataset,
        dataset_path=to_absolute_path(cfg.data.val_dataset_path),
    )

    if val_dataset.materialize_samples:
        list(val_dataset.dataset_iterator_func())
    # model declaration
    model = RelikReaderREPLModule(
        cfg=OmegaConf.to_container(cfg),
        transformer_model=cfg.model.model.transformer_model,
        additional_special_symbols=len(special_symbols),
        training=True,
    )
    model.relik_reader_re_model._tokenizer = train_dataset.tokenizer
    # optimizer declaration
    opt_conf = cfg.model.optimizer

    if "total_reset" not in opt_conf:
        optimizer_factory = AdamWWithWarmupOptimizer(
            lr=opt_conf.lr,
            warmup_steps=opt_conf.warmup_steps,
            total_steps=opt_conf.total_steps,
            no_decay_params=opt_conf.no_decay_params,
            weight_decay=opt_conf.weight_decay,
        )
    else:
        optimizer_factory = LayerWiseLRDecayOptimizer(
            lr=opt_conf.lr,
            warmup_steps=opt_conf.warmup_steps,
            total_steps=opt_conf.total_steps,
            total_reset=opt_conf.total_reset,
            no_decay_params=opt_conf.no_decay_params,
            weight_decay=opt_conf.weight_decay,
            lr_decay=opt_conf.lr_decay,
        )

    model.set_optimizer_factory(optimizer_factory)
    # callbacks declaration
    callbacks = [
        REStrongMatchingCallback(
            to_absolute_path(cfg.data.val_dataset_path), cfg.data.val_dataset
        ),
        ModelCheckpoint(
            "model",
            filename="{epoch}-{val_f1:.2f}",
            monitor="val_f1",
            mode="max",
        ),
        LearningRateMonitor(),
    ]
    wandb_logger = WandbLogger(cfg.model_name, project=cfg.project_name)

    # trainer declaration
    trainer: Trainer = hydra.utils.instantiate(
        cfg.training.trainer,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    # Trainer fit
    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(train_dataset, batch_size=None, num_workers=0),
        val_dataloaders=DataLoader(val_dataset, batch_size=None, num_workers=0),
        ckpt_path=cfg.training.ckpt_path if cfg.training.ckpt_path else None,
    )

    # Load best checkpoint
    if True:
        model = RelikReaderREPLModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        # model.relik_reader_re_model._tokenizer = train_dataset.tokenizer
        # model.relik_reader_re_model.save_pretrained(cfg.training.save_model_path)
        experiment_path = Path(wandb_logger.experiment.dir)
        model.relik_reader_re_model._tokenizer = train_dataset.tokenizer
        model.relik_reader_re_model.save_pretrained(experiment_path / "hf_model")

def main():
    train()


if __name__ == "__main__":
    main()

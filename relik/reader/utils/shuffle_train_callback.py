from lightning.pytorch.callbacks import Callback

from relik.common.log import get_logger

import os

import random

logger = get_logger()


class ShuffleTrainCallback(Callback):
    def __init__(self, shuffle_every: int = 1, data_path: str = None):
        self.shuffle_every = shuffle_every
        self.data_path = data_path

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.shuffle_every == 0:
            logger.info("Shuffling train dataset")
            # os.system(f"shuf {self.data_path} > {self.data_path}.shuf")
            # os.system(f"mv {self.data_path}.shuf {self.data_path}")
            lines = open(self.data_path).readlines()
            random.shuffle(lines)
            open(self.data_path, "w").writelines(lines)

""" Training Model"""
import numpy as np
import torch
import pytorch_lightning as pl
from gupt.data.mnist_data_module import MNISTDataModule
# from gupt.data.emnist_data_module import EMNISTDataModule
# from gupt.data.emnist_lines_data_module import EMNISTLinesDataModule
from gupt.models.ffnn import FeedForwardNN
from gupt.models.cnn import CNN
from gupt import lightning_models
# from pytorch_lightning import loggers as pl_loggers


def main():
    """Main function to train model
    """

    data = MNISTDataModule()
    data_config = data.config()
    model = FeedForwardNN(input_size=np.prod(data_config['input_dims']),
                          hidden_sizes=[128, 256],
                          output_size=len(data_config['mapping']))

    # data = EMNISTDataModule()
    # data_config = data.config()
    # model = CNN(input_dims=data_config['input_dims'],
    #             mapping=data_config['mapping'])

    # data = EMNISTLinesDataModule()

    lit_model = lightning_models.BaseLitModel(model=model, args=None)

    # Setup wandb logger
    # logger = pl_loggers.WandbLogger()
    # logger.watch(model)

    gpus = None  # CPU
    if torch.cuda.is_available():
        gpus = -1  # all available GPUs

    trainer = pl.Trainer(gpus=gpus,
                         fast_dev_run=False,
                         max_epochs=5,
                         weights_save_path='train/logs',
                         weights_summary='full',
                         auto_lr_find=True
                         #  logger=logger
                        )

    trainer.tune(
        model=lit_model,
        datamodule=data,
    )
    trainer.fit(model=lit_model, datamodule=data)
    trainer.test(model=lit_model, datamodule=data)


if __name__ == "__main__":
    main()

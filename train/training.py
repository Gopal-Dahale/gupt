""" Training Model"""
import pytorch_lightning as pl
# from gupt.data.mnist_data_module import MNISTDataModule
from gupt.data.emnist_data_module import EMNISTDataModule
# from gupt.models.ffnn import FeedForwardNN
from gupt.models.cnn import CNN
from gupt import lightning_models
# from pytorch_lightning import loggers as pl_loggers


def main():
    """Main function to train model
    """

    # data = EMNISTDataModule(None)
    # data_config = data.config()
    # model = FeedForwardNN(input_size=np.prod(data_config['input_dims']),
    #                       hidden_sizes=[1024, 128],
    #                       output_size=len(data_config['mapping']))

    data = EMNISTDataModule(None)
    data_config = data.config()
    model = CNN(input_dims=data_config['input_dims'],
                mapping=data_config['mapping'])

    lit_model = lightning_models.BaseLitModel(model=model, args=None)

    # Setup wandb logger
    # logger = pl_loggers.WandbLogger()
    # logger.watch(model)

    trainer = pl.Trainer(
        max_epochs=5,
        weights_save_path='train/logs',
        weights_summary='full',
        #  logger=logger
    )

    trainer.tune(model=lit_model, datamodule=data)
    trainer.fit(model=lit_model, datamodule=data)
    trainer.test(model=lit_model, datamodule=data)


if __name__ == "__main__":
    main()

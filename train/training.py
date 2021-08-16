from pytorch_lightning import callbacks
import numpy as np
import pytorch_lightning as pl
from gupt.data.mnist_data_module import MNISTDataModule, load_data
from gupt.models.ffnn import FeedForwardNN
from gupt import lightning_models

def main():
    data = MNISTDataModule(None)
   
    model = FeedForwardNN(input_size=np.prod(data.config()['input_dims']), hidden_sizes=[1024,256, 64], output_size= len(data.config()['mapping']))
    lit_model = lightning_models.BaseLitModel(model=model,args=None)

    trainer = pl.Trainer(max_epochs = 5,weights_save_path='train/logs',weights_summary='full')

    trainer.tune(model=lit_model,datamodule=data)
    trainer.fit(model=lit_model,datamodule=data)
    trainer.test(model=lit_model,datamodule=data)


if __name__ == "__main__":
    main()
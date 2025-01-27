import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torchvision.transforms import Resize

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class VAEExperimentModule(pl.LightningModule):
    def __init__(self, model, model_name,  
                 len_train_dataloader, 
                 learning_rate=1e-4, in_size=512, **kwargs):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.len_train_dataloader = len_train_dataloader
        self.learning_rate = float(learning_rate)
        
        if in_size != 512:
            self.resize_transform = Resize((in_size, in_size))
        else:
            self.resize_transform = None
        
    def step(self, batch):
        image = batch['image']
        
        results = self.model(image)
        
        if self.resize_transform != None:
            results[0] = self.resize_transform(results[0])
        
        train_loss = self.model.loss_function(
            *results,
            M_N=0.00025
        )
        
        return train_loss["loss"], image, results[0]
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100*self.len_train_dataloader)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def define_callbacks(project_name, every_n_train_steps=5000, save_top_k=1, **kwargs):
    callbacks = [
        ModelCheckpoint(monitor='valid_loss', mode='min',
                        every_n_train_steps=every_n_train_steps, 
                        save_top_k=save_top_k, dirpath=f'weights/{project_name}', 
                        filename='{self.model_name}-{epoch:03d}-{valid_loss:.4f}-{valid_reward:.4f}'),
        ]
    
    return callbacks
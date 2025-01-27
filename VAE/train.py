import yaml
import argparse

import torch
import pytorch_lightning as pl

from utils.config import load_config
from utils.data import define_augmentations, define_datasets_and_dataloaders
from models.model import VanillaVAE
from models.model_interface import VAEExperimentModule, define_callbacks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="specify config file")
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    in_size = config['model_params']['in_size']
    
    # Define datasets and dataloaders
    train_transform, valid_transform = define_augmentations(size=in_size)  
   
    train_dataloader, valid_dataloader, test_dataloader = define_datasets_and_dataloaders(
        **config["data_params"],
        train_transform=train_transform, 
        valid_transform=valid_transform)
    
    # Define model
    model_name = config["model_params"]["name"]
    if model_name == "VanillaVAE":
        model = VanillaVAE(**config["model_params"])
    else:
        raise("Not implemented yet!")
       
    model_interface = VAEExperimentModule(
        model=model, 
        model_name=model_name,
        len_train_dataloader=len(train_dataloader),
        in_size=in_size,
        **config["training_params"]
    )
    
    project_name = f'{model_name}_lr-{config["training_params"]["learning_rate"]}_input-{config["model_params"]["in_size"]}'
    checkpoints_callback = define_callbacks(
        project_name=project_name, **config["training_params"]
    )
    
    trainer = pl.Trainer(
        max_epochs=config["training_params"]["max_epochs"],
        gpus=config["training_params"]["gpus"],
        callbacks=checkpoints_callback, 
        enable_progress_bar=True, 
        precision=16
    )
    trainer.fit(
        model_interface, 
        train_dataloader,
        valid_dataloader
    )
    
    # Do evaluation and save results
    trainer.test(
        model_interface,
        test_dataloader,
        ckpt_path="best"
    )
    
    torch.save(
        model_interface.model.state_dict(), 
        f'weights/{project_name}/{model_name}.pt'
    )
    
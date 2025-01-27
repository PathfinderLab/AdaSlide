import os
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from models.model import VanillaVAE
from utils.config import load_config
from torchvision import datasets, transforms
from albumentations.augmentations import Normalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()
        
    # Load config file
    config = load_config(args.config)
    
    model = VanillaVAE(**config["model_params"]).to(args.device)
    # model.load_state_dict(torch.load(f"weights/VanillaVAE_lr-1e-4/VanillaVAE.pt"))
    # model.load_state_dict(torch.load(f"weights/VanillaVAE_lr-1e-4_input-224/VanillaVAE.pt"))
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(args.device)
    model.eval()
    
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    denorm = Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        always_apply=True,
        max_pixel_value=1.0
    )
    
    flist = glob.glob(args.input + "/*.jpg")
    if len(flist) == 0:
        flist = glob.glob(args.input + "/*.png")
    # if len(flist) == 0:
    #     flist = glob.glob(args.input + "/*.tif")
    
    os.makedirs(args.output, exist_ok=True)
    
    for fpath in tqdm(flist):
        if not os.path.exists(f"{args.output}/{fpath.split('/')[-1]}"):
            orig_image = Image.open(fpath).convert("RGB")
            image = transform(orig_image)

            with torch.no_grad():
                out = model.generate(image.unsqueeze(0).to(args.device))
                
            out = (denorm(image=out.detach().squeeze(0).moveaxis(0, -1).to("cpu").numpy())["image"]*255).astype(np.uint8)

            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{args.output}/{fpath.split('/')[-1]}", out)
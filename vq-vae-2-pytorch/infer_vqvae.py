import os
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from vqvae import VQVAE
from torchvision import datasets, transforms
from albumentations.augmentations import Normalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()
    
    model = VQVAE().to(args.device)
    model.load_state_dict(torch.load(f"checkpoint/vqvae_00{args.epoch}.pt"))
    model.to(args.device)
    model.eval()
    
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    
    transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            # transforms.CenterCrop(args.input_size),
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
    
    os.makedirs(args.output, exist_ok=True)
    
    for fpath in tqdm(flist):
        if not os.path.exists(f"{args.output}/{fpath.split('/')[-1]}"):
            orig_image = Image.open(fpath).convert("RGB")
           
            image = transform(orig_image)

            with torch.no_grad():
                out, _ = model(image.unsqueeze(0).to(args.device))
                
            out = (denorm(image=out.detach().squeeze(0).moveaxis(0, -1).to("cpu").numpy())["image"]*255).astype(np.uint8)

            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{args.output}/{fpath.split('/')[-1]}", out)
import os
import glob
import h5py
import openslide
from PIL import Image
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

PROJECT = "TCGA-BRCA"

dest_hr = f"/project/kimlab_tcga/AdaSlide_dataset/{PROJECT}/CLAM_prepared/HR"
dest_lr = f"/project/kimlab_tcga/AdaSlide_dataset/{PROJECT}/CLAM_prepared/LR-x4_raw"

os.makedirs(dest_hr, exist_ok=True)
os.makedirs(dest_lr, exist_ok=True)

flist = glob.glob(f"/project/kimlab_tcga/AdaSlide_dataset/{PROJECT}/CLAM_prepared/h5_files/*.h5")

def process_coord(args):
    """각 coord에 대해 고해상도 및 저해상도 이미지를 생성하는 함수"""
    file_name, coord, slide_path = args
    slide = openslide.OpenSlide(slide_path)
    
    # HR 저장 경로
    hr_path = f'{dest_hr}/{file_name}_{coord[0]}-{coord[1]}.png'
    # LR 저장 경로
    lr_path = f'{dest_lr}/{file_name}_{coord[0]}-{coord[1]}.png'

    patch = None  # Initialize patch to avoid referencing before assignment

    # 고해상도 이미지 처리
    if not os.path.exists(hr_path):
        patch = slide.read_region(tuple(coord), 1, (512, 512))
        patch.save(hr_path)
    
    # 저해상도 이미지 처리
    if not os.path.exists(lr_path):
        if patch is None:  # If HR image was already saved, load it
            patch = slide.read_region(tuple(coord), 1, (512, 512))
        lr_patch = patch.resize((128, 128))
        lr_patch.save(lr_path)

    slide.close()  # 리소스 해제

def process_file(file_name):
    """각 파일에 대해 작업"""
    slide_path = f'/project/kimlab_tcga/TCGA_raw_image_data/tcga-brca/{file_name.split("/")[-1].replace("h5", "svs")}'
    tasks = []

    with h5py.File(file_name, 'r') as f:
        coords = f['coords']
        for coord in coords:
            tasks.append((file_name.split("/")[-1].replace(".h5", ""), coord, slide_path))
    
    # 병렬 처리
    with Pool(processes=36) as pool:
        list(tqdm(pool.imap(process_coord, tasks), total=len(tasks), desc=f"Processing {file_name}"))

# 외부 루프
for file_name in tqdm(flist, desc="outer"):
    process_file(file_name)

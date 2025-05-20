import datetime
import gc

import numpy as np
import cv2
import os

import timm
import torch.cuda
from torch.ao.nn.quantized import Dropout
from torch.cpu.amp import GradScaler

import extension
from extension import print_with, str_with
from tqdm import tqdm
from data_preprocess import preprocess
from dataset import CarCLFDataset
import random
from torch.utils.data import DataLoader
from model import CLFModel
from criterion import CLFcriterion
from EarlyStopping import EarlyStopping
from torch import optim, autocast

def train():
    # print(timm.list_models(pretrained=True))
    '''
    시드 고정
    '''
    extension.set_seed(42)

    '''
    데이터셋 선언
    '''
    preproc = preprocess('open/train')
    data_list = preproc.get_data()
    train_data_list = random.sample(data_list, int(len(data_list) * 0.8))
    valid_data_list = [data for data in data_list if data not in train_data_list]


    train_dataset = CarCLFDataset(train_data_list, 256)
    valid_dataset = CarCLFDataset(valid_data_list, 256)

    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, pin_memory=True, shuffle=False, drop_last=False)

    '''
    학습전 모델 및 펼요 요소 선언
    '''

    model = CLFModel(image_size=256)
    criterion = CLFcriterion()
    es = EarlyStopping(patience = 50, mode='min', delta = 1e-6)
    device = 'cuda' if torch.cuda.is_available() else 'epu'
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    target_epochs = 1000

    scaler = GradScaler()
    model.to(device)

    '''
    GPU 메모리 클리어
    '''
    torch.cuda.empty_cache()
    gc.collect()


    '''
    모델 학습
    '''

    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_foldername = f'./models/{start_time}'
    if not os.path.exists(model_foldername):
        os.makedirs(model_foldername, exist_ok=True)
    model_list = []
    for epoch in range(1, target_epochs + 1):
        model.train()
        train_loss = 0.0

        for image, target in tqdm(train_dataloader, desc=str_with(f"Epoch {epoch}/{target_epochs}")):
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output_logit = model(image)
                # output = torch.sigmoid(output_logit) # class가 1이 아닌 경우, softmax로 변경할 것
                # output = torch.sigmoid(output_logit)
                output_logit = output_logit.view(-1, 396)
                target = target.squeeze(1)
                loss = criterion(output_logit, target)

            if not torch.isfinite(loss):
                print_with("Loss NaN or Inf")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * image.size(0)
        epoch_loss = train_loss / len(train_dataset)

        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for v_image, v_target in tqdm(valid_dataloader, desc=str_with(f"Validation")):
                v_image, v_target = v_image.to(device), v_target.to(device)
                v_output_logit = model(v_image)
                # v_output = torch.sigmoid(v_output_logit)
                v_output_logit = v_output_logit.view(-1, 396)
                v_target = v_target.squeeze(1)
                v_loss = criterion(v_output_logit, v_target)

                valid_loss += v_loss.item() * v_image.size(0)

        valid_epoch_loss = valid_loss / len(valid_dataset)
        improved, early_stop = es.step(valid_epoch_loss)
        if early_stop:
            print_with("early stopped")
            break
        if improved:
            model_filename = os.path.join(model_foldername, f'epoch{epoch:05d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            },
                model_filename
            )
            model_list = os.listdir(model_foldername)
            if len(model_list) > 5:
                model_list.sort()
                oldest_model_filename = model_list[0]
                os.remove(os.path.join(model_foldername, oldest_model_filename))
            tqdm.write(f'{str_with("new checkpoint Saved")}')
        tqdm.write(f'{str_with(f"[{epoch}/{target_epochs}],Train Loss: {epoch_loss}, Loss: {valid_epoch_loss:.8f}")}')





























if __name__ == '__main__':
    train()
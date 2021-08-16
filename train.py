import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import Model
import Dataset

def train(data_loader, model, optimizer, loss_fn, device, epoch, writer):
    
    model.train()
 
    total_loss = []
    # ミニバッチごとにループ
    for x_batch, y_batch_lt, y_batch_rt, y_batch_lb, y_batch_rb, _ in data_loader:
 
        x_batch = x_batch.to(device, dtype=torch.float)
        y_batch_lt = y_batch_lt.to(device, dtype=torch.float)
        y_batch_rt = y_batch_rt.to(device, dtype=torch.float)
        y_batch_lb = y_batch_lb.to(device, dtype=torch.float)
        y_batch_rb = y_batch_rb.to(device, dtype=torch.float)
 
        optimizer.zero_grad()
        lt, rt, lb, rb = model(x_batch)

        loss_lt = loss_fn(lt, y_batch_lt)
        loss_rt = loss_fn(rt, y_batch_rt)
        loss_lb = loss_fn(lb, y_batch_lb)
        loss_rb = loss_fn(rb, y_batch_rb)
 
        #loss = (0.25 * loss_lt) + (0.25 * loss_rt) + (0.25 * loss_lb) + (0.25 * loss_rb)
        loss = loss_lt + loss_rt + loss_lb + loss_rb
        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    for idx, loss in enumerate(total_loss) :
        writer.add_scalar('data/loss', loss, (epoch * len(total_loss)) + idx)

    return np.median(total_loss), np.max(total_loss), np.min(total_loss)


def main(epochs=25, batch_size=4, lr=0.001):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Transform を作成する。
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = Dataset.Dataset(
                './datasets/DirectionSignboard/data_list.csv', 
                './datasets/DirectionSignboard/images',
                transform
            )

    # DataLoader を作成する。
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model.Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_fn = nn.MSELoss()

    logs_writer = SummaryWriter('F:/corner_detector_pytorch/traind_model/DirectionSignboard/tb_logs')
    #dummy_input = {'source_image': torch.rand([batch_size, 3, 240, 240], device = device)}
    #logs_writer.add_graph(model, dummy_input)
    os.makedirs('traind_model/DirectionSignboard', exist_ok=True)

    print('Starting training...')    
    best_val_loss = float("inf")

    for epoch in range(epochs):
        loss_med, loss_max, loss_min = train(dataloader, model, optimizer, loss_fn, device, epoch, logs_writer)

        print(f'[{epoch+1}], {loss_max}, {loss_med}, {loss_min}')

        is_best = loss_max < best_val_loss
        best_val_loss = min(loss_max, best_val_loss)

        torch.save(
            model.to('cpu').state_dict(), 
            os.path.join(
                './traind_model/DirectionSignboard',
                f'epoch_{epoch+1}_model.pth'
            )
        )
        if is_best:
            torch.save(
                model.to('cpu').state_dict(), 
                os.path.join(
                    './traind_model/DirectionSignboard',
                    f'bset_model.pth'
                )
            )


if __name__ == '__main__':
    epochs = 200
    lr = 0.0001
    batch_size=4

    main( epochs, batch_size, lr)



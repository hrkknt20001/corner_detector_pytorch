import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw

import Model
import Dataset

def cc(size, rate):
    return (size[1]*rate[0][1], size[0]*rate[0][0])

def eval(dataloader, dataset, model, device, output_dir):
    
    model.eval()
 
    # ミニバッチごとにループ
    idx = 0
    for x_batch, y_batch_lt, y_batch_rt, y_batch_lb, y_batch_rb, idx_batch in dataloader:
  
        x_batch = x_batch.to(device, dtype=torch.float)
        y_batch_lt = y_batch_lt.to(device, dtype=torch.float)
        y_batch_rt = y_batch_rt.to(device, dtype=torch.float)
        y_batch_lb = y_batch_lb.to(device, dtype=torch.float)
        y_batch_rb = y_batch_rb.to(device, dtype=torch.float)
 
        lt, rt, lb, rb = model(x_batch)

        for idx in idx_batch:
            img = dataset.getImage(idx)
            filename = dataset.getFilename(idx)

            draw = ImageDraw.Draw(img)

            draw.line(
                ( 
                    cc(img.size, y_batch_lt), cc(img.size, y_batch_rt), 
                    cc(img.size, y_batch_rb), cc(img.size, y_batch_lb), 
                    cc(img.size, y_batch_lt) 
                ),
                fill=(255, 255, 0), width=2
            )

            draw.line(
                ( 
                    cc(img.size, lt), cc(img.size, rt), 
                    cc(img.size, rb), cc(img.size, lb), 
                    cc(img.size, lt) 
                ),
                fill=(0, 255, 0), width=2
            )

            img.save(
                os.path.join(output_dir, filename)
            )

            a = cc(img.size, lt)
            perspective_base = np.float32(
                [
                    np.array(
                        (cc(img.size, lt)[0].to('cpu').detach().numpy().copy(), cc(img.size, lt)[1].to('cpu').detach().numpy().copy())
                    ), 
                    np.array(
                        (cc(img.size, rt)[0].to('cpu').detach().numpy().copy(), cc(img.size, rt)[1].to('cpu').detach().numpy().copy())
                    ), 
                    np.array(
                        (cc(img.size, lb)[0].to('cpu').detach().numpy().copy(), cc(img.size, lb)[1].to('cpu').detach().numpy().copy())
                    ), 
                    np.array(
                        (cc(img.size, rb)[0].to('cpu').detach().numpy().copy(), cc(img.size, rb)[1].to('cpu').detach().numpy().copy())
                    )
                ]
            )
            perspective = np.float32([[0, 0], [img.size[0], 0], [0, img.size[1]], [img.size[0], img.size[1]]])
            psp_matrix = cv2.getPerspectiveTransform(perspective_base, perspective)

            plate_img = cv2.warpPerspective( cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), psp_matrix, (img.size[0], img.size[1]))
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    os.path.splitext(filename)[0] + '_Perspective.JPG'
                ), plate_img
            )

    return 

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Transform を作成する。
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = Dataset.Dataset(
                './datasets/DirectionSignboard/data_list.csv', 
                './datasets/DirectionSignboard/images',
                transform
            )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Model.Model().to(device)
    model_path = './traind_model/DirectionSignboard/bset_model.pth'
    model.load_state_dict(torch.load(model_path))

    os.makedirs('./eval_img', exist_ok=True)

    print('Starting evaling...')  

    eval(dataloader, dataset, model, device, './eval_img')

        
if __name__ == '__main__':
    main()


import pandas as pd
import torch
import os
from PIL import Image

class Dataset(torch.utils.data.Dataset):

    _data = []
    _img_folder = ''

    def __init__(self, cornner_csv, img_folder, transform = None):

        self.transform = transform
        self._img_folder = img_folder

        df = pd.read_csv(cornner_csv)
        for index, row in df.iterrows():
            #Image	LT_X	LT_Y	RT_X	RT_Y	LB_X	LB_Y	RB_X	RB_Y
            self._data.append(
                {
                    'Image': row['Image'],
                    'LT_X' : row['LT_X'],                    
                    'LT_Y' : row['LT_Y'], 
                    'RT_X' : row['RT_X'],
                    'RT_Y' : row['RT_Y'], 
                    'LB_X' : row['LB_X'],
                    'LB_Y' : row['LB_Y'], 
                    'RB_X' : row['RB_X'],
                    'RB_Y' : row['RB_Y'] 
                }
            )
            

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        data = self._data[idx]

        img = Image.open(
                os.path.join(
                    self._img_folder, data['Image']
                )
            ).convert("RGB")

        LT = torch.tensor(
                [float(data['LT_Y']) / float(img.size[0]), float(data['LT_X']) / float(img.size[1])]
            )
        RT = torch.tensor(
                [float(data['RT_Y']) / float(img.size[0]), float(data['RT_X']) / float(img.size[1])]
            )
        LB = torch.tensor(
                [float(data['LB_Y']) / float(img.size[0]), float(data['LB_X']) / float(img.size[1])]
            )
        RB = torch.tensor(
                [float(data['RB_Y']) / float(img.size[0]), float(data['RB_X']) / float(img.size[1])]
            )

        if self.transform:
            img = self.transform(img)

        return img, LT, RT, LB, RB, idx


    def getFilename(self, idx):
        return self._data[idx]['Image']


    def getImage(self, idx):
        img = Image.open(
                os.path.join(
                    self._img_folder, 
                    self._data[idx]['Image']
                )
            ).convert("RGB")
        return img


    def getImgFolder(self):
        return self._img_folder


if __name__ == '__main__':

    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Transform を作成する。
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    ds = Dataset(
            'F:/corner_detector_pytorch/datasets/DirectionSignboard/data_list.csv', 
            'F:\corner_detector_pytorch\datasets\DirectionSignboard\images',
            transform
        )

    print(ds.__len__())

    # DataLoader を作成する。
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)

    for x_batch, y_lt, y_rt, y_lb, y_rb, idx in dataloader:
        print(x_batch.size(), y_lt.size(), y_rt.size(), y_lb.size(), y_rb.size())
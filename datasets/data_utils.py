import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ListDatasets(data.Dataset):
    def __init__(self, root, fname_list):
        """
        Datasets with no labels
        :param root:
        :param fname_list:
        """
        self.root = root
        self.file_names = []
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        with open(fname_list, 'r') as t:
            for line in t.readlines():
                line = line.strip('\n')
                self.file_names.append(line)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        img = Image.open(os.path.join(self.root, file_name))
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.file_names)


def generate_file_names(root, text):
    with open(text, 'w') as t:
        for f in os.listdir(root):
            t.write(os.path.join(root, f))
            t.write('\n')
    print('File names are writen in {}'.format(text))


if __name__ == '__main__':
    root = '/home/lhq/PycharmProjects/gan.pytorch/datasets/data/test/'
    text = '/home/lhq/PycharmProjects/gan.pytorch/datasets/data/labels.txt'
    # generate_file_names(root, text)
    dataset = ListDatasets(root, text)
    print(len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False,
                                 num_workers=4)
    for i, data in enumerate(dataloader):
        print(data.size())
        break

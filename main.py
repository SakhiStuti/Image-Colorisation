import os
import argparse

import torch
from torchvision import transforms, datasets

from Train import training
from rgb2lab import RGB_to_LAB

def main(args):

    data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    RGB_to_LAB()
    ])

    #Loading the Training Set
    train_dataset = datasets.ImageFolder(root=args.trainset_path, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    #Loading the Validation Set
    val_root_path = os.path.join(args.trainset_path,'ILSVRC2012_img_train')

    val_dataset = datasets.ImageFolder(root=val_root_path,
                                               transform=data_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                                 num_workers=args.num_workers)
    print('Total images in training set: ', len(train_dataset))
    print('Total images in validation set: ', len(val_dataset))


    train = training(args)
    #Send to train
    train.train(train_loader, val_loader)


if __name__=='__main__':
    print('Enter')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Specifiy the train batch size')
    parser.add_argument('--val_batch_size', type=int, default=2, help='Specify the val batch size')
    parser.add_argument('--num_iteration', type=int, default=40000, help='Number of training iterations')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for loading batch')
    parser.add_argument('--model', type=str, default='ColorizationNet', help='specify model')
    parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained weights')
    parser.add_argument('--trainset_path', type=str, default='/scratch/datasets/ILSVRC')
    parser.add_argument('--epoch', type=int, default=1, help='Number of training epoch')
    parser.add_argument('--save_directory', type=str, default='./output', help='Set output directory')
    parser.add_argument('--resume', type=int, default=0, help='Specify the epoch')
    parser.add_argument('--lr', type=float, default=3.16e-4, help='learning_rate')
    parser.add_argument('--lr_update_iter', type=int, default=30000, help='Update lr every this iterations')

    args = parser.parse_args()
    main(args)

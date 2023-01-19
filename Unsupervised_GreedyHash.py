from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import mmcv
import cv2
from torchvision import models

torch.multiprocessing.set_sharing_strategy('file_system')


# GreedyHash(NIPS2018)
# paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
# code [GreedyHash](https://github.com/ssppp/GreedyHash)

# [GreedyHash Unsupervised] epoch:105, bit:32, dataset:cifar10-2, MAP:0.467, Best MAP: 0.467
# [GreedyHash Unsupervised] epoch:65, bit:16, dataset:cifar10-2, MAP:0.409, Best MAP: 0.410
# [GreedyHash Unsupervised] epoch:65, bit:64, dataset:cifar10-2, MAP:0.476, Best MAP: 0.476
def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 30,
                      "optim_params": {"lr": 0.0001, "weight_decay": 5e-4, "momentum": 0.9}},

        # "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 30,
        #               "optim_params": {"lr": 5e-5, "weight_decay": 5e-4}},

        "info": "[GreedyHash Unsupervised]",
        "save_path" : "/data/vhngai/DeepHash-pytorch-master/Save_Path",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": GreedyHashModelUnsupervised,
        "dataset": "coco",  # in paper GreedyHash is "Cifar-10(II)"
        "epoch": 200,
        "test_map": 5,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [64],
    }
    config = config_dataset(config)

    config["topK"] = 1000

    return config



class GreedyHashModelUnsupervised(nn.Module):
    def __init__(self, bit): #constructor
        super(GreedyHashModelUnsupervised, self).__init__() #give access to methods and properties
        self.vgg = models.vgg16(pretrained=True)#VGG is a classical convolutional neural network architectures
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters(): #Param is a library providing Parameters
            param.requires_grad = False #freezing the part of the model as no changes happen to its parameters

        self.fc_encode = nn.Linear(4096, bit)

    class Hash(torch.autograd.Function): #importing from torch.autograd import Function
        @staticmethod
        def forward(_, input):
            # _.save_for_backward(input)
            return input.sign()
        @staticmethod
        # input,  = _.saved_tensors
        # grad_output = grad_output.data
        def backward(_, grad_output):
            return grad_output


    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)

        h = self.fc_encode(x)
        b = GreedyHashModelUnsupervised.Hash.apply(h)
        if not self.training:
            return b
        else:#training the loss model
            target_b = F.cosine_similarity(b[:x.size(0) // 2], b[x.size(0) // 2:])
            target_x = F.cosine_similarity(x[:x.size(0) // 2], x[x.size(0) // 2:])
            loss1 = F.mse_loss(target_b, target_x)
            loss2 = config["alpha"] * (h.abs() - 1).pow(3).abs().mean()
            return loss1 + loss2



def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
 


    Best_mAP = 0 #start value at 0

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"])) #defining learning rate with optimizer and epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, _, ind in train_loader: 
            image = image.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            loss = net(image)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.9f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0: # Test the Model
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
        

if __name__ == "__main__":
    config = get_config()
    print(config)
    #load the checkpoint here#
    for bit in config["bit_list"]:
        train_val(config, bit)
        

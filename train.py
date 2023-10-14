import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import PDSNet
from Dataset import Train_Data
from config import opt
from torch.utils.data import DataLoader
from visdom import Visdom
from PIL import Image
from torchvision import transforms as T
from utils import PSNR
import os, time, datetime
from skimage.metrics import structural_similarity as compare_ssim

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        tensor_gpu = torch.tensor(x).cuda()
        tensor_cpu = tensor_gpu.cpu()
        img1 = tensor_cpu.numpy()
        tensor_gpu = torch.tensor(y).cuda()
        tensor_cpu = tensor_gpu.cpu()
        img2 = tensor_cpu.numpy()
        edge_output1 = cv.Canny(img1.astype(np.uint8), 50, 150)
        edge_output1[edge_output1 == 255] = 1

        edge_output2 = cv.Canny(img2.astype(np.uint8), 50, 150)
        edge_output2[edge_output2 == 255] = 1
        a3 = abs(edge_output1.astype(np.float32) - edge_output2.astype(np.float32))
        edge_loss = torch.tensor(sum(a3)).cuda(0)
        mse_loss = torch.mean(torch.pow((x - y), 2))+edge_loss
        return mse_loss

def train(use_gpu=True):

    train_data = Train_Data(opt.data_root)
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True)

    net = PDSNet()
    criterion = CustomLoss()
    if use_gpu:
        net = net.cuda()
        net = nn.DataParallel(net)
        criterion = criterion.cuda()

    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)

    torch.save(net.state_dict(), opt.save_model_path)

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    num_batch = 0
    num_show = 0

    
    vis = Visdom()
    f = open(".......")
    for epoch in range(opt.max_epoch):
        for i, (data, label) in enumerate(train_loader):
            start_time = time.time()
            data = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            num_batch += 1

            if i % 20 == 0: 
                es = time.time()- start_time
                mse_loss, psnr_now, ssim_now = test(epoch, i)
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f time:%.10f' % (
                    epoch + 1, (i + 1)*opt.batch_size, mse_loss, psnr_now,ssim_now, es))
                print('[%d, %5d] loss:%.10f PSNR:%.3f SSIM:%.3f time:%.10f' % (
                    epoch + 1, (i + 1) * opt.batch_size, mse_loss, psnr_now, ssim_now, es),file=f)
                
                num_show += 1
                x = torch.Tensor([num_show])
                y1 = torch.Tensor([mse_loss])
                y2 = torch.Tensor([psnr_now])
                vis.line(X=x, Y=y1, win='loss', update='append', opts={'title': 'loss'})
                vis.line(X=x, Y=y2, win='PSNR', update='append', opts={'title': 'PSNR'})

                torch.save(net.state_dict(), opt.save_model_path)

        if (epoch+1) % 3 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * opt.lr_decay
            print('learning rate: ', optimizer.param_groups[0]['lr'])
            print('learning rate: ', optimizer.param_groups[0]['lr'],file=f)
    print('Finished Training')
    print('Finished Training',file=f)
    f.close()

def test(epoch, i):
    net1 = PDSNet()
    net1 = net1.cuda()
    net1 = nn.DataParallel(net1)
    net1.load_state_dict(torch.load(opt.load_model_path))


    noise = Image.open('........')
    label = Image.open('........')


    img_H = noise.size[0]
    img_W = noise.size[1]

    transform = T.ToTensor()
    transform1 = T.ToPILImage()
    noise = transform(noise)
    noise = noise.resize_(1, 1, img_H, img_W)
    noise = noise.cuda()

    label = np.array(label).astype(np.float32)

    output = net1(noise) 
    output = torch.clamp(output, min=0.0, max=1.0)
    output = torch.tensor(output)
    output = output.resize(img_H, img_W).cpu()
    output_img = transform1(output)

    output = np.array(output_img)
    mse, psnr = PSNR(output, label)
    ssim = compare_ssim(output, label, data_range=255)
    return mse, psnr,ssim


if __name__ == '__main__':
    train()






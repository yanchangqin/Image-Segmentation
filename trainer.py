import torch
import  os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from net_new import *
# from Unet import *
from dataset import Get_data
from torch.utils import data
from focalloss import FocalLoss2d

save_path = r'F:\ycq\UNET\multi_obj\param'
save_param =r'para_net.pt'
if __name__ == '__main__':
    getdata = Get_data()
    dataload = data.DataLoader(getdata,batch_size=1,num_workers=4,shuffle=True)
    net = Main().cuda()
    # net = Unet()
    # net.load_state_dict(torch.load(os.path.join(save_path,save_param)))
    optimizer = optim.Adam(net.parameters())
    loss_fun = nn.MSELoss(reduction='sum')
    # loss_fun = FocalLoss2d()
    # loss_fun = torch.nn.CrossEntropyLoss()
    # loss_fun = nn.BCELoss()
    # plt.ion()
    for epoch in range(1001):
        for img_data,label_data in (dataload):
            data = img_data.cuda()
            label = label_data.cuda()
            output = net(data)
            # print(output.size())
            # label = label.expand(1,2,388,388)
            # print(label.size())
            loss = loss_fun(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('第{}批次'.format(epoch),loss.item())

            # output = output.permute(0,2,3,1)

            output =output.detach().cpu()

            output = output.squeeze(0) * 255
            # print(output.shape)
            # imgs = image.fromarray(output)
            imgs = transforms.ToPILImage(mode='L')(output)
            plt.imshow(imgs)
            plt.pause(0.1)
            # plt.pause(1)
            # plt.clf()


            if epoch % 100 ==0:
                if not os.path.exists(os.path.join(save_path, str(epoch))):
                    os.makedirs(os.path.join(save_path, str(epoch)))
                torch.save(net.state_dict(), os.path.join(save_path, str(epoch), save_param))
    # plt.ioff()



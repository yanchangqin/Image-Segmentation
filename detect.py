from net_new import *
import torch
import numpy as np
from torchvision import transforms
import PIL.Image as image
from acc import *

class Detector():
    def __init__(self,param = r'F:\ycq\UNET\param\100\para_net.pt',iscuda=True ):
        self.param =param
        self.iscuda =iscuda
        self.net = Main()
        self.net.load_state_dict(torch.load(param))
    def detect(self,img,im_label):
        img =img.resize((572,572),image.ANTIALIAS)
        img_label = im_label.resize((388, 388), image.ANTIALIAS)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        data = transform(img).unsqueeze(0)
        label = transform(img_label).unsqueeze(0)
        output = self.net(data)
        # output=output.squeeze(0)
        accuracy = get_accuracy(output,label)
        print('精度：',accuracy)
        output = output.detach().numpy()
        arr = np.reshape(output[0], [388, 388]) * 255
        img = image.fromarray(arr)

        # imgs = transforms.ToPILImage()(output).convert('L')
        img.show()
if __name__ == '__main__':
    image_file = r'F:\ycq\UNET\p_data\10.png'
    label_file = r'F:\ycq\UNET\p_label\10.png'
    num =0
    detector = Detector()
    with image.open(image_file) as img:
        with image.open(image_file) as im:
            detector.detect(img,im)

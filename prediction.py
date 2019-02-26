import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import load_cifar10
import torchvision
import glob
from PIL import Image
import PIL
from torchvision import transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


img = Image.open("horse.jpeg")
img = transforms.ToTensor()(img)
img = transforms.Normalize(mean, std)(img)
img = img.view(1, *img.shape)
img = nn.functional.interpolate(img, size=(256, 256))

layer1 = torchvision.models.resnet18(pretrained=True).conv1
model = torchvision.models.resnet18(pretrained=True)
modules = list(model.children())[:-2]

'''
init = True
for i in modules:
    if init:
        pred = i(img)
        init = False
    else:
        pred = i(pred)
'''
pred = layer1.weight.data

print(pred.shape)
#prediction = pred[0][1].view(7,7, 3)
#prediction = pred.view(7,7, 3)
#plt.imshow(prediction.detach().numpy())
#plt.show()


fig=plt.figure(figsize=(10, 10))
columns = 8
rows = 8
'''
for i in range(1, columns*rows +1):
    #img =  pred[0][i].view(7, 7)
    img =  pred[i-1].view(7, 7, 3)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.imshow(img.detach().numpy())
plt.show()
'''
#torchvision.utils.save_image(prediction, 'filters.png')

for i in range(1, columns*rows +1):
    #img =  pred[0][i].view(7, 7)
    img =  pred[i-1].numpy().transpose(1,2,0)
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.imshow(img)
plt.show()



'''
_,_,dataloader = load_cifar10(1)
for batch_it, (img, Y_batch) in enumerate(dataloader):
    break


def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range
    print("imageshape", image.shape)
    image = image / 2 + 0.5
    image = image.detach().numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1,2,0)



dataiter = iter(dataloader)
images, labels = dataiter.next()

for idx, image in enumerate(images):
    plt.imshow(convert_to_imshow_format(image))
    plt.show()

#print(pred.shape)'''

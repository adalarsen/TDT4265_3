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



img = Image.open("horse.jpeg")
img = transforms.ToTensor()(img)
img = img.view(1, *img.shape)
img = nn.functional.interpolate(img, size=(256, 256))
print(img.shape)

layer1 = torchvision.models.resnet18(pretrained=True).conv1
pred = layer1(img)

prediction = pred[0][1].view(128, 128)

plt.imshow(prediction.detach().numpy())
plt.show()

torchvision.utils.save_image(prediction, 'filters.png')


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

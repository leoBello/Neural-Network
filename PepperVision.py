import torch
import PIL.Image as Image
import numpy as np
import datetime
from torchvision import transforms


def identify(imgPath, model, item_list):

    # Treshold below which images will not be recognized
    minimalRate = 1.1

    startingTime = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The network expect input normalized in the same way
    # Height * Width * 3 RGB Channel ( Height and Width should be at least 224 )
    tr = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #
    img_tensor = tr(Image.open(imgPath)).unsqueeze_(0).to(device)
    model.to(device)

    # Return a tensor giving an index for each class of images
    # The highest index is the class identified by the network
    output = model(img_tensor)

    executionTime = datetime.datetime.now() - startingTime
    if(np.max(output[0].cpu().detach().numpy()) < minimalRate):
        print("I'm not sure but i think that it's a " + item_list[np.argmax(output[0].cpu().detach().numpy())])
    else:
        print(item_list[np.argmax(output[0].cpu().detach().numpy())]+' - Identified in {:.0f}ms '.format(executionTime.microseconds/1000) + imgPath)
    print(output[0].cpu().detach().numpy())



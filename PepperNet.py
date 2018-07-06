
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets
import os
import time
import copy


###################################### PARAMETERS ######################################

# Folder of images that Pepper should recognize
imgFolder = 'PepperRecognize'

# Name of the subfolders
nameEvaluationFolder = 'Evaluation'
nameTrainingFolder = 'Training'

###################################### LOADING DATA ######################################

# The network expect input normalized in the same way
# Height * Width * 3 RGB Channel ( Height and Width should be at least 224 )
data_transforms = {
    nameTrainingFolder: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    nameEvaluationFolder: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Creating a new dataset from the folder above ( imgFolder )
imgDataset = {x: datasets.ImageFolder(os.path.join(imgFolder, x), data_transforms[x])
                  for x in [nameTrainingFolder, nameEvaluationFolder]}

# The batch_size parameter define how many images are going to be propagated in the network
# Zero worker for windows
loader = {x: torch.utils.data.DataLoader(imgDataset[x], batch_size=20, shuffle=True, num_workers=0)
               for x in [nameTrainingFolder, nameEvaluationFolder]}

# Size of our dataset
dataSize = {x: len(imgDataset[x]) for x in [nameTrainingFolder, nameEvaluationFolder]}

class_names = imgDataset[nameTrainingFolder].classes

print("\nItems available in", imgFolder, "directory :")
for x in class_names:
    print("-",x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("\nINFO ( cuda:0 is GPU ) : Usage of", device)

# Get a batch of training data
# Inputs = batch_size Tensors
# Classes = array of indexed classes corresponding to each tensors
inputs, classes = next(iter(loader[nameTrainingFolder]))



###################################### TRAINING MODEL ######################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('----------')

        for phase in [nameTrainingFolder, nameEvaluationFolder]:
            if phase == nameTrainingFolder:
                # Set model to training mode and perform a step to change the learning rate
                scheduler.step()
                model.train()
            else:
                # Set model to evaluation mode
                model.eval()

            run_loss = 0.0
            run_corrects = 0

            for inputs, labels in loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Setting the gradients to zero
                optimizer.zero_grad()

                # If we are in training phase, we should track history to compute gradients
                with torch.set_grad_enabled(phase == nameTrainingFolder):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # If we are in training phase, we are calling backward to compute the derivative
                    # And we call optimize.step to update the value
                    if phase == nameTrainingFolder:
                        loss.backward()
                        optimizer.step()

                # Compute stats
                run_loss += loss.item() * inputs.size(0)
                run_corrects += torch.sum(predictions == labels.data)

            epoch_loss = run_loss / dataSize[phase]
            epoch_acc = run_corrects.double() / dataSize[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model if it's more accurate and update the best accuracy
            if phase == nameEvaluationFolder and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best evaluation accuracy : {:4f}'.format(best_accuracy))

    # Load the best model
    model.load_state_dict(best_model)
    return model


if __name__ == '__main__':
    # Loading the model
    # We are using resnet, there is 5 versions of this network :  18, 34, 50, 101, 152
    pepperNet = torchvision.models.resnet18(pretrained=True)
    # pepperNet = torch.load('PepperNet')

    # Freezing all layer
    for param in pepperNet.parameters():
        param.requires_grad=False

    # Adding the last layer ( requires_grad is true by default )
    # nbInput represent the numbers of inputs of the last layer
    # pepperNet.fc is the output of our network ( x or y is recognized )
    # so we need as many outputs as there are classes of images
    nbInput = pepperNet.fc.in_features
    pepperNet.fc = nn.Linear(nbInput, len(imgDataset)+1)

    # Usage of CPU or GPU
    pepperNet = pepperNet.to(device)


    # Define a loss function
    criterion = nn.CrossEntropyLoss()

    optimizer_conv = optim.SGD(pepperNet.fc.parameters(), lr=0.0001, momentum=0.9)

    # Decay the learning rate of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=6, gamma=0.1)

    model_conv = train_model(pepperNet, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=20)

    torch.save(model_conv, 'pepperNet')
    print('PepperNet saved')


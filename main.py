from PepperVision import identify
import time
import torch

list = ['bottle', 'chair', 'glass']

startLoading = time.time()
model_conv = torch.load('PepperNet_Current')
loadingTime = time.time() - startLoading
print('\nModel loaded in {:.0f}m {:.0f}s'.format(loadingTime // 60, loadingTime % 60))
model_conv.eval()

print('\nTest chairs ----------------------')
identify('TestPepperNet/testChair.jpg', model_conv, list)
identify('TestPepperNet/testChair2.jpg', model_conv, list)
identify('TestPepperNet/testChair3.jpg', model_conv, list)
identify('TestPepperNet/testChair4.jpg', model_conv, list)
identify('TestPepperNet/testChair5.jpg', model_conv, list)
print('\nTest glass  ----------------------')
identify('TestPepperNet/testGlass.jpg', model_conv, list)
identify('TestPepperNet/testGlass2.jpg', model_conv, list)
identify('TestPepperNet/testGlass3.jpg', model_conv, list)
identify('TestPepperNet/testGlass4.jpg', model_conv, list)
identify('TestPepperNet/testGlass5.jpg', model_conv, list)
print('\nTest bottles  ----------------------')
identify('TestPepperNet/testBottle.jpg', model_conv, list)
identify('TestPepperNet/testBottle2.jpg', model_conv, list)
identify('TestPepperNet/testBottle3.jpg', model_conv, list)
identify('TestPepperNet/testBottle4.jpg', model_conv, list)
identify('TestPepperNet/testBottle5.jpg', model_conv, list)
identify('TestPepperNet/testBottle6.jpg', model_conv, list)
print('\nTest nothing  ----------------------')
identify('TestPepperNet/testNothing.jpg', model_conv, list)
identify('TestPepperNet/testNothing2.jpg', model_conv, list)
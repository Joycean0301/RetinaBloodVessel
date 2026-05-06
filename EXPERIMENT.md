# EX 1:
Table: (parameters)
unet, unet++, unet3+ (resnet101)

- Number of parameters in the model Unet: **51,513,233**
- Number of parameters in the model UnetPlusPlus: **67,977,873**
- Number of parameters in the model Unet3Plus: **55,706,113**


# EX 2:
LinePlot: (training)
unet, unet ++, unet 3+ (vgg16) (diceloss) (50epoch)
- test_loss(epoch)
- test_dice(epoch)
- test_iou(epoch)

Ảnh GIF training 50 epoch


# EX 3:
Table: (evaluation)
unet, unet ++, unet 3+ (diceloss) (50epoch)
- vgg 16, resnet 101
- parameters
- dice
- iou


# EX 4:
Table: (evaluation) 
- dice, 2 loss, threshold loss
- backbone: tùy vào cái nào làm tốt hơn


# EX 5:
Table: (cross-validation)
- iou
- backbone
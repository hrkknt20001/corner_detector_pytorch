from torchvision import models

model = models.vgg16(pretrained=True)
print(model)
print(model.features)


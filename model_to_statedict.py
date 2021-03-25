import torch

model = torch.load("weights/resnet50_ibn_a_model_60.pth")
model.eval()
torch.save(model.state_dict(), "weights/state_dict.pth")
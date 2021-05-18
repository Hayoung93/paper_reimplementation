# paper_reimplementation

I followed-up `berniwal/swin-transformer-pytorch` for swin_transformer codes


## Randaugment: Practical automated data augmentation with a reduced search space
- Features
Applies randomly selected augmentations with fixed magnitude to each mini-batch
Supports PIL image as input under torchvision version `0.8.0`
Supports PIL image and torch tensor under torchvision version `0.9.0`

- Usage
```
import torch
import torchvision
import torchvision.transform as tt
from randaugment import RandAugment

RA = RandAugment(3, 3)  # N, M
temp = tt.functional.to_pil_image(torch.randn(3, 5, 5))
result1 = RA(temp)
result2 = RA(temp)

dataset = torchvision.datasets.CIFAR10(..., transform=tt.Compose([RA]), ...)
```

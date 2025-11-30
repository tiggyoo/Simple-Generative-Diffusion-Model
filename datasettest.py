import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# ✅ Simple transform (just convert image to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Custom dataset loader for your own "cars/" folder
class CarsDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

# ✅ Replace StanfordCars with your dataset
data = CarsDataset("cars", transform=transform)

# ✅ Fix and use the image display function
def show_images(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15, 15))
    for i in range(min(num_samples, len(dataset))):
        img, _ = dataset[i]
        img = img.permute(1, 2, 0)  # CHW to HWC
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# ✅ Show your images
show_images(data)
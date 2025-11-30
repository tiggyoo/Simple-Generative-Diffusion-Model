Diffusion Model Image Generator (PyTorch)

This project is a **DDPM-style generative diffusion model** implemented in **PyTorch**.  
Instead of generating text, this model learns to generate **new images** by reversing a gradual noising process.

The included code trains a U-Net–based denoising network on a folder of images and then uses it to produce new samples starting from pure noise.

---

Features
- Fully custom PyTorch diffusion pipeline  
- Forward (noising) and reverse (denoising) processes  
- Configurable noise schedule  
- Image dataset loader  
- Simple visualization utilities
- GPU acceleration (MPS for Mac, CUDA if available)

- 
---IMPORTANT---   Key Hyperparameters ---
IMG_SIZE = 64 
BATCH_SIZE = 8
T = 100  # number of diffusion steps
device = torch.device("mps" if available else "cpu")
  Use cuda if availabe

Feel free to tune:
	•	IMG_SIZE (try 128 for better results)
	•	T (increase for smoother training)
	•	BATCH_SIZE
	•	noise schedule



---
Make sure you have Python 3.9+ and install the dependencies:
  pip install torch torchvision pillow matplotlib
---

The script includes a helper function:
  show_tensor_image(image_tensor)

It automatically denormalizes the image and displays it with Matplotlib.


 Dataset Setup

Replace the included **`cars/`** folder with **your own images**.

Your dataset directory must look like this:
your_dataset/
image1.jpg
image2.png
image3.jpeg


Then update the training code to point to this folder:

  dataset = CarsDataset(folder="your_dataset", transform=data_transforms)



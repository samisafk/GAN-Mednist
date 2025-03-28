# GAN-Based Medical Image Generation and Evaluation

## Overview
This blog provides a summary of our project on medical image generation using GANs. We implemented LS-GAN, WGAN, and WGAN-GP on the MedMNIST dataset and evaluated their performance using Inception Score (IS) and Fréchet Inception Distance (FID).

## Project Details
 **Dataset**: ChestMNIST (28x28 grayscale medical images)
 
 **GAN Variants**:
  - LS-GAN (Least Squares GAN)
  - WGAN (Wasserstein GAN)
  - WGAN-GP (Wasserstein GAN with Gradient Penalty)
    
 **Evaluation Metrics**:
  - **Inception Score (IS)**: Measures diversity and quality of generated images.
  - **Fréchet Inception Distance (FID)**: Measures similarity between real and generated images.
    
 **Technologies Used**:
  - Python
  - PyTorch
  - TorchMetrics (for IS and FID computation)
  - TensorBoard (for visualization)
  - Flask (for potential deployment)

## Model Training
Each GAN model was trained for at least 50 epochs with the following hyperparameters:
- **Batch size**: 64
- **Learning rate**: 0.0002
- **Optimizer**: Adam (with weight clipping for WGAN)
- **Gradient penalty** (for WGAN-GP)

## Evaluation Results
The models were evaluated based on IS and FID scores:

| Model     | Inception Score (IS) | FID Score |
|-----------|----------------------|------------|
| LS-GAN    | 1.71                 | 344.27     |
| WGAN      | 2.02                 | 337.78     |
| WGAN-GP   | 1.89                 | 339.99     |

### Sample Generated Images
#### LS-GAN
![LS-GAN Output](https://github.com/samisafk/GAN-Mednist/blob/main/generated/LS-GAN_epoch_49.png)

#### WGAN
![WGAN Output](https://github.com/samisafk/GAN-Mednist/blob/main/generated/WGAN_epoch_49.png)

#### WGAN-GP
![WGAN-GP Output](https://github.com/samisafk/GAN-Mednist/blob/main/generated/WGAN-GP_epoch_49.png)

## Key Takeaways
- **WGAN** achieved the highest Inception Score (IS), indicating better diversity.
- **WGAN** also had the lowest FID, suggesting it produced images closer to real data.
- **Gradient Penalty** in WGAN-GP improved stability but slightly worsened IS compared to WGAN.
- **LS-GAN** showed lower performance, likely due to its different loss function dynamics.

## Future Improvements
- Increasing dataset size and training epochs for better results.
- Experimenting with advanced architectures like StyleGAN or BigGAN.
- Implementing hybrid loss functions for improved generation quality.

## Setup and Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- torchmetrics
- medmnist
- tqdm
- tensorboard

### Installation
Clone the repository and install dependencies:
```bash
 git clone https://github.com/samisafk/GAN-Mednist.git
 cd GAN-Mednist
 pip install -r requirements.txt
```

## Training the GANs
Run the training script to train all three models:
```bash
python train.py
```
The trained models will be saved in the `models/` directory.

## Evaluating the Models
To compute IS and FID scores:
```bash
python evaluate.py
```

## GitHub Repository
For full code and details, visit: [GAN-Mednist Repository](https://github.com/samisafk/GAN-Mednist)


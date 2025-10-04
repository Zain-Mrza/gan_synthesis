# Brain Tumor Image Synthesis (BraTS 2020)

This project uses the **[BraTS 2020 dataset]**(https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) to generate synthetic brain tumor MRI images.  
It combines **Variational Autoencoders (VAEs)** and **Conditional GANs (cGANs)** to produce realistic tumor-containing scans for research and potential data augmentation.

---

## 📂 Workflow

1. **Data Preprocessing**
   - Cropped MRI slices from the BraTS 2020 dataset.
   - Applied transformations and normalization to prepare brain scans and segmentation masks.

2. **Segmentation Mask Generation (VAE)**
   - Trained a **Variational Autoencoder (VAE)** on tumor segmentation masks.
   - The VAE learns the distribution of tumor shapes and generates **synthetic masks** that capture structural variability.

3. **Tumor Image Synthesis (cGAN)**
   - Implemented a **Conditional GAN (cGAN)**:
     - **Condition** → synthetic segmentation mask from the VAE  
     - **Input** → original brain scan (without tumors)  
     - **Output** → realistic tumor-containing MRI slice

---

## 🎯 Project Goals
- Explore hybrid VAE–GAN pipelines for medical image synthesis.
- Generate realistic tumor masks and images to **augment medical datasets**.

---

## 📊 Pipeline Overview

```text
   BraTS 2020 MRI ----> Preprocessing ----> VAE ----> Fake Segmentation Mask
                                                                |
                                                                v
                                           Original MRI ----> cGAN ----> Synthetic Tumor MRI



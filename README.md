
## DeepSwinLite

Official repository for DeepSwinLite: A Swin Transformer-Based Light Deep Learning Model for Building Extraction from VHR Imagery (submitted to Remote Sensing, 2025).

## 🔍 Overview

The repository will include:
- Source code for model training and evaluation
- Refined Dataset download instructions


## 📁 Dataset

We provide the **refined Massachusetts Building Dataset** (including **train/validation/test splits**) as a ready-to-download package on Kaggle.

- Original dataset on Kaggle: [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)  
- **Refined dataset on Kaggle (ready to download):** [Refined Massachusetts Building Data](https://www.kaggle.com/datasets/yilmazelifozlem/refined-massachusetts-building-data/data)  
- License: **CC BY 4.0** (re-distribution allowed with attribution).

⚠️ Note: This refined version includes corrected/standardized labels, removal of low-quality tiles, and curated train/val/test splits.



## ⚙️ Codebase

This repository provides the model architecture only:
- deepswinlite/model.py → implementation of the DeepSwinLite network, including backbone and modules (MLFP, MSFA, AuxHead).

## 📑 Citation

If you use this repository or dataset refinements, please cite our paper and this repository:

@article{Yilmaz2025DeepSwinLite,  
title   = {DeepSwinLite: A Swin Transformer-Based Light Deep Learning Model for Building Extraction Using VHR Aerial Imagery},  
author  = {Yılmaz, Elif Ozlem and Kavzoglu, Taskin},  
journal = {Remote Sensing},  
volume  = {17},  
number  = {18},  
pages   = {3146},  
year    = {2025},  
doi     = {10.3390/rs17183146}  
}



## 📬 Contact

For inquiries, please contact:  
✉️ eoyilmaz@gtu.edu.tr

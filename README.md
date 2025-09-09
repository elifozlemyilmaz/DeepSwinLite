
## DeepSwinLite

Official repository for DeepSwinLite: A Swin Transformer-Based Light Deep Learning Model for Building Extraction from VHR Imagery (submitted to Remote Sensing, 2025).

## 🔍 Overview

The repository will include:
- Source code for model training and evaluation
- Refined Dataset download instructions

## 📁 Dataset

We share refined train/val/test splits and preprocessing scripts of the Massachusetts Building Dataset.
- Original dataset on Kaggle: [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)
- Refined dataset on Kaggle: [Refined Massachusetts Building Data](https://www.kaggle.com/datasets/yilmazelifozlem/refined-massachusetts-building-data/data)
License: CC BY 4.0 (re-distribution allowed with attribution).

⚠️ Note: The full dataset is not redistributed here. Please download from the original source and use our scripts to generate the refined version.

## ⚙️ Codebase

This repository provides the model architecture only:
- deepswinlite/model.py → implementation of the DeepSwinLite network, including backbone and modules (MLFP, MSFA, AuxHead).

## 📑 Citation

If you use this repository or dataset refinements, please cite our paper and this repository:

@article{Yilmaz2025DeepSwinLite,
  title   = {DeepSwinLite: A Swin Transformer-Based Light Deep Learning Model for Building Extraction from VHR Imagery},
  author  = {Yılmaz, Elif Özlem and Kavzoğlu, Taşkın},
  journal = {Remote Sensing},
  year    = {2025}
}


## 📬 Contact

For inquiries, please contact:  
✉️ eoyilmaz@gtu.edu.tr

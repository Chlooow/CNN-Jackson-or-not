# Jackson or Not Jackson?  

## Project Overview

This project was developed as part of the **Master's program in Computer Science and Big Data**, option *Recherche Big Data*, in the course *Recherche en Big Data* with M. Touati.  

The goal is to build a **Convolutional Neural Network (CNN)** to analyze and classify images of **Jackson Wang (Wang Jia Er)**, a Hong Kong rapper, singer, and performer, member of the K-pop group GOT7. Jackson Wang has a successful international solo career and is also active in fashion and entertainment. This project aims to help identify him correctly, as people sometimes mistake him for other idols.  

---

## Architecture

The CNN architecture used in this project is as follows:

### Base Model (`MyCNN`)
- **Input:** (3, 128, 128) RGB image  
- **Conv Layer 1:** 3 → 32 filters, 3x3, ReLU, MaxPool2d(2x2) → 64x64  
- **Conv Layer 2:** 32 → 64 filters, 3x3, ReLU, MaxPool2d(2x2) → 32x32  
- **Conv Layer 3:** 64 → 128 filters, 3x3, ReLU, MaxPool2d(2x2) → 16x16  
- **Classifier:** Flatten → Linear(128*16*16 → 256) → ReLU → Dropout → Linear(256 → 2)

### Optimized Model (`MyCNNOptimized`)
- **Input:** (3, 128, 128) RGB image  
- **Conv Layer 1:** 3 → 32 filters, 3x3, ReLU, MaxPool2d(2x2) → 64x64  
- **Conv Layer 2:** 32 → 64 filters, 3x3, ReLU, MaxPool2d(2x2) → 32x32  
- **Classifier:** Flatten → Linear(64*32*32 → 256) → ReLU → Dropout → Linear(256 → 2)

The optimized model is lighter and faster to train while keeping good accuracy.  

---

## Dataset

The dataset contains images of Jackson Wang and "other" images (labeled `divers`), split into:

```text
dataset/
├─ train/
│ ├─ jackson/
│ └─ divers/
├─ test/
│ ├─ jackson/
│ └─ divers/
```
- Training set: 160 Jackson, 160 divers  
- Test set: 40 Jackson, 40 divers  

---

## Installation

1. Clone this repository:  
```bash
git clone <repo-url>
cd CNN-Jackson-or-not
```
```bash
pip install -r requirements.txt
```
---
## Usage
**Training the model**
Run the main training script:
```python
python or python3 src/main.py
```
This will train the CNN model on the dataset and save the trained weights as `model.pth`

---
## Streamlit Application
Launch the interactive app to classify images:
```bash
streamlit run app.py
```
- Upload an image of Jackson or someone else.
- Click Analyse to see the prediction and confidence.
- Provide feedback if the prediction is correct or not.

## Author
Chloé Makoundou / Cholorsplash
This project was created as part of a Master's program in Computer Science and Big Data, option Recherche Big Data.

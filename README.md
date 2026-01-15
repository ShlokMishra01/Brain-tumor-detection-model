#  NeuroDetect: AI-Powered Clinical Decision Support System (CDSS)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deep Learning](https://img.shields.io/badge/Domain-Medical%20Imaging-red)
![Status](https://img.shields.io/badge/Status-Prototype-green)

##  Project Overview
**NeuroDetect** is an end-to-end Deep Learning framework designed to assist radiologists in the early detection of brain tumors from MRI scans. Unlike standard classification models, this project functions as a **Clinical Decision Support System (CDSS)**, providing not just a binary diagnosis but a comprehensive **automated medical report** with risk stratification, confidence metrics, and urgency grading.

The system leverages a **Custom Deep Convolutional Neural Network (CNN)** optimized for medical image feature extraction, achieving high sensitivity to minimize false negatives—a critical requirement in healthcare AI.

##  Key Features
* **Deep CNN Architecture:** A custom-built architecture utilizing `Conv2D`, `BatchNormalization`, and `GlobalAveragePooling` to extract intricate tumor textures and boundaries.
* **Automated Diagnostic Reporting:** Generates professional, IBM Watson-style text reports including:
    * **Risk Stratification:** (Critical / High / Moderate / Low)
    * **Urgency Tiers:** Color-coded triage recommendations.
    * **Audit Trails:** Session IDs and timestamps for medical record keeping.
* **Visual Dashboard:** Displays the MRI scan side-by-side with a real-time confidence bar chart for interpretable AI.
* **Robust Data Pipeline:** Implements advanced data augmentation (Rotation, Zoom, Shear) to handle data scarcity and prevent overfitting.
* **Scientific Validity:** Ensures reproducibility via fixed random seeds (`Seed: 42`) and rigorous Train/Val/Test splitting (70/15/15).

##  Tech Stack
* **Core Framework:** TensorFlow, Keras
* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`)
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

##  Methodology & Architecture
1.  **Data Ingestion:** Loads the *Brain MRI Images for Brain Tumor Detection* dataset (253 samples).
2.  **Preprocessing:** Resizing to `(224, 224)`, Normalization (`1./255`), and Augmentation.
3.  **Model Training:**
    * **Input Layer:** `224x224x3` RGB Images.
    * **Feature Extraction:** 4 Blocks of Conv2D + BatchNorm + MaxPool + Dropout.
    * **Classifier:** GlobalAveragePooling -> Dense(256) -> Sigmoid Output.
4.  **Optimization:** Uses `Adam` optimizer with `EarlyStopping` and `ReduceLROnPlateau` callbacks to converge at the global minimum.

##  Output Demo
*(The system analyzing an unseen MRI scan and generating a risk profile)*

> **[INSERT YOUR SCREENSHOT HERE]**
> *Replace this text with the screenshot you took from Google Colab showing the Image + Bar Graph + Text Report.*

## 📈 Performance Metrics
* **Recall (Sensitivity):** High focus on maximizing recall to ensure no tumors are missed.
* **Precision:** Optimized to reduce false alarms.
* **Loss Function:** Binary Cross-Entropy.

*(Note: Actual metrics vary slightly per run due to stochastic training, targeting ~90%+ validation accuracy)*

##  Dataset
The model was trained on the **Brain MRI Images for Brain Tumor Detection** dataset hosted on Kaggle.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* **Classes:** `Yes` (Tumor Detected), `No` (Healthy).

##  How to Run
This project is optimized for Google Colab but can run locally.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/NeuroDetect.git](https://github.com/YOUR_USERNAME/NeuroDetect.git)
    cd NeuroDetect
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Data:**
    Download the dataset from Kaggle and place the `archive.zip` file in the root directory.
4.  **Run the Notebook:**
    Open `Brain_Tumor_Detection.ipynb` and run all cells.

##  Future Scope 
* **Grad-CAM Integration:** To visualize "heatmaps" showing exactly which part of the brain the AI is looking at.
* **Web Deployment:** Deploying the model using Streamlit or Flask for a user-friendly doctor interface.
* **Multiclass Classification:** Extending the model to classify tumor types (Glioma, Meningioma, Pituitary).

---
### ⚠️ Disclaimer
*This project is a prototype developed for educational and research purposes. It is NOT a certified medical device and should not be used for actual clinical diagnosis without regulatory approval.*

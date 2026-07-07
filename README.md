# MarineBio-Net: Automated Bioacoustic Classification for Marine Mammals 🌊🐋

An end-to-end Machine Learning pipeline designed to automate the monitoring and identification of marine mammal species using deep learning techniques on bioacoustic data. By processing Mel-frequency cepstral coefficients (MFCC) spectrograms, this system successfully classifies 33 different species.

## Project Overview
- **Objective:** Automated classification of marine mammal audio recordings for passive acoustic monitoring (PAM).
- **Data Source:** Watkins Marine Mammal Sound Database (extracted via Hugging Face).
- **Dataset Size:** 1,086 audio files converted into 64x64 grayscale MFCC spectrograms.
- **Model Architecture:** Custom 3-layer Convolutional Neural Network (CNN) optimized for audio-visual feature mapping.
- **Performance:** Reached a peak classification accuracy/F1-score of ~85% across 33 highly imbalanced and acoustically complex classes.

## Pipeline Architecture
1. **Data Acquisition (`download_data.py`):** Automatically connects to Hugging Face, streams raw audio bytes from the Watkins database, and organizes them into species-specific directories while bypassing heavy external dependencies.
2. **Feature Engineering (`create_spectrogram.py`):** Loads `.wav` files via Librosa, extracts 13-channel MFCC features, maps them to time-frequency grids, and saves them as standardized `.png` spectrogram images. Includes a caching check to prevent re-processing existing files.
3. **Model Training & Evaluation (`03_train_model.ipynb`):**
   - Normalizes and splits data into Train (80%) and Test (20%) sets.
   - Fits a custom sequential CNN utilizing strategic Dropout (0.5) to combat overfitting.
   - Employs `EarlyStopping` to freeze training weights at the absolute peak of validation accuracy.
   - Quantifies results using a normalized Confusion Matrix and a comprehensive F1-score species breakdown.

## Repository Structure
    ├── download_data.py          # Dataset download and extraction script
    ├── create_spectrogram.py     # MFCC feature extraction & spectrogram generator
    ├── 03_train_model.ipynb      # CNN architecture, training loop, and evaluation
    └── README.md                 # Project documentation

## Technologies & Libraries Used
- **Core Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Audio Processing:** Librosa
- **Computer Vision:** OpenCV (cv2)
- **Data Science & Analytics:** Scikit-learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## Key Engineering Insights
- **Custom vs. Pre-trained:** Standard computer vision architectures (like MobileNet or ResNet) underperformed due to the abstract structural nature of spectrograms compared to natural images. A custom-tuned CNN built specifically for structural audio patterns yielded significantly higher generalization.
- **Combating Data Imbalance:** Evaluation was strictly driven by **F1-Score** rather than simple accuracy to ensure the model was fairly penalized for minority classes, establishing a reliable baseline for real-world deployment.

---
*Developed as an open-source contribution to Marine Bioacoustic Research.*

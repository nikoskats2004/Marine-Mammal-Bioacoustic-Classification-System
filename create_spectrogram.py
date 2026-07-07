import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Path definition 
base_path = r"C:\Users\CHANGE ME\Desktop\Vaquita_Acoustic_Monitoring_AI"
data_path = os.path.join(base_path, "data")

print("Starting MFCC Spectrogram generation for AI training...")

# Scan all subfolders
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            # Avoid re-processing if the image already exists
            output_name = file.replace('.wav', '_mfcc.png')
            if os.path.exists(os.path.join(root, output_name)):
                continue
            
            try:
                # Load audio using librosa
                y, sr = librosa.load(file_path, sr=None)
                
                if len(y) == 0:
                    continue
                
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Create the plot
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                plt.title(f'MFCC: {file}')
                
                # Save as .png in the same folder
                plt.savefig(os.path.join(root, output_name), bbox_inches='tight')
                plt.close() 
                
            except Exception as e:
                print(f"[ANALYSIS ERROR] Problem with {file} (Skipping): {e}")

print("\n[SUCCESS] MFCC spectrograms have been successfully generated!")
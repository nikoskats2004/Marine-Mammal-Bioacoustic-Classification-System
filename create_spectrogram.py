import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Η διαδρομή σου
base_path = r"C:\Users\CHANGE ME\Desktop\Vaquita_Acoustic_Monitoring_AI"
data_path = os.path.join(base_path, "data")

print("Εκκίνηση δημιουργίας χαρακτηριστικών AI (MFCC Spectrograms)...")

# Σάρωση όλων των υποφακέλων
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            try:
                # Φόρτωση ήχου μέσω της αυτόνομης librosa
                y, sr = librosa.load(file_path, sr=None)
                
                if len(y) == 0:
                    continue
                
                # Εξαγωγή MFCCs
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Δημιουργία Γραφήματος
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                plt.title(f'MFCC: {file}')
                
                # Αποθήκευση εικόνας .png στον ίδιο φάκελο
                output_name = file.replace('.wav', '_mfcc.png')
                plt.savefig(os.path.join(root, output_name), bbox_inches='tight')
                plt.close() 
                
            except Exception as e:
                print(f"[ΣΦΑΛΜΑ ΑΝΑΛΥΣΗΣ] Πρόβλημα στο {file} (Παραβλέπεται): {e}")

print("\n[ΕΠΙΤΥΧΙΑ] Τα γραφήματα MFCC δημιουργήθηκαν επιτυχώς!")
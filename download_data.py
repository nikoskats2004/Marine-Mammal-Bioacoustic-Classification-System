import os
import datasets

# Η διαδρομή σου (ΠΡΟΣΟΧΗ: Άλλαξε το "CHANGE ME" στο δικό σου όνομα χρήστη Windows)
base_path = r"C:\Users\CHANGE ME\Desktop\Vaquita_Acoustic_Monitoring_AI"
data_path = os.path.join(base_path, "data")
os.makedirs(data_path, exist_ok=True)

print("Σύνδεση στο Hugging Face για λήψη του Watkins Database...")

try:
    # Φορτώνουμε το dataset
    raw_dataset = datasets.load_dataset("confit/wmms-parquet", split="train")
    
    # Απενεργοποιούμε ρητά την αποκωδικοποίηση ήχου για να μην ζητάει Torch/FFmpeg
    dataset = raw_dataset.cast_column("audio", datasets.features.Audio(decode=False))
    
    print("Τα δεδομένα φορτώθηκαν! Εκκίνηση ασφαλούς εξαγωγής των αρχείων .wav...")
    
    for i, item in enumerate(dataset):
        species_name = item['species'] 
        
        # Παίρνουμε τα ωμά bytes του ήχου
        audio_bytes = item['audio']['bytes']
        
        # Δημιουργία φακέλου για το κάθε είδος ζώου
        species_folder = os.path.join(data_path, species_name)
        os.makedirs(species_folder, exist_ok=True)
        
        # Ορισμός της τελικής διαδρομής του αρχείου .wav
        file_path = os.path.join(species_folder, f"{species_name}_{i}.wav")
        
        # Αποθήκευση των bytes ως κανονικό αρχείο ήχου .wav στον δίσκο
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Ενημέρωση προόδου ανά 200 αρχεία
        if (i + 1) % 200 == 0:
            print(f"Αποθηκεύτηκαν επιτυχώς {i + 1} αρχεία ήχου...")

    print(f"\n[ΕΠΙΤΥΧΙΑ] Όλοι οι ήχοι εξαχθήκαν και οργανώθηκαν στον φάκελο: {data_path}")

except Exception as e:
    print(f"\n[ΣΦΑΛΜΑ ΛΗΨΗΣ]: {e}")
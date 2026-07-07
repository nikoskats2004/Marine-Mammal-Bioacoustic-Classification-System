import os
import datasets

# Path definition 
base_path = r"C:\Users\CHANGE ME\Desktop\Vaquita_Acoustic_Monitoring_AI"
data_path = os.path.join(base_path, "data")
os.makedirs(data_path, exist_ok=True)

print("Connecting to Hugging Face to download the Watkins Database...")

try:
    # Load the dataset
    raw_dataset = datasets.load_dataset("confit/wmms-parquet", split="train")
    
    # Explicitly disable audio decoding to avoid external dependencies like Torch/FFmpeg
    dataset = raw_dataset.cast_column("audio", datasets.features.Audio(decode=False))
    
    print("Dataset loaded! Starting safe extraction of .wav files...")
    
    for i, item in enumerate(dataset):
        species_name = item['species'] 
        
        # Get raw audio bytes
        audio_bytes = item['audio']['bytes']
        
        # Create a folder for each species
        species_folder = os.path.join(data_path, species_name)
        os.makedirs(species_folder, exist_ok=True)
        
        # Define the final .wav file path
        file_path = os.path.join(species_folder, f"{species_name}_{i}.wav")
        
        # Save raw bytes as a .wav file to the disk
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        # Progress update every 200 files
        if (i + 1) % 200 == 0:
            print(f"Successfully saved {i + 1} audio files...")

    print(f"\n[SUCCESS] All audio files extracted and organized in: {data_path}")

except Exception as e:
    print(f"\n[DOWNLOAD ERROR]: {e}")
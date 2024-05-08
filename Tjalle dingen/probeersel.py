import numpy as np
import matplotlib.pyplot as plt


import numpy as np

# Laden van de data uit de dictionaries
acc = np.load("Data_tests/ACC_signal.npy", allow_pickle=True).item()
rot = np.load("Data_tests/Gyro_signal.npy", allow_pickle=True).item()

# Functie om data te normaliseren tussen -1 en 1
def normalize_data(data):
    # Zoek de maximale en minimale waarden in de data
    max_val = max(data.values())
    min_val = min(data.values())
    
    # Bereken de schaal factor om de data tussen -1 en 1 te normaliseren
    scale_factor = 2 / (max_val - min_val)
    
    # Normaliseer de data
    normalized_data = {key: ((value - min_val) * scale_factor - 1) for key, value in data.items()}
    
    return normalized_data

# Normaliseer de acceleratie data
normalized_acc = normalize_data(acc)

# Normaliseer de rotatie data
normalized_rot = normalize_data(rot)

# Laat de genormaliseerde data zien
print("Genormaliseerde acceleratie data:")
print(normalized_acc)
print("\nGenormaliseerde rotatie data:")
print(normalized_rot)

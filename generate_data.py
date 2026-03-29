import pandas as pd
import numpy as np

def generate_realistic_skin_tone():
    """Generate realistic skin tones with proper undertone characteristics"""
    
    # Random skin depth (20-80, where higher = darker skin)
    skin_depth = np.random.uniform(20, 80)
    
    # Select undertone type
    undertone_bias = np.random.choice(['warm', 'cool', 'neutral'], p=[0.35, 0.35, 0.3])
    
    if undertone_bias == 'warm':
        # Warm: More red/yellow, less blue
        # Base values increase with skin depth
        r = int(120 + skin_depth * 1.2 + np.random.normal(0, 8))
        g = int(80 + skin_depth * 1.0 + np.random.normal(0, 8))
        b = int(60 + skin_depth * 0.8 + np.random.normal(0, 8))
        label = "Warm"
        
    elif undertone_bias == 'cool':
        # Cool: More blue, less red
        b = int(120 + skin_depth * 1.2 + np.random.normal(0, 8))
        g = int(80 + skin_depth * 1.0 + np.random.normal(0, 8))
        r = int(60 + skin_depth * 0.8 + np.random.normal(0, 8))
        label = "Cool"
        
    else:  # neutral
        # Neutral: Balanced RGB values
        base = int(80 + skin_depth * 1.0 + np.random.normal(0, 10))
        r = base + np.random.randint(-12, 12)
        g = base + np.random.randint(-12, 12)
        b = base + np.random.randint(-12, 12)
        label = "Neutral"
    
    # Clamp values to valid RGB range (0-255)
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    
    return [r, g, b], label

def generate_synthetic_data(n_samples=1500):
    """Generate comprehensive dataset"""
    data = []
    
    for _ in range(n_samples):
        rgb, label = generate_realistic_skin_tone()
        data.append(rgb + [label])
    
    return pd.DataFrame(data, columns=["R", "G", "B", "Label"])

# Generate dataset
print("Generating skin undertone dataset...")
df = generate_synthetic_data(1500)
df.to_csv("skin_undertone_dataset.csv", index=False)

print(f"Dataset generated successfully! Shape: {df.shape}")
print(f"Class distribution:\n{df['Label'].value_counts()}")
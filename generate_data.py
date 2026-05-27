import pandas as pd
import numpy as np

def generate_realistic_skin_tone():
    """
    Generate realistic skin tones with proper undertone characteristics.
    
    KEY FIX: All three undertones (Warm, Cool, Neutral) now produce actual
    skin-colored RGB values. Previously, 'Cool' was generating blue/purple
    values (R=86, B=172) which are NOT real skin colors, making the model
    unable to classify real human faces.
    
    Real skin undertone differences are SUBTLE:
    - Warm: red-yellow dominant, higher R, lower B
    - Cool: pink-rosy cast, R slightly higher but B is elevated too
    - Neutral: balanced, no strong channel dominance
    """

    # Skin depth controls overall lightness (20 = lighter, 80 = deeper)
    skin_depth = np.random.uniform(20, 80)

    undertone_bias = np.random.choice(['warm', 'cool', 'neutral'], p=[0.35, 0.35, 0.3])

    if undertone_bias == 'warm':
        # Warm: golden/yellow/olive cast
        # R is clearly dominant, G is mid, B is lowest
        r = int(130 + skin_depth * 1.15 + np.random.normal(0, 7))
        g = int(85  + skin_depth * 0.95 + np.random.normal(0, 7))
        b = int(60  + skin_depth * 0.70 + np.random.normal(0, 7))
        label = "Warm"

    elif undertone_bias == 'cool':
        # Cool: pink/rosy cast
        # FIXED: Still skin-colored, but B is elevated relative to G
        # R is still dominant (skin is always reddish), but B approaches G
        r = int(120 + skin_depth * 1.05 + np.random.normal(0, 7))
        g = int(78  + skin_depth * 0.88 + np.random.normal(0, 7))
        b = int(88  + skin_depth * 0.95 + np.random.normal(0, 7))  # B close to G, not dominant
        label = "Cool"

    else:  # neutral
        # Neutral: balanced warmth, no strong cast
        r = int(125 + skin_depth * 1.10 + np.random.normal(0, 7))
        g = int(82  + skin_depth * 0.92 + np.random.normal(0, 7))
        b = int(74  + skin_depth * 0.82 + np.random.normal(0, 7))
        label = "Neutral"

    # Clamp to valid RGB range
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    return [r, g, b], label


def generate_synthetic_data(n_samples=2000):
    """Generate comprehensive dataset with more samples for better accuracy."""
    data = []
    for _ in range(n_samples):
        rgb, label = generate_realistic_skin_tone()
        data.append(rgb + [label])
    return pd.DataFrame(data, columns=["R", "G", "B", "Label"])


if __name__ == "__main__":
    print("Generating skin undertone dataset...")
    df = generate_synthetic_data(2000)
    df.to_csv("skin_undertone_dataset.csv", index=False)
    print(f"Dataset generated! Shape: {df.shape}")
    print(f"\nClass distribution:\n{df['Label'].value_counts()}")
    print(f"\nMean RGB per class:")
    print(df.groupby("Label")[["R", "G", "B"]].mean().round(1))
    print("\nSanity check — all values should look like skin (R > G > B roughly):")
    print(df.groupby("Label")[["R", "G", "B"]].min())

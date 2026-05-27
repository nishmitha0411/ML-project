# AI Personal Color Analysis

An AI-powered Streamlit app that analyzes your skin undertone from a photo and
recommends your personal color season — the framework used by professional image
consultants worldwide.

---

## What it does

1. Detects your face using MediaPipe Face Mesh
2. Samples skin color from precise landmark zones (forehead, cheeks, chin)
3. Classifies your undertone (Warm / Cool / Neutral) using a Random Forest model + ITA score cross-check
4. Determines your skin depth (Light / Medium / Deep)
5. Assigns you one of 9 color seasons with curated palettes, colors to avoid, and jewelry recommendations

---

## Setup instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Generate the dataset

You **must** run this before starting the app. It creates `skin_undertone_dataset.csv`.

```bash
python generate_data.py
```

You should see output like:
```
Dataset generated! Shape: (2000, 4)

Class distribution:
Warm       700
Cool       700
Neutral    600

Mean RGB per class:
          R      G      B
Label
Cool    178.3   114.2  141.8   ← all three are skin-colored
Neutral 181.4   117.6  129.5
Warm    184.7   119.2  106.1
```

If Cool's B value is extremely high (e.g. 172+) and R is very low (e.g. 86),
you have the old broken dataset. Delete it and regenerate.

### Step 3 — Run the app

If the `streamlit` command is not recognized, use Python to launch it instead:

```bash
python -m streamlit run app.py
```

On Windows, you can also run the helper script:

```bash
run_app.bat
```

---

## Photography tips for best results

For the most accurate analysis:

- Use **natural daylight** — avoid warm yellow indoor lighting (it skews everything Warm)
- Face the camera **directly** — no angled shots
- Remove heavy makeup if possible — foundation especially alters detected undertone
- No Instagram/Snapchat filters
- Your face should fill most of the frame

---

## How the seasons work

| Season        | Undertone | Depth  | Characteristics                        |
|---------------|-----------|--------|----------------------------------------|
| Spring        | Warm      | Light  | Fresh, clear, warm — peach and coral   |
| Autumn        | Warm      | Medium | Rich, earthy — olive and mustard        |
| Deep Autumn   | Warm      | Deep   | Bold, dark earth tones                  |
| Summer        | Cool      | Light  | Soft, muted, dusty rose and lavender    |
| Winter        | Cool      | Medium | High contrast, jewel tones              |
| Deep Winter   | Cool      | Deep   | Dramatic, deep cool jewel shades        |
| Neutral Light | Neutral   | Light  | Versatile, soft balanced tones          |
| Neutral Medium| Neutral   | Medium | Earthy mid-tones, both warm and cool   |
| Neutral Deep  | Neutral   | Deep   | Rich, grounded from both families      |

---

## Key technical improvements (v2)

### 1. Fixed dataset (generate_data.py)
The original dataset had Cool-undertone samples with RGB values like R=86, B=172,
which are blue/purple — not real skin. No human skin looks blue. The model had
never seen a real cool-undertone face, making it essentially random for that class.

Fixed: Cool tones now use R≈178, G≈114, B≈142 — still skin-colored, but with an
elevated B channel relative to G, which is the real marker of cool skin.

### 2. Landmark-based skin sampling (app.py)
Original: extracted skin from the entire face bounding box using an HSV mask,
which picked up lips, eyes, eyebrows, and shadows — all of which corrupt the reading.

Fixed: MediaPipe Face Mesh samples specific landmark points (forehead, left cheek,
right cheek, chin, two additional cheek zones) and averages small patches around them.

### 3. ITA cross-check
Added the Individual Typology Angle (ITA), a dermatology-standard formula that
classifies skin undertone from L*a*b* color space. When the ML model and ITA
disagree and confidence is below 75%, ITA is used as the tiebreaker.

### 4. Confidence gating
- Below 60%: shows "Inconclusive" and asks user to retake
- 60–75%: shows result with a moderate-confidence note
- Above 75%: shows full result with green confidence bar

### 5. Definitive single-season output
Original showed "Spring / Autumn" — unhelpfully vague.
Fixed: brightness (skin depth) is used to pick exactly one season per combination.

### 6. Color swatches with hex values
Results now show rendered color swatches alongside avoid colors and jewelry recommendations.

---

## File structure

```
├── app.py                      # Main Streamlit app
├── generate_data.py            # Dataset generator (run first)
├── skin_undertone_dataset.csv  # Generated dataset (git-ignored)
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Requirements

```
streamlit==1.28.0
opencv-python==4.8.1.78
mediapipe==0.10.7
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
Pillow==10.0.0
```

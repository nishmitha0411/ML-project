import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =========================
# 🎨 PAGE SETUP
# =========================
st.set_page_config(
    page_title="AI Personal Color Analysis", 
    layout="centered"
)

# Custom CSS for better look
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.result-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
.tip-box {
    background: #fef3c7;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🎯 TITLE
# =========================
st.title("🎨 AI Personal Color Analysis")
st.markdown("*Discover your perfect colors with AI*")
st.markdown("---")

# =========================
# 📊 LOAD AND TRAIN MODEL
# =========================
@st.cache_resource
def train_model():
    """Load data and train the model"""
    try:
        data = pd.read_csv("skin_undertone_dataset.csv")
    except:
        st.error("❌ Dataset not found! Please run generate_data.py first.")
        return None, None
    
    X = data[['R','G','B']].values
    y = data['Label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    # Show in sidebar
    st.sidebar.success(f"✅ Model Ready! Accuracy: {accuracy*100:.1f}%")
    
    return model, scaler

# Load the model
model, scaler = train_model()
if model is None:
    st.stop()

# =========================
# 📸 INPUT METHOD
# =========================
st.subheader("📸 Upload Your Photo")

col1, col2 = st.columns(2)
with col1:
    option = st.radio("Choose method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    file = st.file_uploader("Choose a photo", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file)
        st.success("✅ Photo uploaded!")
else:
    st.markdown("""
    <div class="tip-box">
    💡 Tips for best results:<br>
    • Face the camera directly<br>
    • Use natural, even lighting<br>
    • Remove glasses if possible<br>
    • Avoid harsh shadows
    </div>
    """, unsafe_allow_html=True)
    
    cam = st.camera_input("Take a photo")
    if cam:
        image = Image.open(cam)

# =========================
# 🔍 PROCESS IMAGE
# =========================
if image is not None:
    
    # Convert to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Initialize face detection
    mp_face_detection = mp.solutions.face_detection
    
    with mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.6
    ) as face_detection:
        
        # Process image
        results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            st.warning("⚠️ No face detected! Please try another photo with better lighting.")
        else:
            # Get the first face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Get face coordinates
            h, w, _ = img_array.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)
            
            # Draw rectangle around face
            cv2.rectangle(img_array, (x, y), (x + w_box, y + h_box), (102, 126, 234), 3)
            
            # Extract face region
            face = img_array[y:y+h_box, x:x+w_box]
            
            # Convert to HSV for better skin detection
            face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV (adjusted for better detection)
            lower_skin = np.array([0, 15, 60], dtype=np.uint8)
            upper_skin = np.array([25, 180, 255], dtype=np.uint8)
            
            # Create mask for skin
            skin_mask = cv2.inRange(face_hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3,3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Get skin pixels
            skin_pixels = face[skin_mask > 0]
            
            if len(skin_pixels) < 50:
                st.warning("⚠️ Not enough skin detected! Please ensure good lighting.")
            else:
                # Calculate average skin color
                avg_color = np.mean(skin_pixels, axis=0).astype(int)
                
                # FIX: Convert BGR to RGB (OpenCV gives BGR)
                b, g, r = avg_color  # Unpack as B, G, R
                # Now r, g, b are correct RGB values
                
                # Show image with detected face
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), 
                        caption="Detected Face", use_column_width=True)
                
                # Show color box (now correct RGB!)
                col1, col2 = st.columns(2)
                with col1:
                    color_box = np.zeros((100, 100, 3), dtype=np.uint8)
                    color_box[:] = [r, g, b]  # RGB order
                    st.image(color_box, caption="Detected Skin Color")
                
                with col2:
                    st.write(f"**RGB Values:**")
                    st.write(f"🔴 Red: {r}")
                    st.write(f"🟢 Green: {g}")
                    st.write(f"🔵 Blue: {b}")
                
                # ML Prediction
                scaled = scaler.transform([[r, g, b]])
                prediction = model.predict(scaled)[0]
                probabilities = model.predict_proba(scaled)[0]
                confidence = max(probabilities) * 100
                
                # Calculate brightness
                brightness = (r + g + b) / 3
                
                # Calculate warmth score (positive = warm, negative = cool)
                warmth_score = (r - b) / (r + g + b + 1) * 2
                
                # Determine seasonal type
                if brightness > 170:
                    tone = "Light"
                elif brightness < 120:
                    tone = "Deep"
                else:
                    tone = "Medium"
                
                if prediction == "Warm":
                    season = "Spring 🌼" if tone == "Light" else "Autumn 🍂"
                elif prediction == "Cool":
                    season = "Summer 🌸" if tone == "Light" else "Winter ❄️"
                else:
                    season = "Neutral 🌿"
                
                # Show results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Skin Undertone", prediction)
                with col2:
                    st.metric("Seasonal Type", season)
                with col3:
                    st.metric("Confidence", f"{confidence:.0f}%")
                
                # Confidence indicator
                if confidence > 80:
                    st.success(f"🎯 High confidence! ({confidence:.0f}%)")
                elif confidence > 60:
                    st.warning(f"📊 Moderate confidence ({confidence:.0f}%)")
                else:
                    st.error(f"⚠️ Low confidence ({confidence:.0f}%) - Try a better photo")
                
                st.markdown("---")
                
                # Color Recommendations based on undertone
                st.subheader("🎨 Your Color Palette")
                
                if prediction == "Warm":
                    st.markdown("""
                    ### 🌟 Best Colors for You (Warm Undertone):
                    
                    | Category | Recommended Colors |
                    |----------|-------------------|
                    | **Clothing** | 🧡 Coral, 💛 Mustard Yellow, 💚 Olive Green, 🤎 Terracotta, 🧡 Peach |
                    | **Jewelry** | ✨ **Gold** - brings out your warmth |
                    | **Lipstick** | 💄 Coral, Peach, Warm Red, Terracotta |
                    | **Blush** | 🌸 Peach, Apricot, Warm Coral |
                    | **Hair Color** | 💇‍♀️ Golden brown, Honey blonde, Auburn |
                    
                    ### ⚠️ Colors to Avoid:
                    - Cool blues, Lavender, Silver, Hot pink
                    """)
                    
                elif prediction == "Cool":
                    st.markdown("""
                    ### 🌟 Best Colors for You (Cool Undertone):
                    
                    | Category | Recommended Colors |
                    |----------|-------------------|
                    | **Clothing** | 💙 Royal Blue, 💜 Lavender, 🩷 Rose Pink, 🤍 Pure White, 💙 Sapphire |
                    | **Jewelry** | ✨ **Silver & Platinum** - enhances your coolness |
                    | **Lipstick** | 💄 Berry, Plum, Blue-based Red, Mauve |
                    | **Blush** | 🌸 Rose, Pink, Berry |
                    | **Hair Color** | 💇‍♀️ Ash brown, Platinum blonde, Cool black |
                    
                    ### ⚠️ Colors to Avoid:
                    - Orange, Mustard, Earthy browns, Olive green
                    """)
                    
                else:  # Neutral
                    st.markdown("""
                    ### 🌟 Best Colors for You (Neutral Undertone):
                    
                    | Category | Recommended Colors |
                    |----------|-------------------|
                    | **Clothing** | 🩶 Soft Gray, 🤍 Ivory, 💙 Dusty Blue, 💚 Sage Green, 🩷 Mauve |
                    | **Jewelry** | ✨ **Both Gold & Silver** work well |
                    | **Lipstick** | 💄 Rose, Mauve, Soft Peach, Neutral Red |
                    | **Blush** | 🌸 Rose, Soft Peach |
                    | **Hair Color** | 💇‍♀️ Most natural shades work well |
                    
                    ### ✨ You're versatile!
                    Most colors suit you - just avoid extremes of warm or cool.
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 📚 LEARN MORE
# =========================
with st.expander("📚 How does this work?"):
    st.markdown("""
    ### How the AI Works:
    
    1. **Face Detection**: Finds your face in the photo using MediaPipe
    2. **Skin Analysis**: Extracts skin color from your face using HSV color space
    3. **AI Prediction**: Uses Random Forest ML model to determine undertone
    4. **Color Suggestions**: Recommends colors based on your undertone
    
    ### What is Skin Undertone?
    
    - **Warm**: Golden, peachy, yellow undertones → Earthy colors suit you
    - **Cool**: Pink, red, bluish undertones → Jewel tones suit you
    - **Neutral**: Balanced mix of warm and cool → Most colors suit you
    
    ### The Four Seasons System:
    
    | Season | Undertone | Value | Best Colors |
    |--------|-----------|-------|-------------|
    | **Spring** | Warm | Light | Coral, Peach, Mint, Gold |
    | **Summer** | Cool | Light | Lavender, Rose, Powder Blue |
    | **Autumn** | Warm | Deep | Mustard, Olive, Rust, Terracotta |
    | **Winter** | Cool | Deep | Royal Blue, Emerald, Burgundy |
    
    ### Tips for Best Results:
    - Use natural daylight
    - Remove makeup for accurate analysis
    - Face the camera directly
    - Ensure even lighting (no harsh shadows)
    """)

st.markdown("---")
st.caption("⚠️ **Note**: Results may vary based on lighting and image quality. For professional advice, consult a color analyst.")
# =========================
# 🔍 IMAGE QUALITY CHECKER
# =========================
def check_image_quality(image_array):
    """Check if image is good for analysis"""
    
    # Calculate brightness
    brightness = np.mean(image_array)
    
    # Check if image is too dark
    if brightness < 50:
        return False, "⚠️ Image is too dark! Please use better lighting."
    
    # Check if image is too bright
    if brightness > 220:
        return False, "⚠️ Image is too bright! Avoid direct sunlight."
    
    # Check image size
    h, w = image_array.shape[:2]
    if h < 100 or w < 100:
        return False, "⚠️ Image is too small! Please use a larger photo."
    
    return True, "✅ Good quality image!"
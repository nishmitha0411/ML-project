# =========================
# 🎨 COLOR PALETTE GENERATOR
# =========================
def generate_color_palette(undertone):
    """Generate color palette based on undertone"""
    
    if undertone == "Warm":
        palette = {
            "Primary": {"name": "Coral", "hex": "#FF7F50", "rgb": "(255, 127, 80)"},
            "Secondary": {"name": "Mustard", "hex": "#E1AD01", "rgb": "(225, 173, 1)"},
            "Accent": {"name": "Olive Green", "hex": "#556B2F", "rgb": "(85, 107, 47)"},
            "Neutral": {"name": "Warm Beige", "hex": "#D2B48C", "rgb": "(210, 180, 140)"}
        }
    elif undertone == "Cool":
        palette = {
            "Primary": {"name": "Royal Blue", "hex": "#4169E1", "rgb": "(65, 105, 225)"},
            "Secondary": {"name": "Lavender", "hex": "#9370DB", "rgb": "(147, 112, 219)"},
            "Accent": {"name": "Rose Pink", "hex": "#DB7093", "rgb": "(219, 112, 147)"},
            "Neutral": {"name": "Silver Gray", "hex": "#C0C0C0", "rgb": "(192, 192, 192)"}
        }
    else:  # Neutral
        palette = {
            "Primary": {"name": "Soft Gray", "hex": "#A9A9A9", "rgb": "(169, 169, 169)"},
            "Secondary": {"name": "Ivory", "hex": "#F5F5DC", "rgb": "(245, 245, 220)"},
            "Accent": {"name": "Dusty Blue", "hex": "#7B9EA8", "rgb": "(123, 158, 168)"},
            "Neutral": {"name": "Taupe", "hex": "#C4A882", "rgb": "(196, 168, 130)"}
        }
    
    return palette

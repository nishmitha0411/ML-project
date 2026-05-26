import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Feature added by Person B ──────────────────
def check_image_quality(image_array):
    """Check if the image is too dark or too bright for accurate analysis."""
    brightness = np.mean(image_array)
    if brightness < 50:
        return False, "too_dark"
    if brightness > 220:
        return False, "too_bright"
    return True, "ok"

st.set_page_config(page_title="ChromaMe", layout="centered", page_icon="🎨")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,600;1,300;1,600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, .stApp { background:#0f0c0a !important; color:#e8ddd0; font-family:'DM Sans',sans-serif; }
.block-container { max-width:700px !important; padding:2.5rem 1.2rem 5rem !important; }

.brand { font-family:'Cormorant Garamond',serif; font-size:clamp(2.6rem,7vw,4.2rem); font-weight:300;
  letter-spacing:0.18em; color:#e8ddd0; text-align:center; line-height:1; }
.brand em { font-style:italic; color:#c9956a; }
.tagline { text-align:center; font-size:9px; letter-spacing:0.32em; text-transform:uppercase;
  color:#5a4e42; margin-top:8px; margin-bottom:40px; }

.card { background:#181410; border:1px solid #2a221a; border-radius:18px; padding:24px 26px; margin-top:14px; }
.card-gold { background:linear-gradient(135deg,#1a1408,#181410); border:1px solid #3d2e18;
  border-radius:18px; padding:24px 26px; margin-top:14px; }
.card-hero { background:#1c1510; border:1px solid #3d2e18; border-radius:22px;
  padding:36px 28px; margin-top:14px; text-align:center; }
.card-dark { background:#0b0908; border:1px solid #1e1810; border-radius:18px;
  padding:30px 28px; margin-top:14px; position:relative; overflow:hidden; }
.card-alert { background:#0f0c0a; border:1px solid #2a221a; border-left:2px solid #c9956a;
  border-radius:0 14px 14px 0; padding:18px 22px; margin-top:14px; }

.lbl { font-size:8.5px; letter-spacing:0.26em; text-transform:uppercase;
  color:#5a4e42; font-weight:500; margin-bottom:9px; }

.season-pill { display:inline-block; border-radius:99px; padding:5px 20px;
  font-size:9.5px; letter-spacing:0.2em; text-transform:uppercase; font-weight:600; margin-bottom:14px; }
.season-name { font-family:'Cormorant Garamond',serif; font-size:clamp(2rem,5vw,3rem);
  font-weight:300; color:#e8ddd0; line-height:1.1; margin-bottom:5px; }
.season-sub { font-size:12px; color:#5a4e42; letter-spacing:0.08em; margin-bottom:18px; }
.season-desc { font-family:'Cormorant Garamond',serif; font-style:italic;
  font-size:17px; color:#9a8270; line-height:1.7; max-width:480px; margin:0 auto; }

.conf-track { background:#0b0908; border-radius:99px; height:3px; margin:16px auto 0;
  max-width:300px; overflow:hidden; }

.swatch-row { display:flex; flex-wrap:wrap; gap:9px; margin-top:12px; }
.sw { width:62px; height:70px; border-radius:14px; display:flex; align-items:flex-end;
  justify-content:center; padding-bottom:7px; font-size:7.5px; font-weight:600;
  letter-spacing:0.05em; text-align:center; text-shadow:0 1px 5px rgba(0,0,0,0.7);
  border:1px solid rgba(255,255,255,0.06); flex-shrink:0; line-height:1.2; }
.sw-avoid { width:62px; height:70px; border-radius:14px; display:flex; align-items:flex-end;
  justify-content:center; padding-bottom:7px; font-size:7.5px; font-weight:600;
  letter-spacing:0.05em; text-align:center; text-shadow:0 1px 5px rgba(0,0,0,0.7);
  border:1.5px solid rgba(180,60,60,0.3); opacity:0.6; flex-shrink:0;
  position:relative; line-height:1.2; }

.chip-row { display:flex; flex-wrap:wrap; gap:7px; margin-top:9px; }
.chip { display:flex; align-items:center; gap:8px; background:#111009;
  border:1px solid #2a221a; border-radius:10px; padding:7px 13px;
  font-size:12px; color:#c8b89a; font-weight:400; }
.dot { width:13px; height:13px; border-radius:50%; flex-shrink:0; border:1px solid rgba(255,255,255,0.1); }

.tag-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
.tag { display:inline-block; background:#111009; border:1px solid #2a221a;
  border-radius:7px; padding:5px 13px; font-size:11.5px; color:#9a8270; }

.info-val { font-size:13.5px; color:#9a8270; line-height:1.7; }
.style-tip { font-family:'Cormorant Garamond',serif; font-style:italic;
  font-size:15.5px; color:#c9956a; line-height:1.65; }

.metals-name { font-family:'Cormorant Garamond',serif; font-size:26px;
  font-weight:300; color:#e8ddd0; margin-bottom:5px; }

.msg-quote { font-family:'Cormorant Garamond',serif; font-size:110px; color:#c9956a;
  opacity:0.1; position:absolute; top:-22px; left:14px; line-height:1; }
.msg-text { font-family:'Cormorant Garamond',serif; font-style:italic;
  font-size:18px; color:#c8b89a; line-height:1.8; position:relative; z-index:1; }

.sec-div { display:flex; align-items:center; gap:12px; margin:24px 0 0;
  color:#3a2e22; font-size:8.5px; letter-spacing:0.24em; text-transform:uppercase; }
.sec-div::before, .sec-div::after { content:''; flex:1; border-top:1px solid #1e1810; }

.skin-ring { display:flex; flex-direction:column; align-items:center; gap:6px; padding-top:14px; }
.rgb-small { font-size:8.5px; letter-spacing:0.1em; color:#3a2e22; font-family:monospace; text-align:center; }

.stRadio > div { gap:14px !important; }
.stRadio label { font-size:12px !important; color:#5a4e42 !important; }
div[data-testid="stFileUploader"] label { color:#5a4e42 !important; font-size:12px !important; }
.stSpinner > div { border-top-color:#c9956a !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SEASON DATA
# ─────────────────────────────────────────────
SD = {
  ("Warm","Light"):{
    "season":"Spring","emoji":"✦","badge_bg":"#d4a040","badge_fg":"#1a0e00",
    "description":"Clear, warm, and luminous. You carry the energy of golden-hour light — fresh, vibrant, and naturally radiant.",
    "palette":[{"name":"Peach","hex":"#FFAD90"},{"name":"Coral","hex":"#FF6B6B"},{"name":"Warm Yellow","hex":"#FFD166"},{"name":"Mint","hex":"#06C090"},{"name":"Ivory","hex":"#FFF0D4"},{"name":"Gold","hex":"#E8B830"}],
    "avoid":[{"name":"Black","hex":"#1a1a1a"},{"name":"Burgundy","hex":"#800020"},{"name":"Charcoal","hex":"#36454F"},{"name":"Navy","hex":"#001F5B"}],
    "metals":"Gold & Rose Gold",
    "metals_why":"They mirror the warmth in your skin — creating a sun-kissed harmony that looks completely natural on you.",
    "makeup":{"Foundation":[{"name":"Warm Beige","hex":"#D4A574"},{"name":"Ivory Warm","hex":"#EDD5A3"}],"Lipstick":[{"name":"Coral Red","hex":"#FF6B6B"},{"name":"Warm Peach","hex":"#FFAD90"},{"name":"Apricot","hex":"#F4A460"}],"Eyeshadow":[{"name":"Warm Brown","hex":"#8B5E3C"},{"name":"Champagne","hex":"#E8C890"},{"name":"Terracotta","hex":"#CC7A4A"}],"Blush":[{"name":"Peach","hex":"#FFAD90"},{"name":"Apricot","hex":"#F4A460"}]},
    "clothing_tags":["Peach","Coral","Ivory","Warm Yellow","Mint","Camel","Light Gold"],
    "fabrics":"Soft cottons, silk blends, lightweight linen — breathable and natural.",
    "patterns":"Delicate florals, watercolour prints, small geometric in warm tones.",
    "style_tip":"Monochromatic warm outfits elongate you beautifully. Try an ivory-to-peach gradient look.",
    "why":"Your golden undertones amplify alongside warm, clear shades — they create radiance from your skin outward rather than competing with it.",
    "personal_msg":"There's something effortlessly luminous about you. Your coloring has a rare clarity that comes alive in warmth and light. The right palette doesn't just make you look good — it makes you look like yourself, turned all the way up.",
  },
  ("Warm","Medium"):{
    "season":"Autumn","emoji":"✦","badge_bg":"#a03800","badge_fg":"#fde8c0",
    "description":"Rich, earthy, and deeply warm. You carry the richness of harvest — golden, grounded, and magnetic.",
    "palette":[{"name":"Burnt Orange","hex":"#CC5500"},{"name":"Olive","hex":"#7a7a00"},{"name":"Mustard","hex":"#D4A800"},{"name":"Rust","hex":"#B7410E"},{"name":"Camel","hex":"#C19A6B"},{"name":"Forest","hex":"#1a7a1a"}],
    "avoid":[{"name":"Pastel Pink","hex":"#FFB6C1"},{"name":"Icy Blue","hex":"#99C5C4"},{"name":"Silver","hex":"#C0C0C0"},{"name":"Lavender","hex":"#E6E6FA"}],
    "metals":"Gold, Bronze & Copper",
    "metals_why":"Earthy metals echo the warm depth of your complexion — rich, never garish.",
    "makeup":{"Foundation":[{"name":"Golden Tan","hex":"#C8874A"},{"name":"Warm Honey","hex":"#D4956A"}],"Lipstick":[{"name":"Brick Red","hex":"#B7410E"},{"name":"Warm Nude","hex":"#C19A6B"},{"name":"Terracotta","hex":"#CC7A4A"}],"Eyeshadow":[{"name":"Deep Olive","hex":"#556B2F"},{"name":"Burnt Sienna","hex":"#C87040"},{"name":"Rich Brown","hex":"#7B3F00"}],"Blush":[{"name":"Copper","hex":"#B87333"},{"name":"Rust Rose","hex":"#B7410E"}]},
    "clothing_tags":["Burnt Orange","Olive","Camel","Mustard","Rust","Forest Green","Chocolate"],
    "fabrics":"Suede, corduroy, raw silk, chunky knits — textures with real depth.",
    "patterns":"Earthy plaids, abstract botanical, tortoiseshell, leopard.",
    "style_tip":"Layering within your palette is your superpower. A mustard turtleneck under a forest green jacket is pure autumn magic.",
    "why":"Your warm medium depth craves pigment-rich earthy tones. Muted-warm shades match the sophistication already in your skin.",
    "personal_msg":"You have an effortless warmth that draws people in without trying. The earthy, spiced palette of Autumn was practically made for someone like you — deep enough to hold your complexity, warm enough to match your energy.",
  },
  ("Warm","Deep"):{
    "season":"Deep Autumn","emoji":"✦","badge_bg":"#4a1a00","badge_fg":"#f0c890",
    "description":"Intense, bold, and powerfully warm. Made for depth — saturated dark earth tones that match your commanding presence.",
    "palette":[{"name":"Deep Olive","hex":"#4a5e20"},{"name":"Terracotta","hex":"#D06050"},{"name":"Dark Gold","hex":"#9a7000"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Burgundy","hex":"#800020"},{"name":"Warm Brown","hex":"#A0522D"}],
    "avoid":[{"name":"Pastel Yellow","hex":"#EEEE80"},{"name":"Baby Blue","hex":"#89CFF0"},{"name":"Neon Green","hex":"#39FF14"},{"name":"Cool Gray","hex":"#909090"}],
    "metals":"Gold, Bronze & Warm Copper",
    "metals_why":"Rich deep metals ground your complexion and add luxurious dimension.",
    "makeup":{"Foundation":[{"name":"Deep Warm","hex":"#8B5A2B"},{"name":"Mahogany","hex":"#A0522D"}],"Lipstick":[{"name":"Deep Berry","hex":"#800020"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Warm Wine","hex":"#722F37"}],"Eyeshadow":[{"name":"Gold","hex":"#9a7000"},{"name":"Deep Plum","hex":"#673147"},{"name":"Forest","hex":"#1a7a1a"}],"Blush":[{"name":"Deep Peach","hex":"#C67C52"},{"name":"Bronze","hex":"#A07030"}]},
    "clothing_tags":["Chocolate","Burgundy","Dark Olive","Terracotta","Dark Gold","Deep Teal"],
    "fabrics":"Rich velvets, heavyweight silk, structured leather — fabrics with presence.",
    "patterns":"Bold animal prints, abstract art prints, rich dark plaids.",
    "style_tip":"Don't fear head-to-toe depth. A full chocolate look or all-burgundy ensemble is your signature move.",
    "why":"Your deep warm coloring craves saturation. Light or cool colors fade against your richness — bold and earthy lets you be fully seen.",
    "personal_msg":"There's a gravitational pull to your coloring — bold, warm, and impossible to ignore. Deep Autumn is rare and extraordinary. Own the depth that's already there; it's your greatest style asset.",
  },
  ("Cool","Light"):{
    "season":"Summer","emoji":"✦","badge_bg":"#4a7090","badge_fg":"#e0f0ff",
    "description":"Soft, cool, and quietly elegant. A dreamlike delicacy — misty, silvery, and effortlessly refined.",
    "palette":[{"name":"Dusty Rose","hex":"#C89080"},{"name":"Lavender","hex":"#9a70c0"},{"name":"Powder Blue","hex":"#90c0cc"},{"name":"Mauve","hex":"#b090b0"},{"name":"Soft Gray","hex":"#a0a0b0"},{"name":"Rose Beige","hex":"#D8B8A8"}],
    "avoid":[{"name":"Orange","hex":"#E07020"},{"name":"Rust","hex":"#B7410E"},{"name":"Warm Yellow","hex":"#D4A800"},{"name":"Olive","hex":"#7a7a00"}],
    "metals":"Silver & White Gold",
    "metals_why":"Cool metals reflect the delicate rosiness in your skin and feel effortlessly natural.",
    "makeup":{"Foundation":[{"name":"Cool Porcelain","hex":"#E8CCC0"},{"name":"Pink Beige","hex":"#D8B8A8"}],"Lipstick":[{"name":"Dusty Rose","hex":"#B07070"},{"name":"Berry Pink","hex":"#B01070"},{"name":"Rose Petal","hex":"#C05080"}],"Eyeshadow":[{"name":"Lavender","hex":"#9a70c0"},{"name":"Dusty Plum","hex":"#7a4080"},{"name":"Soft Gray","hex":"#8888a0"}],"Blush":[{"name":"Rose","hex":"#E08080"},{"name":"Mauve","hex":"#b090b0"}]},
    "clothing_tags":["Dusty Rose","Soft Lavender","Powder Blue","Mauve","Soft Gray","Rose White"],
    "fabrics":"Chiffon, satin, soft cashmere — ethereal fabrics that float.",
    "patterns":"Soft watercolour florals, delicate ditsy prints, ombre, subtle stripe.",
    "style_tip":"Tone-on-tone muted looks are magical on you. Mauve trousers + dusty rose top + silver jewellery = effortless.",
    "why":"Your soft cool undertone shines with muted dusty tones. Bright or warm colors overpower your delicate coloring — softness lets your natural elegance lead.",
    "personal_msg":"There's a soft, understated quality to your beauty that gets more striking the longer you look. Summer is the season of quiet confidence — you don't need to announce yourself. The right palette lets that grace simply breathe.",
  },
  ("Cool","Medium"):{
    "season":"Winter","emoji":"✦","badge_bg":"#1a2880","badge_fg":"#c0d8ff",
    "description":"Bold, cool, and striking. High contrast is your superpower — vivid jewel shades create looks you won't forget.",
    "palette":[{"name":"Royal Blue","hex":"#2850cc"},{"name":"Magenta","hex":"#b00060"},{"name":"Emerald","hex":"#006840"},{"name":"Pure White","hex":"#E8F0F8"},{"name":"True Red","hex":"#c00020"},{"name":"Ice Pink","hex":"#E080A0"}],
    "avoid":[{"name":"Warm Beige","hex":"#E8D8B0"},{"name":"Mustard","hex":"#D4A800"},{"name":"Camel","hex":"#C19A6B"},{"name":"Burnt Orange","hex":"#CC5500"}],
    "metals":"Silver, Platinum & White Gold",
    "metals_why":"Crisp cool metals amplify your natural contrast and look undeniably sharp.",
    "makeup":{"Foundation":[{"name":"Cool Beige","hex":"#C8A880"},{"name":"Neutral Med","hex":"#B89870"}],"Lipstick":[{"name":"True Red","hex":"#c00020"},{"name":"Berry","hex":"#800050"},{"name":"Raspberry","hex":"#b01050"}],"Eyeshadow":[{"name":"Charcoal","hex":"#384050"},{"name":"Sapphire","hex":"#0a48a0"},{"name":"Deep Plum","hex":"#673147"}],"Blush":[{"name":"Cool Pink","hex":"#d05090"},{"name":"Berry","hex":"#b01060"}]},
    "clothing_tags":["True White","True Black","Royal Blue","Magenta","Emerald","Ice Pink","True Red"],
    "fabrics":"Structured cotton, crisp silk, tailored wool — clothes with architecture.",
    "patterns":"Graphic prints, bold stripes, strong geometrics, classic houndstooth.",
    "style_tip":"Embrace contrast. Black + white or full jewel-tone — you're one of the rare types who looks incredible in stark, high-contrast combinations.",
    "why":"Your cool undertone and medium depth create natural contrast. Vivid shades work with that contrast — murky or warm tones cancel it out.",
    "personal_msg":"You have the kind of coloring that stops people mid-sentence. Winter is bold, precise, and high-impact — and so are you. When you dress in your true colors, there's a crispness to your whole look that reads as effortlessly commanding.",
  },
  ("Cool","Deep"):{
    "season":"Deep Winter","emoji":"✦","badge_bg":"#060812","badge_fg":"#a0c0d8",
    "description":"Dramatic, intense, and powerfully cool. Built for depth and richness — the jeweled darkness of midnight.",
    "palette":[{"name":"Burgundy","hex":"#800020"},{"name":"Cobalt","hex":"#0040a0"},{"name":"Plum","hex":"#673147"},{"name":"Charcoal","hex":"#364050"},{"name":"Deep Teal","hex":"#007070"},{"name":"Onyx","hex":"#1a1a1a"}],
    "avoid":[{"name":"Light Beige","hex":"#E8E0C8"},{"name":"Warm Orange","hex":"#E07020"},{"name":"Gold","hex":"#D4A800"},{"name":"Peach","hex":"#FFAD90"}],
    "metals":"Silver, Platinum & Dark Rhodium",
    "metals_why":"Cool dark metals match your deep cool richness and add a moody, dramatic dimension.",
    "makeup":{"Foundation":[{"name":"Deep Cool","hex":"#706050"},{"name":"Ebony Cool","hex":"#604838"}],"Lipstick":[{"name":"Deep Plum","hex":"#4B0082"},{"name":"Oxblood","hex":"#800020"},{"name":"Dark Berry","hex":"#501880"}],"Eyeshadow":[{"name":"Midnight","hex":"#181870"},{"name":"Deep Plum","hex":"#673147"},{"name":"Graphite","hex":"#383838"}],"Blush":[{"name":"Deep Rose","hex":"#904060"},{"name":"Berry","hex":"#800050"}]},
    "clothing_tags":["True Black","Cobalt","Plum","Burgundy","Deep Teal","Charcoal","Deep Navy"],
    "fabrics":"Velvet, heavyweight silk, luxe jersey — fabrics with gravitas.",
    "patterns":"Bold abstract, deep jewel-tone florals, dramatic geometric.",
    "style_tip":"All-black is your baseline — you pull it off like no one else. Add one jewel-toned piece (cobalt or burgundy) for dimension.",
    "why":"Your deep cool complexion craves rich dark cool-toned depth. Pale or warm tones look washed against your richness.",
    "personal_msg":"There's an intensity to your coloring that's completely arresting. Deep Winter is the rarest season — dramatic, cool, and undeniably powerful. You were made for dark jeweled depth. Lean into it completely.",
  },
  ("Neutral","Light"):{
    "season":"Neutral Light","emoji":"✦","badge_bg":"#507040","badge_fg":"#d8f0c0",
    "description":"Versatile, fresh, and beautifully balanced. You carry the unique gift of flexibility — soft tones from both warm and cool families suit you.",
    "palette":[{"name":"Blush","hex":"#cc5070"},{"name":"Soft Teal","hex":"#30909a"},{"name":"Warm Cream","hex":"#E8DCA0"},{"name":"Stone Gray","hex":"#888078"},{"name":"Sage","hex":"#80a060"},{"name":"Dusty Purple","hex":"#8a6898"}],
    "avoid":[{"name":"Neon Yellow","hex":"#BBEE00"},{"name":"Harsh Black","hex":"#000000"},{"name":"Pure White","hex":"#FFFFFF"},{"name":"Neon Orange","hex":"#FF6500"}],
    "metals":"Gold & Silver — both work",
    "metals_why":"Your balanced undertone gives you the rare ability to wear both — mix and match freely.",
    "makeup":{"Foundation":[{"name":"Neutral Ivory","hex":"#E0C8A8"},{"name":"Warm Light","hex":"#D8C098"}],"Lipstick":[{"name":"Blush Pink","hex":"#cc5070"},{"name":"Soft Coral","hex":"#E07860"},{"name":"Rosy Nude","hex":"#b07070"}],"Eyeshadow":[{"name":"Soft Taupe","hex":"#908070"},{"name":"Sage","hex":"#80a060"},{"name":"Dusty Mauve","hex":"#8a6898"}],"Blush":[{"name":"Soft Blush","hex":"#E0A0A8"},{"name":"Peach Pink","hex":"#E8A080"}]},
    "clothing_tags":["Sage Green","Soft Teal","Blush","Warm Cream","Stone Gray","Dusty Purple"],
    "fabrics":"Soft knits, light linen, airy cotton — understated quality.",
    "patterns":"Small-scale prints, subtle texture, tonal patterns.",
    "style_tip":"You're one of the few who can mix a warm scarf with a cool outfit. Use this gift — eclectic colour mixing is uniquely yours.",
    "why":"Your neutral undertone means neither warm nor cool shades clash. Soft balanced tones bring harmony — extremes tip the balance.",
    "personal_msg":"Having a neutral undertone is genuinely rare — you exist in a beautiful in-between space that most people never access. You have styling freedoms others would envy. Use them.",
  },
  ("Neutral","Medium"):{
    "season":"Neutral Medium","emoji":"✦","badge_bg":"#6a5040","badge_fg":"#f0e0c8",
    "description":"Balanced and beautifully adaptable. Earthy mid-tones from both warm and cool families suit you naturally.",
    "palette":[{"name":"Sage","hex":"#80a060"},{"name":"Terracotta","hex":"#c06048"},{"name":"Slate Blue","hex":"#5a7088"},{"name":"Caramel","hex":"#b07830"},{"name":"Muted Teal","hex":"#3a8888"},{"name":"Warm Taupe","hex":"#907060"}],
    "avoid":[{"name":"Neon","hex":"#39FF14"},{"name":"Pure White","hex":"#F8F8F8"},{"name":"Cool Pastels","hex":"#90c0d0"},{"name":"Harsh Black","hex":"#000000"}],
    "metals":"Gold & Silver — both work",
    "metals_why":"You can mix metals beautifully — try gold rings with a silver watch for an eclectic edge.",
    "makeup":{"Foundation":[{"name":"Neutral Beige","hex":"#c09060"},{"name":"Warm Medium","hex":"#b08040"}],"Lipstick":[{"name":"Terracotta","hex":"#b06838"},{"name":"Muted Berry","hex":"#885058"},{"name":"Warm Mauve","hex":"#986070"}],"Eyeshadow":[{"name":"Warm Taupe","hex":"#907060"},{"name":"Muted Teal","hex":"#3a8888"},{"name":"Caramel","hex":"#b07830"}],"Blush":[{"name":"Warm Peach","hex":"#E09070"},{"name":"Soft Coral","hex":"#D08060"}]},
    "clothing_tags":["Sage","Terracotta","Caramel","Muted Teal","Slate Blue","Warm Taupe"],
    "fabrics":"Relaxed linen, soft leather, woven textures — grounded and natural.",
    "patterns":"Earthy abstract, botanical prints, subtle plaid.",
    "style_tip":"Mix teal with terracotta — it works because your undertone bridges both worlds. You can build an eclectic wardrobe others can't.",
    "why":"Your balanced medium-depth coloring is versatile. Muted earthy tones from both families complement your adaptability without clashing.",
    "personal_msg":"Being Neutral Medium means you're the chameleon of the colour world — you can walk into almost any palette and make it work. That's a creative superpower. Choose tones with depth; your coloring has richness that deserves to be met in kind.",
  },
  ("Neutral","Deep"):{
    "season":"Neutral Deep","emoji":"✦","badge_bg":"#201808","badge_fg":"#e0c890",
    "description":"Rich, grounded, and powerfully balanced. Deep sophisticated tones from warm and cool families suit your remarkable depth.",
    "palette":[{"name":"Forest","hex":"#1a6a1a"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Navy","hex":"#001850"},{"name":"Berry","hex":"#783050"},{"name":"Deep Teal","hex":"#005050"},{"name":"Warm Brown","hex":"#8a4020"}],
    "avoid":[{"name":"Pastel Yellow","hex":"#EEEE80"},{"name":"Baby Pink","hex":"#F4C2C2"},{"name":"Pale Pastels","hex":"#D0D0F0"},{"name":"Light Beige","hex":"#E8E0C8"}],
    "metals":"Gold & Silver — both work",
    "metals_why":"Deep neutrals carry both metals effortlessly — use gold for warmth, silver for edge.",
    "makeup":{"Foundation":[{"name":"Deep Neutral","hex":"#785030"},{"name":"Rich Ebony","hex":"#684020"}],"Lipstick":[{"name":"Deep Berry","hex":"#783050"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Dark Plum","hex":"#673147"}],"Eyeshadow":[{"name":"Forest","hex":"#1a6a1a"},{"name":"Deep Teal","hex":"#005050"},{"name":"Rich Brown","hex":"#7B3F00"}],"Blush":[{"name":"Deep Rose","hex":"#904060"},{"name":"Rich Berry","hex":"#783050"}]},
    "clothing_tags":["Forest Green","Chocolate","Navy","Berry","Deep Teal","Warm Brown","Charcoal"],
    "fabrics":"Rich denim, structured wool, luxe cotton — depth and presence.",
    "patterns":"Bold graphic prints, jewel-toned patterns, strong geometrics.",
    "style_tip":"Deep saturated colors are your signature. Use rich contrasts — forest green + chocolate, navy + deep berry.",
    "why":"Your deep neutral coloring handles rich saturated tones from both families. Pale pastels disappear against your richness — depth meets depth.",
    "personal_msg":"You have one of the most grounded, versatile, and powerful color profiles possible. Deep Neutral is commanding — it holds warmth and coolness at once with ease. You don't need to choose a side. You're the rare person who gets both worlds.",
  },
}


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    try:
        data = pd.read_csv("skin_undertone_dataset.csv")
    except FileNotFoundError:
        st.error("skin_undertone_dataset.csv not found. Run python generate_data.py first.")
        st.stop()
    X, y = data[["R","G","B"]], data["Label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    mdl = RandomForestClassifier(n_estimators=200, random_state=42)
    mdl.fit(sc.fit_transform(Xtr), ytr)
    return mdl, sc, mdl.score(sc.transform(Xte), yte)

model, scaler, model_acc = train_model()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def classify_depth(br):
    if br > 165: return "Light"
    if br < 115: return "Deep"
    return "Medium"

def ita_undertone(r, g, b):
    arr = np.uint8([[[int(b),int(g),int(r)]]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2Lab)[0][0]
    L = float(lab[0])/255.0*100
    bl = float(lab[2])-128
    if abs(bl)<1e-3: bl=1e-3
    ita = np.degrees(np.arctan((L-50)/bl))
    if ita>28: return "Warm"
    if ita<10: return "Cool"
    return "Neutral"

def extract_landmarks(img_bgr):
    h,w = img_bgr.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: return None
        lm = res.multi_face_landmarks[0].landmark
        pixels, ps = [], 6
        for idx in [10,234,454,152,50,280]:
            cx,cy = int(lm[idx].x*w), int(lm[idx].y*h)
            patch = img_bgr[max(0,cy-ps):min(h,cy+ps), max(0,cx-ps):min(w,cx+ps)]
            if patch.size>0: pixels.append(np.mean(patch.reshape(-1,3),axis=0))
        if not pixels: return None
        b,g,r = np.mean(pixels,axis=0)
        return float(r),float(g),float(b)

def is_dark(hex_c):
    h=hex_c.lstrip("#"); r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return (0.299*r+0.587*g+0.114*b)<128

def swatches(colors, avoid=False):
    items=""
    for c in colors:
        tc="#fff" if is_dark(c["hex"]) else "#222"
        cls="sw-avoid" if avoid else "sw"
        xmark='<span style="position:absolute;top:6px;right:7px;font-size:10px;color:#e05050;font-weight:700">✕</span>' if avoid else ""
        items+=f'<div class="{cls}" style="background:{c["hex"]};color:{tc};position:relative">{xmark}{c["name"]}</div>'
    return f'<div class="swatch-row">{items}</div>'

def chips(items):
    out="".join(f'<div class="chip"><div class="dot" style="background:{i["hex"]}"></div>{i["name"]}</div>' for i in items)
    return f'<div class="chip-row">{out}</div>'

def tags(items):
    out="".join(f'<span class="tag">{t}</span>' for t in items)
    return f'<div class="tag-row">{out}</div>'

def conf_bar(c):
    col="#6a9a70" if c>=80 else "#c9956a" if c>=65 else "#a05040"
    return f'<div class="conf-track"><div style="height:3px;border-radius:99px;background:{col};width:{c:.0f}%"></div></div>'


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="brand">Chroma<em>Me</em></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">AI · Personal Color Analysis</div>', unsafe_allow_html=True)

st.markdown("""<div class="card" style="text-align:center">
<p style="font-size:13px;color:#3a2e22;line-height:1.75;margin:0">
Upload a <strong style="color:#5a4e42">clear, front-facing photo</strong> in natural daylight.<br>
No filters, no heavy makeup. Natural light gives the most accurate reading.
</p></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
opt = st.radio("Input:", ["📁  Upload photo", "📷  Use camera"], horizontal=True)
image = None
if opt == "📁  Upload photo":
    f = st.file_uploader("Upload face photo (jpg / png)", type=["jpg","png","jpeg"])
    if f: image = Image.open(f)
else:
    cam = st.camera_input("Face the camera in natural light")
    if cam: image = Image.open(cam)


# ─────────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────────
if image is not None:
    with st.spinner("Analyzing your color profile…"):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # ── Person B: Image quality check ──────────────────────────
        quality_ok, quality_status = check_image_quality(np.array(image))
        if not quality_ok:
            pass  # shown outside spinner below
        # ───────────────────────────────────────────────────────────

        result = extract_landmarks(img_bgr)

        if result is None:
            mp_fd = mp.solutions.face_detection
            with mp_fd.FaceDetection(min_detection_confidence=0.5) as fd:
                det = fd.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not det.detections:
                st.error("No face detected. Use a clear front-facing photo in good lighting.")
                st.stop()
            bb = det.detections[0].location_data.relative_bounding_box
            H,W = img_bgr.shape[:2]
            face = img_bgr[max(0,int(bb.ymin*H)):int((bb.ymin+bb.height)*H), max(0,int(bb.xmin*W)):int((bb.xmin+bb.width)*W)]
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            skin = face[cv2.inRange(hsv,np.array([0,15,60]),np.array([25,180,255]))>0]
            if len(skin)<50:
                st.error("Couldn't extract skin pixels. Try better lighting or move closer.")
                st.stop()
            avg=np.mean(skin,axis=0); r,g,b=float(avg[2]),float(avg[1]),float(avg[0])
        else:
            r,g,b = result

        scaled=scaler.transform([[r,g,b]])
        pred=model.predict(scaled)[0]
        probs=model.predict_proba(scaled)[0]
        conf=float(max(probs))*100
        ita=ita_undertone(r,g,b)
        if ita!=pred and conf<75: pred=ita; conf=max(conf,55.0)
        depth=classify_depth((r+g+b)/3)
        info=SD.get((pred,depth), SD[("Neutral","Medium")])

    # ── Person B: Image quality banner ────────────────────────────
    if quality_status == "too_dark":
        st.markdown("""<div class="card-alert" style="border-left-color:#a05040">
        <div class="lbl">image quality warning</div>
        <div class="info-val" style="margin-top:6px">
        Image appears <strong style="color:#c9956a">too dark</strong> — results may be less accurate.
        Try again in brighter natural light for the best reading.
        </div></div>""", unsafe_allow_html=True)
    elif quality_status == "too_bright":
        st.markdown("""<div class="card-alert" style="border-left-color:#a05040">
        <div class="lbl">image quality warning</div>
        <div class="info-val" style="margin-top:6px">
        Image appears <strong style="color:#c9956a">overexposed</strong> — results may be less accurate.
        Try again away from direct harsh light or flash.
        </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="card-alert" style="border-left-color:#5a8a60">
        <div class="lbl">image quality</div>
        <div class="info-val" style="margin-top:4px;color:#7aaa80">Good lighting detected — analysis should be accurate.</div>
        </div>""", unsafe_allow_html=True)
    # ──────────────────────────────────────────────────────────────

    if conf<60:
        st.markdown("""<div class="card-hero"><div style="font-size:2rem;margin-bottom:12px;opacity:.5">◈</div>
        <div class="season-name" style="font-size:1.6rem">Inconclusive</div>
        <div class="season-desc" style="margin-top:12px">Confidence too low for a reliable reading.
        Try again in natural daylight, no heavy makeup, face directly forward, and let your face fill the frame.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    hex_skin="#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))

    # Photo + skin swatch
    c1,c2=st.columns([5,1])
    with c1: st.image(image, use_column_width=True)
    with c2:
        st.markdown(f"""<div class="skin-ring">
        <div style="width:50px;height:50px;border-radius:50%;background:{hex_skin};
            border:2px solid #2a221a;box-shadow:0 0 20px {hex_skin}44"></div>
        <div class="rgb-small">detected<br>skin tone</div>
        </div>""", unsafe_allow_html=True)

    # Hero card
    bb_bg,bb_fg=info["badge_bg"],info["badge_fg"]
    bar=conf_bar(conf)
    st.markdown(f"""<div class="card-hero">
    <span class="season-pill" style="background:{bb_bg};color:{bb_fg}">{info["emoji"]}  {info["season"]}</span>
    <div class="season-name">{pred} · {depth}</div>
    <div class="season-sub">Confidence {conf:.0f}%  ·  Model accuracy {model_acc*100:.0f}%</div>
    {bar}
    <div class="season-desc" style="margin-top:22px">{info["description"]}</div>
    <div style="margin-top:14px;font-size:10px;color:#2a1e12;letter-spacing:0.12em">
        {hex_skin.upper()}  ·  rgb({int(r)}, {int(g)}, {int(b)})</div>
    </div>""", unsafe_allow_html=True)

    if 60<=conf<75:
        st.info("Moderate confidence — a solid starting point. Retaking in natural daylight can sharpen this further.")

    # Best colors
    st.markdown('<div class="sec-div">your colors</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{swatches(info["palette"])}</div>', unsafe_allow_html=True)

    # ── Person B: Color palette with hex codes ─────────────────────
    palette_chips = "".join(
        f'<div class="chip" style="flex-direction:column;align-items:flex-start;gap:4px;padding:10px 14px">'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div class="dot" style="background:{c["hex"]};width:16px;height:16px"></div>'
        f'<span style="font-size:12px;color:#c8b89a;font-weight:500">{c["name"]}</span>'
        f'</div>'
        f'<span style="font-size:10px;color:#3a2e22;font-family:monospace;letter-spacing:0.08em">{c["hex"].upper()}</span>'
        f'</div>'
        for c in info["palette"]
    )
    st.markdown(f"""<div class="card" style="margin-top:8px">
    <div class="lbl">hex codes</div>
    <div class="chip-row" style="margin-top:10px">{palette_chips}</div>
    </div>""", unsafe_allow_html=True)
    # ──────────────────────────────────────────────────────────────

    # Avoid
    st.markdown('<div class="sec-div">colors to avoid</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card" style="border-color:#3a1a1a">
    <div class="lbl" style="margin-bottom:6px">These clash with your undertone and can wash you out</div>
    {swatches(info["avoid"],avoid=True)}</div>""", unsafe_allow_html=True)

    # Makeup
    st.markdown('<div class="sec-div">makeup</div>', unsafe_allow_html=True)
    makeup_html=""
    for label,items in info["makeup"].items():
        makeup_html+=f'<div class="lbl" style="margin-top:16px">{label}</div>{chips(items)}'
    st.markdown(f'<div class="card">{makeup_html}</div>', unsafe_allow_html=True)

    # Clothing
    st.markdown('<div class="sec-div">clothing & style</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card">
    <div class="lbl">best colors to wear</div>{tags(info["clothing_tags"])}
    <div style="border-top:1px solid #1e1810;margin-top:18px;padding-top:16px">
        <div class="lbl">fabrics</div><div class="info-val">{info["fabrics"]}</div></div>
    <div style="border-top:1px solid #1e1810;margin-top:14px;padding-top:14px">
        <div class="lbl">patterns</div><div class="info-val">{info["patterns"]}</div></div>
    <div style="border-top:1px solid #1e1810;margin-top:14px;padding-top:14px">
        <div class="lbl">style tip</div><div class="style-tip">{info["style_tip"]}</div></div>
    </div>""", unsafe_allow_html=True)

    # Jewelry
    st.markdown('<div class="sec-div">jewelry & metals</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card">
    <div class="metals-name">{info["metals"]}</div>
    <div class="info-val" style="margin-top:6px">{info["metals_why"]}</div>
    </div>""", unsafe_allow_html=True)

    # Why
    st.markdown('<div class="sec-div">why this works for you</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card-gold">
    <div class="lbl">the science</div>
    <div class="info-val" style="margin-top:4px">{info["why"]}</div>
    </div>""", unsafe_allow_html=True)

    # Fairness
    st.markdown('<div class="sec-div">accuracy & fairness</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card-alert">
    <div class="lbl">transparency note</div>
    <div class="info-val" style="margin-top:8px">
    <strong style="color:#c9956a">Lighting matters most.</strong>
    Yellow indoor light shifts readings Warm; blue LED shifts Cool. Natural daylight is always best.<br><br>
    <strong style="color:#c9956a">Camera sensors vary.</strong>
    Some phones compress warm tones. If results feel off, try a different device or step outside.<br><br>
    <strong style="color:#c9956a">For deeper skin tones,</strong> undertone is the most reliable signal — trust that above depth if uncertain.<br><br>
    This is a guide, not a verdict. Professional in-person analysis with physical drapes is always more precise.
    </div></div>""", unsafe_allow_html=True)

    # Personal message
    st.markdown('<div class="sec-div">a note just for you</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card-dark">
    <div class="msg-quote">"</div>
    <div class="lbl" style="color:#3a2e22;margin-bottom:14px">personal message</div>
    <div class="msg-text">{info["personal_msg"]}</div>
    </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""<div style="text-align:center;margin-top:44px;color:#2a1e12;font-size:10px;letter-spacing:0.18em">
    CHROMAME  ·  AI PERSONAL COLOR ANALYSIS<br>
    <span style="color:#201810;font-size:9px;letter-spacing:0.1em">Results are a guide. Professional draping gives maximum precision.</span>
    </div>""", unsafe_allow_html=True)

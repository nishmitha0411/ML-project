import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
import datetime
import warnings
warnings.filterwarnings("ignore")

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.graphics.shapes import Drawing, Rect, String
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# UPGRADE 1 — WHITE BALANCE CORRECTION
# ═══════════════════════════════════════════════════════════════
def white_balance(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    lab2 = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab2)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ═══════════════════════════════════════════════════════════════
# UPGRADE 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def engineer_features(r, g, b):
    rg = r / (g + 1e-6)
    rb = r / (b + 1e-6)
    gb = g / (b + 1e-6)
    br = (r + g + b) / 3.0
    arr = np.uint8([[[int(b), int(g), int(r)]]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2Lab)[0][0]
    L = float(lab[0]) / 255.0 * 100
    bl = float(lab[2]) - 128
    if abs(bl) < 1e-3: bl = 1e-3
    ita = np.degrees(np.arctan((L - 50) / bl))
    return [r, g, b, rg, rb, gb, br, ita]

def engineer_df(df):
    df = df.copy()
    df["RG"] = df["R"] / (df["G"] + 1e-6)
    df["RB"] = df["R"] / (df["B"] + 1e-6)
    df["GB"] = df["G"] / (df["B"] + 1e-6)
    df["BR"] = (df["R"] + df["G"] + df["B"]) / 3.0
    def row_ita(row):
        arr = np.uint8([[[int(row.B), int(row.G), int(row.R)]]])
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2Lab)[0][0]
        L = float(lab[0]) / 255.0 * 100
        bl = float(lab[2]) - 128
        if abs(bl) < 1e-3: bl = 1e-3
        return np.degrees(np.arctan((L - 50) / bl))
    df["ITA"] = df.apply(row_ita, axis=1)
    return df

FEATURES = ["R","G","B","RG","RB","GB","BR","ITA"]

# ═══════════════════════════════════════════════════════════════
# IMAGE QUALITY CHECK
# ═══════════════════════════════════════════════════════════════
def check_image_quality(image_array):
    brightness = np.mean(image_array)
    if brightness < 50: return False, "too_dark"
    if brightness > 220: return False, "too_bright"
    return True, "ok"

# NOTE: manual_colour_picker_analysis and show_top_3_probs are defined
# after the SD dict (below) since they reference it directly.

# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 2: UNDERTONE QUIZ
# ═══════════════════════════════════════════════════════════════
def undertone_quiz():
    """5-question quiz for determining undertone without a camera"""
    score_warm = 0
    score_cool = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Question 1")
        q1 = st.radio("Vein colour on inner wrist:", 
                      ["Blue/Purple", "Blue-Green", "Green"], key="quiz1")
    with col2:
        st.markdown("### Question 2")
        q2 = st.radio("Which metal looks better?", 
                      ["Silver/Platinum", "Both equally", "Gold/Rose Gold"], key="quiz2")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Question 3")
        q3 = st.radio("Sun reaction:", 
                      ["Burns easily, rarely tans", "Burns then tans", 
                       "Tans easily, rarely burns", "Naturally tan, never burns"], key="quiz3")
    with col4:
        st.markdown("### Question 4")
        q4 = st.radio("Best white shade:", 
                      ["Bright/Pure White", "Both", "Ivory/Cream"], key="quiz4")
    
    st.markdown("### Question 5")
    q5 = st.radio("Most flattering lipstick:", 
                  ["Berry/Magenta", "Both", "Coral/Peach"], key="quiz5", horizontal=True)
    
    # Scoring
    if q1 == "Blue/Purple": score_cool += 2
    elif q1 == "Green": score_warm += 2
    else: score_warm += 1; score_cool += 1
    
    if q2 == "Silver/Platinum": score_cool += 2
    elif q2 == "Gold/Rose Gold": score_warm += 2
    else: score_warm += 1; score_cool += 1
    
    if q3 == "Burns easily, rarely tans": score_cool += 2
    elif q3 == "Burns then tans": score_warm += 1; score_cool += 1
    elif q3 == "Tans easily, rarely burns": score_warm += 2
    else: score_warm += 3
    
    if q4 == "Bright/Pure White": score_cool += 2
    elif q4 == "Ivory/Cream": score_warm += 2
    else: score_warm += 1; score_cool += 1
    
    if q5 == "Berry/Magenta": score_cool += 2
    elif q5 == "Coral/Peach": score_warm += 2
    else: score_warm += 1; score_cool += 1
    
    if score_warm > score_cool + 2:
        return "Warm", max(60, min(95, 60 + (score_warm - score_cool) * 5))
    elif score_cool > score_warm + 2:
        return "Cool", max(60, min(95, 60 + (score_cool - score_warm) * 5))
    else:
        return "Neutral", 70

# show_top_3_probs is defined after SD dict below

# ═══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════
def generate_pdf(pred, depth, season, conf, model_acc, hex_skin, r, g, b, info, secondary=None):
    if not REPORTLAB_AVAILABLE:
        return io.BytesIO(b"PDF generation requires reportlab. Run: pip install reportlab")

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    HRFlowable, Table, TableStyle, KeepTogether)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.graphics.shapes import Drawing, Rect, String

    buf = io.BytesIO()
    PAGE_W = A4[0] - 40*mm  # usable width

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=16*mm, bottomMargin=16*mm)

    # ── Colours ──
    C_GOLD   = rl_colors.HexColor("#c97b3a")
    C_DARK   = rl_colors.HexColor("#2e1f14")
    C_MID    = rl_colors.HexColor("#7a5c44")
    C_DIM    = rl_colors.HexColor("#b09070")
    C_BG     = rl_colors.HexColor("#fdf8f4")
    C_CARD   = rl_colors.HexColor("#ffffff")
    C_BORDER = rl_colors.HexColor("#ede0d4")

    # ── Styles ──
    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=10, textColor=C_DARK,
                    leading=15, spaceAfter=4, spaceBefore=0)
        base.update(kw)
        return ParagraphStyle(name, **base)

    ST = S("title",  fontName="Helvetica-Bold", fontSize=30, textColor=C_GOLD,
           alignment=TA_CENTER, spaceAfter=2, leading=34)
    SS = S("sub",    fontSize=8,  textColor=C_DIM, alignment=TA_CENTER,
           spaceAfter=14, letterSpacing=2)
    SH = S("season", fontName="Helvetica-Bold", fontSize=22, textColor=C_DARK,
           alignment=TA_CENTER, spaceAfter=4, leading=26)
    SD2= S("sdep",   fontSize=11, textColor=C_MID, alignment=TA_CENTER,
           spaceAfter=4, leading=15)
    SDE= S("desc",   fontSize=10, textColor=C_MID, alignment=TA_CENTER,
           leading=16, spaceAfter=12)
    SL = S("lbl",    fontName="Helvetica-Bold", fontSize=7,  textColor=C_DIM,
           spaceAfter=6, leading=10, letterSpacing=1.5)
    SV = S("val",    fontSize=10, textColor=C_DARK, leading=15, spaceAfter=6)
    SI = S("ital",   fontName="Helvetica-Oblique", fontSize=11, textColor=C_MID,
           leading=17, spaceAfter=6)
    SF = S("foot",   fontSize=7,  textColor=C_DIM,  alignment=TA_CENTER,
           leading=11, spaceAfter=2)

    def hr(top=8, bot=8):
        return HRFlowable(width="100%", thickness=0.4,
                          color=C_BORDER, spaceBefore=top*mm, spaceAfter=bot*mm)
    def sp(n=4): return Spacer(1, n*mm)
    def lbl(t): return Paragraph(t.upper(), SL)
    def val(t): return Paragraph(t, SV)
    def italic(t): return Paragraph(t, SI)

    # ── Swatch table (fixed-size cells, no overlap) ──
    def swatch_table(palette):
        CELL = 22*mm
        GAP  = 2*mm
        n    = len(palette)
        total_w = n * CELL + (n-1) * GAP
        # Clamp to page width
        if total_w > PAGE_W:
            CELL = (PAGE_W - (n-1)*GAP) / n

        rows_color = []
        rows_name  = []
        rows_hex   = []
        col_w      = []
        for c in palette:
            hex_val = c["hex"]
            lum = sum(int(hex_val.lstrip("#")[i*2:i*2+2], 16) * w
                      for i, w in enumerate([0.299, 0.587, 0.114]))
            fg = rl_colors.HexColor("#ffffff") if lum < 140 else rl_colors.HexColor("#2e1f14")
            cell_para = Paragraph(
                f'<font color="{"#ffffff" if lum<140 else "#2e1f14"}" size="6">{hex_val.upper()}</font>',
                S("sc", alignment=TA_CENTER, leading=8, spaceAfter=0))
            rows_color.append(cell_para)
            rows_name.append(Paragraph(c["name"], S("sn", fontSize=7, textColor=C_MID,
                             alignment=TA_CENTER, leading=9, spaceAfter=0)))
            col_w.append(CELL)

        tbl = Table([rows_color, rows_name], colWidths=col_w,
                    rowHeights=[18*mm, 8*mm])
        style = [
            ("ALIGN",     (0,0), (-1,-1), "CENTER"),
            ("VALIGN",    (0,0), (n-1,0), "MIDDLE"),
            ("VALIGN",    (0,1), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,0), (-1,0),
             [rl_colors.HexColor(c["hex"]) for c in palette]),
            ("ROUNDEDCORNERS", [6, 6, 6, 6]),
            ("TOPPADDING",  (0,0), (-1,0), 5),
            ("BOTTOMPADDING",(0,0),(-1,0), 5),
            ("TOPPADDING",  (0,1), (-1,1), 3),
            ("BOTTOMPADDING",(0,1),(-1,1), 0),
        ]
        tbl.setStyle(TableStyle(style))
        return tbl

    # ── Build story ──
    story = []

    # Header
    story += [sp(2), Paragraph("TINTA", ST),
              Paragraph("AI  ·  PERSONAL COLOUR ANALYSIS", SS), hr(4, 6)]

    # Season hero
    story += [
        KeepTogether([
            Paragraph(season, SH),
            Paragraph(f"{pred} Undertone  ·  {depth} Depth", SD2),
            sp(2),
            Paragraph(info["description"], SDE),
        ])
    ]
    if secondary:
        story += [Paragraph(
            f"Secondary season: <b>{secondary['info']['season']}</b> — you sit between these two palettes.",
            S("sec", fontSize=9, textColor=C_DIM, alignment=TA_CENTER)), sp(1)]

    # Stats table
    story += [sp(2)]
    stats = Table(
        [["Confidence", "Model Accuracy", "Detected Skin"],
         [f"{conf:.0f}%", f"{model_acc*100:.0f}%", hex_skin.upper()]],
        colWidths=[PAGE_W/3]*3
    )
    stats.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), rl_colors.HexColor("#fdf0e6")),
        ("BACKGROUND",    (0,1), (-1,1), C_CARD),
        ("TEXTCOLOR",     (0,0), (-1,0), C_DIM),
        ("TEXTCOLOR",     (0,1), (-1,1), C_DARK),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica"),
        ("FONTNAME",      (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, C_BORDER),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [rl_colors.HexColor("#fdf0e6"), C_CARD]),
    ]))
    story += [stats, hr()]

    # Best colours
    story += [KeepTogether([lbl("Your Best Colours"), sp(2), swatch_table(info["palette"]), sp(2)])]
    story += [val("  ·  ".join(f'{c["name"]}  {c["hex"].upper()}' for c in info["palette"])), hr()]

    # Avoid
    story += [KeepTogether([lbl("Colours to Avoid"), sp(2), swatch_table(info["avoid"]), sp(2)]), hr()]

    # Makeup
    story += [lbl("Makeup Recommendations"), sp(1)]
    for cat, items in info["makeup"].items():
        line = f"<b>{cat}:</b>  " + "   ·   ".join(
            f'{i["name"]} <font color="#b09070">({i["hex"].upper()})</font>' for i in items)
        story += [Paragraph(line, S("mk", fontSize=9, textColor=C_DARK, leading=14, spaceAfter=5))]
    story += [hr()]

    # Clothing
    story += [lbl("Clothing & Style"), sp(1),
              val("<b>Best colours:</b>  " + "  ·  ".join(info["clothing_tags"])),
              val(f"<b>Fabrics:</b>  {info['fabrics']}"),
              val(f"<b>Patterns:</b>  {info['patterns']}"),
              italic(f"Style tip: {info['style_tip']}"), hr()]

    # Metals
    story += [lbl("Jewellery & Metals"), sp(1),
              Paragraph(info["metals"], S("jw", fontName="Helvetica-Bold", fontSize=14,
                        textColor=C_DARK, spaceAfter=4, leading=18)),
              val(info["metals_why"]), hr()]

    # Why / Personal
    story += [lbl("Why This Works For You"), sp(1), val(info["why"]), hr()]
    story += [lbl("A Note Just For You"), sp(1),
              italic(f'"{info["personal_msg"]}"'), hr()]

    # Footer
    ts = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    story += [sp(4),
              Paragraph(f"Generated by Tinta  ·  {ts}", SF),
              Paragraph("Results are a guide. Professional colour draping gives maximum precision.", SF)]

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ═══════════════════════════════════════════════════════════════
# COLOR OVERLAY
# ═══════════════════════════════════════════════════════════════
def make_color_overlay(pil_image, palette):
    img = pil_image.convert("RGB").copy()
    w, h = img.size
    strip_h = max(int(h * 0.08), 18)
    n, sw = len(palette), w // len(palette)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    for i, c in enumerate(palette):
        hx = c["hex"].lstrip("#")
        rgb = tuple(int(hx[j*2:j*2+2], 16) for j in range(3))
        draw.rectangle([i*sw, h-strip_h, (i+1)*sw if i<n-1 else w, h], fill=rgb)
    return Image.blend(img, overlay, 0.6)

# ═══════════════════════════════════════════════════════════════
# SEASON DATA (Complete 9-season system)
# ═══════════════════════════════════════════════════════════════
SD = {
  ("Warm","Light"):{"season":"Spring","emoji":"✦","badge_bg":"#d4a040","badge_fg":"#1a0e00","description":"Clear, warm, and luminous. You carry the energy of golden-hour light — fresh, vibrant, and naturally radiant.","palette":[{"name":"Peach","hex":"#FFAD90"},{"name":"Coral","hex":"#FF6B6B"},{"name":"Warm Yellow","hex":"#FFD166"},{"name":"Mint","hex":"#06C090"},{"name":"Ivory","hex":"#FFF0D4"},{"name":"Gold","hex":"#E8B830"}],"avoid":[{"name":"Black","hex":"#1a1a1a"},{"name":"Burgundy","hex":"#800020"},{"name":"Charcoal","hex":"#36454F"},{"name":"Navy","hex":"#001F5B"}],"metals":"Gold & Rose Gold","metals_why":"They mirror the warmth in your skin — creating a sun-kissed harmony that looks completely natural on you.","makeup":{"Foundation":[{"name":"Warm Beige","hex":"#D4A574"},{"name":"Ivory Warm","hex":"#EDD5A3"}],"Lipstick":[{"name":"Coral Red","hex":"#FF6B6B"},{"name":"Warm Peach","hex":"#FFAD90"},{"name":"Apricot","hex":"#F4A460"}],"Eyeshadow":[{"name":"Warm Brown","hex":"#8B5E3C"},{"name":"Champagne","hex":"#E8C890"},{"name":"Terracotta","hex":"#CC7A4A"}],"Blush":[{"name":"Peach","hex":"#FFAD90"},{"name":"Apricot","hex":"#F4A460"}]},"clothing_tags":["Peach","Coral","Ivory","Warm Yellow","Mint","Camel","Light Gold"],"fabrics":"Soft cottons, silk blends, lightweight linen — breathable and natural.","patterns":"Delicate florals, watercolour prints, small geometric in warm tones.","style_tip":"Monochromatic warm outfits elongate you beautifully. Try an ivory-to-peach gradient look.","why":"Your golden undertones amplify alongside warm, clear shades — they create radiance from your skin outward rather than competing with it.","personal_msg":"There's something effortlessly luminous about you. Your coloring has a rare clarity that comes alive in warmth and light. The right palette doesn't just make you look good — it makes you look like yourself, turned all the way up."},
  ("Warm","Medium"):{"season":"Autumn","emoji":"✦","badge_bg":"#a03800","badge_fg":"#fde8c0","description":"Rich, earthy, and deeply warm. You carry the richness of harvest — golden, grounded, and magnetic.","palette":[{"name":"Burnt Orange","hex":"#CC5500"},{"name":"Olive","hex":"#7a7a00"},{"name":"Mustard","hex":"#D4A800"},{"name":"Rust","hex":"#B7410E"},{"name":"Camel","hex":"#C19A6B"},{"name":"Forest","hex":"#1a7a1a"}],"avoid":[{"name":"Pastel Pink","hex":"#FFB6C1"},{"name":"Icy Blue","hex":"#99C5C4"},{"name":"Silver","hex":"#C0C0C0"},{"name":"Lavender","hex":"#E6E6FA"}],"metals":"Gold, Bronze & Copper","metals_why":"Earthy metals echo the warm depth of your complexion — rich, never garish.","makeup":{"Foundation":[{"name":"Golden Tan","hex":"#C8874A"},{"name":"Warm Honey","hex":"#D4956A"}],"Lipstick":[{"name":"Brick Red","hex":"#B7410E"},{"name":"Warm Nude","hex":"#C19A6B"},{"name":"Terracotta","hex":"#CC7A4A"}],"Eyeshadow":[{"name":"Deep Olive","hex":"#556B2F"},{"name":"Burnt Sienna","hex":"#C87040"},{"name":"Rich Brown","hex":"#7B3F00"}],"Blush":[{"name":"Copper","hex":"#B87333"},{"name":"Rust Rose","hex":"#B7410E"}]},"clothing_tags":["Burnt Orange","Olive","Camel","Mustard","Rust","Forest Green","Chocolate"],"fabrics":"Suede, corduroy, raw silk, chunky knits — textures with real depth.","patterns":"Earthy plaids, abstract botanical, tortoiseshell, leopard.","style_tip":"Layering within your palette is your superpower. A mustard turtleneck under a forest green jacket is pure autumn magic.","why":"Your warm medium depth craves pigment-rich earthy tones. Muted-warm shades match the sophistication already in your skin.","personal_msg":"You have an effortless warmth that draws people in without trying. The earthy, spiced palette of Autumn was practically made for someone like you — deep enough to hold your complexity, warm enough to match your energy."},
  ("Warm","Deep"):{"season":"Deep Autumn","emoji":"✦","badge_bg":"#4a1a00","badge_fg":"#f0c890","description":"Intense, bold, and powerfully warm. Made for depth — saturated dark earth tones that match your commanding presence.","palette":[{"name":"Deep Olive","hex":"#4a5e20"},{"name":"Terracotta","hex":"#D06050"},{"name":"Dark Gold","hex":"#9a7000"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Burgundy","hex":"#800020"},{"name":"Warm Brown","hex":"#A0522D"}],"avoid":[{"name":"Pastel Yellow","hex":"#EEEE80"},{"name":"Baby Blue","hex":"#89CFF0"},{"name":"Neon Green","hex":"#39FF14"},{"name":"Cool Gray","hex":"#909090"}],"metals":"Gold, Bronze & Warm Copper","metals_why":"Rich deep metals ground your complexion and add luxurious dimension.","makeup":{"Foundation":[{"name":"Deep Warm","hex":"#8B5A2B"},{"name":"Mahogany","hex":"#A0522D"}],"Lipstick":[{"name":"Deep Berry","hex":"#800020"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Warm Wine","hex":"#722F37"}],"Eyeshadow":[{"name":"Gold","hex":"#9a7000"},{"name":"Deep Plum","hex":"#673147"},{"name":"Forest","hex":"#1a7a1a"}],"Blush":[{"name":"Deep Peach","hex":"#C67C52"},{"name":"Bronze","hex":"#A07030"}]},"clothing_tags":["Chocolate","Burgundy","Dark Olive","Terracotta","Dark Gold","Deep Teal"],"fabrics":"Rich velvets, heavyweight silk, structured leather — fabrics with presence.","patterns":"Bold animal prints, abstract art prints, rich dark plaids.","style_tip":"Don't fear head-to-toe depth. A full chocolate look or all-burgundy ensemble is your signature move.","why":"Your deep warm coloring craves saturation. Light or cool colors fade against your richness — bold and earthy lets you be fully seen.","personal_msg":"There's a gravitational pull to your coloring — bold, warm, and impossible to ignore. Deep Autumn is rare and extraordinary. Own the depth that's already there; it's your greatest style asset."},
  ("Cool","Light"):{"season":"Summer","emoji":"✦","badge_bg":"#4a7090","badge_fg":"#e0f0ff","description":"Soft, cool, and quietly elegant. A dreamlike delicacy — misty, silvery, and effortlessly refined.","palette":[{"name":"Dusty Rose","hex":"#C89080"},{"name":"Lavender","hex":"#9a70c0"},{"name":"Powder Blue","hex":"#90c0cc"},{"name":"Mauve","hex":"#b090b0"},{"name":"Soft Gray","hex":"#a0a0b0"},{"name":"Rose Beige","hex":"#D8B8A8"}],"avoid":[{"name":"Orange","hex":"#E07020"},{"name":"Rust","hex":"#B7410E"},{"name":"Warm Yellow","hex":"#D4A800"},{"name":"Olive","hex":"#7a7a00"}],"metals":"Silver & White Gold","metals_why":"Cool metals reflect the delicate rosiness in your skin and feel effortlessly natural.","makeup":{"Foundation":[{"name":"Cool Porcelain","hex":"#E8CCC0"},{"name":"Pink Beige","hex":"#D8B8A8"}],"Lipstick":[{"name":"Dusty Rose","hex":"#B07070"},{"name":"Berry Pink","hex":"#B01070"},{"name":"Rose Petal","hex":"#C05080"}],"Eyeshadow":[{"name":"Lavender","hex":"#9a70c0"},{"name":"Dusty Plum","hex":"#7a4080"},{"name":"Soft Gray","hex":"#8888a0"}],"Blush":[{"name":"Rose","hex":"#E08080"},{"name":"Mauve","hex":"#b090b0"}]},"clothing_tags":["Dusty Rose","Soft Lavender","Powder Blue","Mauve","Soft Gray","Rose White"],"fabrics":"Chiffon, satin, soft cashmere — ethereal fabrics that float.","patterns":"Soft watercolour florals, delicate ditsy prints, ombre, subtle stripe.","style_tip":"Tone-on-tone muted looks are magical on you. Mauve trousers + dusty rose top + silver jewellery = effortless.","why":"Your soft cool undertone shines with muted dusty tones. Bright or warm colors overpower your delicate coloring — softness lets your natural elegance lead.","personal_msg":"There's a soft, understated quality to your beauty that gets more striking the longer you look. Summer is the season of quiet confidence — you don't need to announce yourself. The right palette lets that grace simply breathe."},
  ("Cool","Medium"):{"season":"Winter","emoji":"✦","badge_bg":"#1a2880","badge_fg":"#c0d8ff","description":"Bold, cool, and striking. High contrast is your superpower — vivid jewel shades create looks you won't forget.","palette":[{"name":"Royal Blue","hex":"#2850cc"},{"name":"Magenta","hex":"#b00060"},{"name":"Emerald","hex":"#006840"},{"name":"Pure White","hex":"#E8F0F8"},{"name":"True Red","hex":"#c00020"},{"name":"Ice Pink","hex":"#E080A0"}],"avoid":[{"name":"Warm Beige","hex":"#E8D8B0"},{"name":"Mustard","hex":"#D4A800"},{"name":"Camel","hex":"#C19A6B"},{"name":"Burnt Orange","hex":"#CC5500"}],"metals":"Silver, Platinum & White Gold","metals_why":"Crisp cool metals amplify your natural contrast and look undeniably sharp.","makeup":{"Foundation":[{"name":"Cool Beige","hex":"#C8A880"},{"name":"Neutral Med","hex":"#B89870"}],"Lipstick":[{"name":"True Red","hex":"#c00020"},{"name":"Berry","hex":"#800050"},{"name":"Raspberry","hex":"#b01050"}],"Eyeshadow":[{"name":"Charcoal","hex":"#384050"},{"name":"Sapphire","hex":"#0a48a0"},{"name":"Deep Plum","hex":"#673147"}],"Blush":[{"name":"Cool Pink","hex":"#d05090"},{"name":"Berry","hex":"#b01060"}]},"clothing_tags":["True White","True Black","Royal Blue","Magenta","Emerald","Ice Pink","True Red"],"fabrics":"Structured cotton, crisp silk, tailored wool — clothes with architecture.","patterns":"Graphic prints, bold stripes, strong geometrics, classic houndstooth.","style_tip":"Embrace contrast. Black + white or full jewel-tone — you're one of the rare types who looks incredible in stark, high-contrast combinations.","why":"Your cool undertone and medium depth create natural contrast. Vivid shades work with that contrast — murky or warm tones cancel it out.","personal_msg":"You have the kind of coloring that stops people mid-sentence. Winter is bold, precise, and high-impact — and so are you. When you dress in your true colors, there's a crispness to your whole look that reads as effortlessly commanding."},
  ("Cool","Deep"):{"season":"Deep Winter","emoji":"✦","badge_bg":"#060812","badge_fg":"#a0c0d8","description":"Dramatic, intense, and powerfully cool. Built for depth and richness — the jeweled darkness of midnight.","palette":[{"name":"Burgundy","hex":"#800020"},{"name":"Cobalt","hex":"#0040a0"},{"name":"Plum","hex":"#673147"},{"name":"Charcoal","hex":"#364050"},{"name":"Deep Teal","hex":"#007070"},{"name":"Onyx","hex":"#1a1a1a"}],"avoid":[{"name":"Light Beige","hex":"#E8E0C8"},{"name":"Warm Orange","hex":"#E07020"},{"name":"Gold","hex":"#D4A800"},{"name":"Peach","hex":"#FFAD90"}],"metals":"Silver, Platinum & Dark Rhodium","metals_why":"Cool dark metals match your deep cool richness and add a moody, dramatic dimension.","makeup":{"Foundation":[{"name":"Deep Cool","hex":"#706050"},{"name":"Ebony Cool","hex":"#604838"}],"Lipstick":[{"name":"Deep Plum","hex":"#4B0082"},{"name":"Oxblood","hex":"#800020"},{"name":"Dark Berry","hex":"#501880"}],"Eyeshadow":[{"name":"Midnight","hex":"#181870"},{"name":"Deep Plum","hex":"#673147"},{"name":"Graphite","hex":"#383838"}],"Blush":[{"name":"Deep Rose","hex":"#904060"},{"name":"Berry","hex":"#800050"}]},"clothing_tags":["True Black","Cobalt","Plum","Burgundy","Deep Teal","Charcoal","Deep Navy"],"fabrics":"Velvet, heavyweight silk, luxe jersey — fabrics with gravitas.","patterns":"Bold abstract, deep jewel-tone florals, dramatic geometric.","style_tip":"All-black is your baseline — you pull it off like no one else. Add one jewel-toned piece (cobalt or burgundy) for dimension.","why":"Your deep cool complexion craves rich dark cool-toned depth. Pale or warm tones look washed against your richness.","personal_msg":"There's an intensity to your coloring that's completely arresting. Deep Winter is the rarest season — dramatic, cool, and undeniably powerful. You were made for dark jeweled depth. Lean into it completely."},
  ("Neutral","Light"):{"season":"Neutral Light","emoji":"✦","badge_bg":"#507040","badge_fg":"#d8f0c0","description":"Versatile, fresh, and beautifully balanced. You carry the unique gift of flexibility — soft tones from both warm and cool families suit you.","palette":[{"name":"Blush","hex":"#cc5070"},{"name":"Soft Teal","hex":"#30909a"},{"name":"Warm Cream","hex":"#E8DCA0"},{"name":"Stone Gray","hex":"#888078"},{"name":"Sage","hex":"#80a060"},{"name":"Dusty Purple","hex":"#8a6898"}],"avoid":[{"name":"Neon Yellow","hex":"#BBEE00"},{"name":"Harsh Black","hex":"#000000"},{"name":"Pure White","hex":"#FFFFFF"},{"name":"Neon Orange","hex":"#FF6500"}],"metals":"Gold & Silver — both work","metals_why":"Your balanced undertone gives you the rare ability to wear both — mix and match freely.","makeup":{"Foundation":[{"name":"Neutral Ivory","hex":"#E0C8A8"},{"name":"Warm Light","hex":"#D8C098"}],"Lipstick":[{"name":"Blush Pink","hex":"#cc5070"},{"name":"Soft Coral","hex":"#E07860"},{"name":"Rosy Nude","hex":"#b07070"}],"Eyeshadow":[{"name":"Soft Taupe","hex":"#908070"},{"name":"Sage","hex":"#80a060"},{"name":"Dusty Mauve","hex":"#8a6898"}],"Blush":[{"name":"Soft Blush","hex":"#E0A0A8"},{"name":"Peach Pink","hex":"#E8A080"}]},"clothing_tags":["Sage Green","Soft Teal","Blush","Warm Cream","Stone Gray","Dusty Purple"],"fabrics":"Soft knits, light linen, airy cotton — understated quality.","patterns":"Small-scale prints, subtle texture, tonal patterns.","style_tip":"You're one of the few who can mix a warm scarf with a cool outfit. Use this gift — eclectic colour mixing is uniquely yours.","why":"Your neutral undertone means neither warm nor cool shades clash. Soft balanced tones bring harmony — extremes tip the balance.","personal_msg":"Having a neutral undertone is genuinely rare — you exist in a beautiful in-between space that most people never access. You have styling freedoms others would envy. Use them."},
  ("Neutral","Medium"):{"season":"Neutral Medium","emoji":"✦","badge_bg":"#6a5040","badge_fg":"#f0e0c8","description":"Balanced and beautifully adaptable. Earthy mid-tones from both warm and cool families suit you naturally.","palette":[{"name":"Sage","hex":"#80a060"},{"name":"Terracotta","hex":"#c06048"},{"name":"Slate Blue","hex":"#5a7088"},{"name":"Caramel","hex":"#b07830"},{"name":"Muted Teal","hex":"#3a8888"},{"name":"Warm Taupe","hex":"#907060"}],"avoid":[{"name":"Neon","hex":"#39FF14"},{"name":"Pure White","hex":"#F8F8F8"},{"name":"Cool Pastels","hex":"#90c0d0"},{"name":"Harsh Black","hex":"#000000"}],"metals":"Gold & Silver — both work","metals_why":"You can mix metals beautifully — try gold rings with a silver watch for an eclectic edge.","makeup":{"Foundation":[{"name":"Neutral Beige","hex":"#c09060"},{"name":"Warm Medium","hex":"#b08040"}],"Lipstick":[{"name":"Terracotta","hex":"#b06838"},{"name":"Muted Berry","hex":"#885058"},{"name":"Warm Mauve","hex":"#986070"}],"Eyeshadow":[{"name":"Warm Taupe","hex":"#907060"},{"name":"Muted Teal","hex":"#3a8888"},{"name":"Caramel","hex":"#b07830"}],"Blush":[{"name":"Warm Peach","hex":"#E09070"},{"name":"Soft Coral","hex":"#D08060"}]},"clothing_tags":["Sage","Terracotta","Caramel","Muted Teal","Slate Blue","Warm Taupe"],"fabrics":"Relaxed linen, soft leather, woven textures — grounded and natural.","patterns":"Earthy abstract, botanical prints, subtle plaid.","style_tip":"Mix teal with terracotta — it works because your undertone bridges both worlds. You can build an eclectic wardrobe others can't.","why":"Your balanced medium-depth coloring is versatile. Muted earthy tones from both families complement your adaptability without clashing.","personal_msg":"Being Neutral Medium means you're the chameleon of the colour world — you can walk into almost any palette and make it work. That's a creative superpower. Choose tones with depth; your coloring has richness that deserves to be met in kind."},
  ("Neutral","Deep"):{"season":"Neutral Deep","emoji":"✦","badge_bg":"#201808","badge_fg":"#e0c890","description":"Rich, grounded, and powerfully balanced. Deep sophisticated tones from warm and cool families suit your remarkable depth.","palette":[{"name":"Forest","hex":"#1a6a1a"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Navy","hex":"#001850"},{"name":"Berry","hex":"#783050"},{"name":"Deep Teal","hex":"#005050"},{"name":"Warm Brown","hex":"#8a4020"}],"avoid":[{"name":"Pastel Yellow","hex":"#EEEE80"},{"name":"Baby Pink","hex":"#F4C2C2"},{"name":"Pale Pastels","hex":"#D0D0F0"},{"name":"Light Beige","hex":"#E8E0C8"}],"metals":"Gold & Silver — both work","metals_why":"Deep neutrals carry both metals effortlessly — use gold for warmth, silver for edge.","makeup":{"Foundation":[{"name":"Deep Neutral","hex":"#785030"},{"name":"Rich Ebony","hex":"#684020"}],"Lipstick":[{"name":"Deep Berry","hex":"#783050"},{"name":"Chocolate","hex":"#7B3F00"},{"name":"Dark Plum","hex":"#673147"}],"Eyeshadow":[{"name":"Forest","hex":"#1a6a1a"},{"name":"Deep Teal","hex":"#005050"},{"name":"Rich Brown","hex":"#7B3F00"}],"Blush":[{"name":"Deep Rose","hex":"#904060"},{"name":"Rich Berry","hex":"#783050"}]},"clothing_tags":["Forest Green","Chocolate","Navy","Berry","Deep Teal","Warm Brown","Charcoal"],"fabrics":"Rich denim, structured wool, luxe cotton — depth and presence.","patterns":"Bold graphic prints, jewel-toned patterns, strong geometrics.","style_tip":"Deep saturated colors are your signature. Use rich contrasts — forest green + chocolate, navy + deep berry.","why":"Your deep neutral coloring handles rich saturated tones from both families. Pale pastels disappear against your richness — depth meets depth.","personal_msg":"You have one of the most grounded, versatile, and powerful color profiles possible. Deep Neutral is commanding — it holds warmth and coolness at once with ease. You don't need to choose a side. You're the rare person who gets both worlds."},
}

# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 1: MANUAL COLOUR PICKER FALLBACK
# (defined after SD so the dict is available at call time cleanly)
# ═══════════════════════════════════════════════════════════════
def manual_colour_picker_analysis(skin_hex, eye_hex, hair_hex):
    """Fallback analysis when user can't take a good photo"""
    skin_rgb = tuple(int(skin_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = skin_rgb

    brightness = (r + g + b) / 3
    if brightness > 165:
        depth = "Light"
    elif brightness < 115:
        depth = "Deep"
    else:
        depth = "Medium"

    arr = np.uint8([[[b, g, r]]])
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2Lab)[0][0]
    L = float(lab[0]) / 255.0 * 100
    bl = float(lab[2]) - 128
    if abs(bl) < 1e-3:
        bl = 1e-3
    ita = np.degrees(np.arctan((L - 50) / bl))

    if ita > 28:
        pred = "Warm"
    elif ita < 10:
        pred = "Cool"
    else:
        pred = "Neutral"

    conf = 85.0
    info = SD.get((pred, depth), SD[("Neutral", "Medium")])

    return {"r": r, "g": g, "b": b, "pred": pred, "conf": conf,
            "depth": depth, "info": info, "manual": True}


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE 3: TOP 3 PROBABILITIES DISPLAY
# (fixed: removed duplicate st.bar_chart; uses custom HTML bars only)
# ═══════════════════════════════════════════════════════════════
def show_top_3_probs(probs, classes, pred, depth):
    """Display top 3 season probabilities as styled HTML bars (no duplicate chart)."""
    if probs is None:
        return

    top3_idx = np.argsort(probs)[::-1][:3]

    season_names = []
    season_probs = []
    for idx in top3_idx:
        undertone = classes[idx]
        info = SD.get((undertone, depth), SD[("Neutral", "Medium")])
        season_names.append(f"{undertone} · {info['season']}")
        season_probs.append(probs[idx] * 100)

    bars_html = ""
    colors_map = ["#c97b3a", "#7a6a58", "#c9a882"]
    for i, (season, prob) in enumerate(zip(season_names, season_probs)):
        bars_html += (
            f'<div style="margin:10px 0">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#7a5c44;margin-bottom:4px">'
            f'<span>{season}</span><span>{prob:.1f}%</span>'
            f'</div>'
            f'<div style="background:#ede0d4;border-radius:10px;height:8px;overflow:hidden">'
            f'<div style="background:{colors_map[i]};width:{prob:.1f}%;height:100%;border-radius:10px"></div>'
            f'</div></div>'
        )

    st.markdown('<div class="sec-div">top 3 season probabilities</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card" style="margin-top:12px">'
        '<div class="lbl">Season Distribution</div>'
        '<div class="info-val" style="font-size:12px;margin-bottom:12px">Your colour profile sits between seasons — here\'s the full distribution:</div>'
        + bars_html +
        '</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
# REAL DATASET LOADER (age_gender.csv / UTKFace format)
# Columns: age, ethnicity, gender, img_name, pixels
# Ethnicity: 0=White(fair) 1=Black(deep) 2=Asian(olive) 3=Indian(dark) 4=Other(medium)
# ═══════════════════════════════════════════════════════════════
IMG_SIZE = 48
ETHNICITY_MAP = {0: "fair", 1: "deep", 2: "olive", 3: "dark", 4: "medium"}
TONE_TO_UNDERTONE = {"fair": "Cool", "deep": "Warm", "olive": "Neutral", "dark": "Warm", "medium": "Neutral"}
TONE_TO_DEPTH = {"fair": "Light", "deep": "Deep", "olive": "Medium", "dark": "Deep", "medium": "Medium"}

def _pixels_to_rgb(pixel_str):
    vals = np.array(pixel_str.strip().split(), dtype=np.float32)
    if len(vals) != IMG_SIZE * IMG_SIZE:
        return None
    img = vals.reshape(IMG_SIZE, IMG_SIZE)
    r = float(np.clip(img * 1.10, 0, 255).mean())
    g = float(np.clip(img * 0.85, 0, 255).mean())
    b = float(np.clip(img * 0.75, 0, 255).mean())
    return r, g, b

def _load_real_dataset(csv_path):
    """Load age_gender.csv and return a DataFrame with R, G, B, Undertone columns."""
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        rgb = _pixels_to_rgb(str(row["pixels"]))
        if rgb is None:
            continue
        r, g, b = rgb
        tone = ETHNICITY_MAP.get(int(row["ethnicity"]), "medium")
        undertone = TONE_TO_UNDERTONE[tone]
        rows.append({"R": r, "G": g, "B": b, "Undertone": undertone})
    return pd.DataFrame(rows)

@st.cache_resource
def train_model():
    import os
    data = None

    # Priority 1: real UTKFace / age_gender dataset
    for real_path in ["dataset/age_gender.csv", "age_gender.csv"]:
        if os.path.exists(real_path):
            try:
                data = _load_real_dataset(real_path)
                break
            except Exception:
                pass

    # Priority 2: synthetic skin_undertone_dataset.csv
    if data is None:
        for syn_path in ["skin_undertone_dataset.csv", "dataset/skin_undertone_dataset.csv"]:
            if os.path.exists(syn_path):
                try:
                    data = pd.read_csv(syn_path)
                    # Normalise column names
                    rename = {}
                    for c in data.columns:
                        cl = c.lower().strip()
                        if cl in ("undertone", "label", "season", "class", "target") and "Undertone" not in rename.values():
                            rename[c] = "Undertone"
                        elif cl in ("r", "red") and "R" not in rename.values():
                            rename[c] = "R"
                        elif cl in ("g", "green") and "G" not in rename.values():
                            rename[c] = "G"
                        elif cl in ("b", "blue") and "B" not in rename.values():
                            rename[c] = "B"
                    if rename:
                        data = data.rename(columns=rename)
                    break
                except Exception:
                    pass

    if data is None or "Undertone" not in data.columns:
        st.error("No dataset found. Place age_gender.csv in a 'dataset/' folder OR skin_undertone_dataset.csv in the project folder.")
        st.stop()

    data = engineer_df(data)
    X, y = data[FEATURES], data["Undertone"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    mdl = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    mdl.fit(sc.fit_transform(Xtr), ytr)
    acc = mdl.score(sc.transform(Xte), yte)
    return mdl, sc, acc

model, scaler, model_acc = train_model()

# ═══════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════
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
    """Detect face with OpenCV Haar cascade and sample skin pixels."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, fw, fh = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    pixels = []
    ps = 8
    sample_points = [
        (x + fw//2,        y + int(fh*0.20)),
        (x + int(fw*0.25), y + int(fh*0.50)),
        (x + int(fw*0.75), y + int(fh*0.50)),
        (x + int(fw*0.30), y + int(fh*0.35)),
        (x + int(fw*0.70), y + int(fh*0.35)),
        (x + fw//2,        y + int(fh*0.40)),
    ]
    for cx, cy in sample_points:
        patch = img_bgr[max(0, cy-ps):min(h, cy+ps), max(0, cx-ps):min(w, cx+ps)]
        if patch.size > 0:
            pixels.append(np.mean(patch.reshape(-1, 3), axis=0))
    if not pixels:
        return None
    b, g, r = np.mean(pixels, axis=0)
    return float(r), float(g), float(b)

def is_dark(hex_c):
    h=hex_c.lstrip("#")
    r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return (0.299*r+0.587*g+0.114*b)<128

def swatches(colors, avoid=False):
    items=""
    for c in colors:
        tc = "#fff" if is_dark(c["hex"]) else "#222"
        cls = "sw-avoid" if avoid else "sw"
        xm = '<span style="position:absolute;top:6px;right:7px;font-size:10px;color:#e05050;font-weight:700">✕</span>' if avoid else ""
        items += f'<div class="{cls}" style="background:{c["hex"]};color:{tc};position:relative">{xm}{c["name"]}</div>'
    return f'<div class="swatch-row">{items}</div>'

def chips(items):
    out="".join(f'<div class="chip"><div class="dot" style="background:{i["hex"]}"></div>{i["name"]}</div>' for i in items)
    return f'<div class="chip-row">{out}</div>'

def tags(items):
    out="".join(f'<span class="tag">{t}</span>' for t in items)
    return f'<div class="tag-row">{out}</div>'

def conf_bar(c):
    col="#6a9a70" if c>=80 else "#c97b3a" if c>=65 else "#a05040"
    return f'<div class="conf-track"><div style="height:3px;border-radius:99px;background:{col};width:{c:.0f}%"></div></div>'

def hex_chips_with_copy(palette):
    items=""
    for c in palette:
        hx = c["hex"].upper()
        items += (
            f'<div class="chip" style="flex-direction:column;align-items:flex-start;gap:4px;padding:10px 14px">'
            f'<div style="display:flex;align-items:center;gap:8px;width:100%;justify-content:space-between">'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div class="dot" style="background:{c["hex"]};width:16px;height:16px"></div>'
            f'<span style="font-size:12px;color:#5a3a22;font-weight:500">{c["name"]}</span>'
            f'</div>'
            f'<span onclick="navigator.clipboard.writeText(\'{hx}\').then(()=>{{this.innerText=\'✓\';setTimeout(()=>this.innerText=\'copy\',1200)}})" '
            f'style="cursor:pointer;font-size:9px;color:#b09070;letter-spacing:0.1em;padding:2px 8px;'
            f'border:1px solid #2a221a;border-radius:6px;user-select:none">copy</span>'
            f'</div>'
            f'<span style="font-size:10px;color:#c9a882;font-family:monospace;letter-spacing:0.08em">{hx}</span>'
            f'</div>'
        )
    return f'<div class="chip-row">{items}</div>'

def get_secondary(probs, classes, primary_pred, depth):
    sorted_idx = np.argsort(probs)[::-1]
    if len(sorted_idx) < 2: return None
    top_p = probs[sorted_idx[0]]
    sec_p = probs[sorted_idx[1]]
    sec_cl = classes[sorted_idx[1]]
    if (top_p - sec_p) > 0.20: return None
    sec_info = SD.get((sec_cl, depth), None)
    if sec_info is None: return None
    return {"undertone": sec_cl, "prob": sec_p*100, "info": sec_info}

# ═══════════════════════════════════════════════════════════════
# CORE ANALYSIS
# ═══════════════════════════════════════════════════════════════
def analyse(image, wb_correction=True):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    quality_ok, quality_status = check_image_quality(np.array(image))
    if wb_correction:
        img_bgr = white_balance(img_bgr)
    result = extract_landmarks(img_bgr)
    if result is None:
        # Fallback: OpenCV HSV skin segmentation on full image
        H, W = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([25, 180, 255]))
        skin = img_bgr[mask > 0]
        if len(skin) < 50:
            return None, "no_face", quality_status
        avg = np.mean(skin, axis=0)
        r, g, b = float(avg[2]), float(avg[1]), float(avg[0])
    else:
        r, g, b = result
    feats = engineer_features(r, g, b)
    scaled = scaler.transform([feats])
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]
    conf = float(max(probs))*100
    ita = ita_undertone(r, g, b)
    if ita != pred and conf < 75:
        pred = ita
        conf = max(conf, 55.0)
    depth = classify_depth((r+g+b)/3)
    info = SD.get((pred, depth), SD[("Neutral","Medium")])
    secondary = get_secondary(probs, model.classes_, pred, depth)
    return {"r":r,"g":g,"b":b,"pred":pred,"conf":conf,"depth":depth,
            "info":info,"secondary":secondary, "probs": probs}, "ok", quality_status

def analyse_multi(images, wb_correction=True):
    all_rgb, quality_statuses = [], []
    for img in images:
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _, qs = check_image_quality(np.array(img))
        quality_statuses.append(qs)
        if wb_correction:
            img_bgr = white_balance(img_bgr)
        result = extract_landmarks(img_bgr)
        if result:
            all_rgb.append(list(result))
    if not all_rgb:
        return None, "no_face", quality_statuses[0] if quality_statuses else "ok"
    r = np.mean([x[0] for x in all_rgb])
    g = np.mean([x[1] for x in all_rgb])
    b = np.mean([x[2] for x in all_rgb])
    feats = engineer_features(r, g, b)
    scaled = scaler.transform([feats])
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]
    conf = float(max(probs))*100
    ita = ita_undertone(r, g, b)
    if ita != pred and conf < 75:
        pred = ita; conf = max(conf, 55.0)
    depth = classify_depth((r+g+b)/3)
    info = SD.get((pred, depth), SD[("Neutral","Medium")])
    secondary = get_secondary(probs, model.classes_, pred, depth)
    worst_qs = "too_dark" if "too_dark" in quality_statuses else \
               "too_bright" if "too_bright" in quality_statuses else "ok"
    return {"r":r,"g":g,"b":b,"pred":pred,"conf":conf,"depth":depth,
            "info":info,"secondary":secondary,"n_frames":len(all_rgb), "probs": probs}, "ok", worst_qs

# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════
def render_quality_banner(qs):
    if qs=="too_dark":
        st.markdown("""<div class="card-alert" style="border-left-color:#a05040"><div class="lbl">image quality warning</div>
        <div class="info-val" style="margin-top:6px">Image appears <strong style="color:#c97b3a">too dark</strong> — results may be less accurate. Try brighter natural light.</div></div>""", unsafe_allow_html=True)
    elif qs=="too_bright":
        st.markdown("""<div class="card-alert" style="border-left-color:#a05040"><div class="lbl">image quality warning</div>
        <div class="info-val" style="margin-top:6px">Image appears <strong style="color:#c97b3a">overexposed</strong> — results may be less accurate. Move away from direct harsh light.</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="card-alert" style="border-left-color:#5a8a60"><div class="lbl">image quality</div>
        <div class="info-val" style="margin-top:4px;color:#7aaa80">Good lighting detected — analysis should be accurate.</div></div>""", unsafe_allow_html=True)

def render_retake():
    st.markdown("""<div class="card-hero"><div style="font-size:1.6rem;margin-bottom:6px;opacity:.4">◈</div>
    <div class="season-name" style="font-size:1.5rem">Let's try again</div>
    <div class="season-desc" style="margin-top:10px">Confidence too low for a reliable reading. Work through this checklist then re-upload.</div></div>""", unsafe_allow_html=True)
    steps=[("Face a window","Natural light only — no lamps, no flash."),("Fill the frame","Your face should take up most of the photo."),("Look straight ahead","No angles — camera at eye level."),("Remove glasses","Frames cast shadows that affect the reading."),("No beauty filters","Turn off phone beauty mode and all filters."),("Minimal makeup","Remove heavy foundation if you can.")]
    html="".join(f'<div class="retake-step"><div class="retake-num">{i+1}</div><div class="retake-txt"><strong style="color:#5a3a22">{t}</strong><br>{d}</div></div>' for i,(t,d) in enumerate(steps))
    st.markdown(f'<div class="card" style="margin-top:12px">{html}</div>', unsafe_allow_html=True)

def render_full_result(res, image):
    r,g,b = res["r"],res["g"],res["b"]
    pred = res["pred"]
    conf = res["conf"]
    depth = res["depth"]
    info = res["info"]
    secondary = res.get("secondary")
    hex_skin = "#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))
    n_frames = res.get("n_frames", 1)
    probs = res.get("probs", None)

    overlay_img = make_color_overlay(image, info["palette"])
    c1,c2,c3 = st.columns([3,3,1])
    with c1: st.image(image, use_container_width=True, caption="Your photo")
    with c2: st.image(overlay_img, use_container_width=True, caption="Palette preview")
    with c3:
        st.markdown(f"""<div class="skin-ring">
        <div style="width:46px;height:46px;border-radius:50%;background:{hex_skin};border:2px solid #2a221a;box-shadow:0 0 18px {hex_skin}44"></div>
        <div class="rgb-small">skin<br>tone</div></div>""", unsafe_allow_html=True)

    if n_frames > 1:
        st.markdown(f"""<div class="card-alert" style="border-left-color:#5a8a60;margin-top:10px">
        <div class="lbl">multi-frame analysis</div>
        <div class="info-val" style="margin-top:4px;color:#7aaa80">Averaged across {n_frames} photos — more stable and reliable than a single reading.</div>
        </div>""", unsafe_allow_html=True)

    bar = conf_bar(conf)
    st.markdown(f"""<div class="card-hero">
    <span class="season-pill" style="background:{info['badge_bg']};color:{info['badge_fg']}">{info['emoji']}  {info['season']}</span>
    <div class="season-name">{pred} · {depth}</div>
    <div class="season-sub">Confidence {conf:.0f}%  ·  Model accuracy {model_acc*100:.0f}%  ·  8-feature model</div>
    {bar}
    <div class="season-desc" style="margin-top:22px">{info['description']}</div>
    <div style="margin-top:14px;font-size:10px;color:#e8d8c8;letter-spacing:0.12em">{hex_skin.upper()}  ·  rgb({int(r)}, {int(g)}, {int(b)})</div>
    </div>""", unsafe_allow_html=True)

    if 60<=conf<75:
        st.info("Moderate confidence — a solid starting point. Retaking in natural daylight can sharpen this.")

    if secondary:
        sec_info = secondary["info"]
        st.markdown(f"""<div class="card-secondary">
        <div class="lbl">border profile — you sit between two seasons</div>
        <div style="display:flex;align-items:center;gap:14px;margin-top:10px">
            <span class="season-pill" style="background:{sec_info['badge_bg']};color:{sec_info['badge_fg']};margin:0;font-size:8px">{sec_info['season']}</span>
            <div>
                <div style="font-family:'Cormorant Garamond',serif;font-size:18px;color:#5a3a22">{secondary['undertone']} · {depth}</div>
                <div style="font-size:11px;color:#c9a882;margin-top:2px">Secondary confidence {secondary['prob']:.0f}% — you can also wear from this palette</div>
            </div>
        </div>
        <div style="margin-top:12px">{swatches(sec_info['palette'])}</div>
        </div>""", unsafe_allow_html=True)

    # NEW FEATURE: Show top 3 probabilities
    if probs is not None:
        show_top_3_probs(probs, model.classes_, pred, depth)

    st.markdown('<div class="sec-div">your colors</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{swatches(info["palette"])}</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card" style="margin-top:8px">
    <div class="lbl">hex codes — tap copy to grab any shade</div>
    {hex_chips_with_copy(info["palette"])}</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">colors to avoid</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card" style="border-color:#3a1a1a">
    <div class="lbl" style="margin-bottom:6px">These clash with your undertone and can wash you out</div>
    {swatches(info["avoid"],avoid=True)}</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">makeup</div>', unsafe_allow_html=True)
    mh="".join(f'<div class="lbl" style="margin-top:16px">{lb}</div>{chips(it)}' for lb,it in info["makeup"].items())
    st.markdown(f'<div class="card">{mh}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-div">clothing & style</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card">
    <div class="lbl">best colors to wear</div>{tags(info["clothing_tags"])}
    <div style="border-top:1px solid #ede0d4;margin-top:18px;padding-top:16px"><div class="lbl">fabrics</div><div class="info-val">{info["fabrics"]}</div></div>
    <div style="border-top:1px solid #ede0d4;margin-top:14px;padding-top:14px"><div class="lbl">patterns</div><div class="info-val">{info["patterns"]}</div></div>
    <div style="border-top:1px solid #ede0d4;margin-top:14px;padding-top:14px"><div class="lbl">style tip</div><div class="style-tip">{info["style_tip"]}</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">jewelry & metals</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card"><div class="metals-name">{info["metals"]}</div>
    <div class="info-val" style="margin-top:6px">{info["metals_why"]}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">why this works for you</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card-gold"><div class="lbl">the science</div>
    <div class="info-val" style="margin-top:4px">{info["why"]}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">accuracy & fairness</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card-alert"><div class="lbl">transparency note</div>
    <div class="info-val" style="margin-top:8px">
    <strong style="color:#c97b3a">White balance correction is applied automatically</strong> to reduce yellow indoor and blue LED lighting bias before any analysis is done.<br><br>
    <strong style="color:#c97b3a">Camera sensors vary.</strong> Some phones compress warm tones. If results feel off, try a different device or step outside.<br><br>
    <strong style="color:#c97b3a">For deeper skin tones,</strong> undertone is the most reliable signal — trust that above depth if uncertain.<br><br>
    This is a guide, not a verdict. Professional in-person analysis with physical drapes is always more precise.
    </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">a note just for you</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card-dark"><div class="msg-quote">"</div>
    <div class="lbl" style="color:#c9a882;margin-bottom:14px">personal message</div>
    <div class="msg-text">{info["personal_msg"]}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-div">your report</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card" style="text-align:center;padding:20px 26px">
    <div class="lbl" style="margin-bottom:10px">download your full color report</div>
    <div class="info-val" style="font-size:12px;margin-bottom:14px">Season, hex codes, makeup shades, clothing guide, personal message — all in one PDF.</div>""", unsafe_allow_html=True)
    pdf_bytes = generate_pdf(pred, depth, info["season"], conf, model_acc, hex_skin, r, g, b, info, secondary)
    st.download_button("⬇  Download Tinta Report (PDF)", data=pdf_bytes,
        file_name=f"tinta_{info['season'].lower().replace(' ','_')}_report.pdf", mime="application/pdf")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Tinta", layout="centered", page_icon="🎨")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,600;1,300;1,600&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: #fdf8f4 !important;
  color: #2e1f14 !important;
  font-family: 'DM Sans', sans-serif !important;
}
.block-container { max-width:700px !important; padding:2.5rem 1.2rem 5rem !important; }

/* ── ALL text visible ── */
p, span, div, label, li, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span { color: #2e1f14 !important; }

/* ── Brand ── */
.brand{font-family:'Cormorant Garamond',serif;font-size:clamp(2.6rem,7vw,4.2rem);font-weight:300;letter-spacing:.18em;color:#2e1f14 !important;text-align:center;line-height:1}
.brand em{font-style:italic;color:#c97b3a !important}
.tagline{text-align:center;font-size:9px;letter-spacing:.32em;text-transform:uppercase;color:#b09070 !important;margin-top:8px;margin-bottom:40px}

/* ── Cards ── */
.card{background:#ffffff;border:1px solid #ede0d4;border-radius:18px;padding:24px 26px;margin-top:14px;box-shadow:0 2px 12px rgba(180,120,60,.07)}
.card-gold{background:linear-gradient(135deg,#fffaf4,#fff8f0);border:1px solid #e8c89a;border-radius:18px;padding:24px 26px;margin-top:14px;box-shadow:0 2px 12px rgba(180,120,60,.08)}
.card-hero{background:linear-gradient(160deg,#fff9f4,#fef3e8);border:1px solid #e8c89a;border-radius:22px;padding:36px 28px;margin-top:14px;text-align:center;box-shadow:0 4px 20px rgba(180,120,60,.10)}
.card-dark{background:#fff5ec;border:1px solid #e8c89a;border-radius:18px;padding:30px 28px;margin-top:14px;position:relative;overflow:hidden}
.card-alert{background:#fffaf6;border:1px solid #ede0d4;border-left:3px solid #c97b3a;border-radius:0 14px 14px 0;padding:18px 22px;margin-top:14px}
.card-secondary{background:#fdf5ee;border:1px dashed #ddc8b0;border-radius:16px;padding:20px 24px;margin-top:10px}

/* ── Typography ── */
.lbl{font-size:8.5px !important;letter-spacing:.26em;text-transform:uppercase;color:#b09070 !important;font-weight:600;margin-bottom:9px}
.season-pill{display:inline-block;border-radius:99px;padding:5px 20px;font-size:9.5px;letter-spacing:.2em;text-transform:uppercase;font-weight:600;margin-bottom:14px}
.season-name{font-family:'Cormorant Garamond',serif;font-size:clamp(2rem,5vw,3rem);font-weight:300;color:#2e1f14 !important;line-height:1.1;margin-bottom:5px}
.season-sub{font-size:12px;color:#b09070 !important;letter-spacing:.08em;margin-bottom:18px}
.season-desc{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:17px;color:#7a5c44 !important;line-height:1.7;max-width:480px;margin:0 auto}
.info-val{font-size:13.5px !important;color:#7a5c44 !important;line-height:1.7}
.style-tip{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:15.5px;color:#c97b3a !important;line-height:1.65}
.metals-name{font-family:'Cormorant Garamond',serif;font-size:26px;font-weight:300;color:#2e1f14 !important;margin-bottom:5px}
.msg-quote{font-family:'Cormorant Garamond',serif;font-size:110px;color:#c97b3a !important;opacity:.12;position:absolute;top:-22px;left:14px;line-height:1}
.msg-text{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:18px;color:#5a3a22 !important;line-height:1.8;position:relative;z-index:1}

/* ── Dividers ── */
.sec-div{display:flex;align-items:center;gap:12px;margin:24px 0 0;color:#c9a882 !important;font-size:8.5px;letter-spacing:.24em;text-transform:uppercase}
.sec-div::before,.sec-div::after{content:'';flex:1;border-top:1px solid #ede0d4}

/* ── Confidence bar ── */
.conf-track{background:#f0e8df;border-radius:99px;height:3px;margin:16px auto 0;max-width:300px;overflow:hidden}

/* ── Swatches ── */
.swatch-row{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px}
.sw{width:62px;height:70px;border-radius:14px;display:flex;align-items:flex-end;justify-content:center;padding-bottom:7px;font-size:7.5px;font-weight:600;letter-spacing:.05em;text-align:center;text-shadow:0 1px 5px rgba(0,0,0,.35);border:1px solid rgba(0,0,0,.08);flex-shrink:0;line-height:1.2;box-shadow:0 2px 8px rgba(0,0,0,.10)}
.sw-avoid{width:62px;height:70px;border-radius:14px;display:flex;align-items:flex-end;justify-content:center;padding-bottom:7px;font-size:7.5px;font-weight:600;letter-spacing:.05em;text-align:center;text-shadow:0 1px 5px rgba(0,0,0,.35);border:1.5px solid rgba(180,60,60,.25);opacity:.65;flex-shrink:0;position:relative;line-height:1.2}

/* ── Chips & Tags ── */
.chip-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}
.chip{display:flex;align-items:center;gap:8px;background:#fdf5ee;border:1px solid #ede0d4;border-radius:10px;padding:7px 13px;font-size:12px;color:#5a3a22 !important;font-weight:400}
.dot{width:13px;height:13px;border-radius:50%;flex-shrink:0;border:1px solid rgba(0,0,0,.12)}
.tag-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}
.tag{display:inline-block;background:#fdf5ee;border:1px solid #ede0d4;border-radius:7px;padding:5px 13px;font-size:11.5px;color:#7a5c44 !important}

/* ── Misc ── */
.skin-ring{display:flex;flex-direction:column;align-items:center;gap:6px;padding-top:14px}
.rgb-small{font-size:8.5px;letter-spacing:.1em;color:#c9a882 !important;font-family:monospace;text-align:center}
.retake-step{display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid #ede0d4}
.retake-num{font-family:'Cormorant Garamond',serif;font-size:22px;color:#c97b3a !important;font-weight:300;line-height:1;min-width:24px}
.retake-txt{font-size:13px;color:#7a5c44 !important;line-height:1.5;padding-top:3px}
.calib-box{background:#fff8f2;border:2px dashed #e0c8b0;border-radius:14px;padding:20px;text-align:center;margin-top:12px}

/* ══ STREAMLIT WIDGET OVERRIDES — force light theme on all controls ══ */

/* Radio buttons */
[data-testid="stRadio"] label, [data-testid="stRadio"] span,
.stRadio label, .stRadio span { color: #2e1f14 !important; font-size:13px !important; }
[data-testid="stRadio"] > div { gap: 10px !important; }

/* All generic text */
[data-testid="stText"], [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * { color: #2e1f14 !important; }

/* File uploader */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #7a5c44 !important; font-size:12px !important; }
[data-testid="stFileUploader"] { background: #fff8f2 !important; border-radius: 12px !important; }

/* Section headers / captions */
[data-testid="stCaptionContainer"] p { color: #b09070 !important; }

/* Buttons */
.stButton > button { background:#fff8f2 !important; border:1px solid #e0c8b0 !important; color:#5a3a22 !important; border-radius:12px !important; font-size:13px !important; }
.stButton > button:hover { background:#fdeedd !important; border-color:#c97b3a !important; }

/* Download button */
[data-testid="stDownloadButton"] button { background:#fff0e4 !important; border:1px solid #e0c8b0 !important; color:#c97b3a !important; border-radius:12px !important; font-size:12px !important; letter-spacing:.1em !important; padding:10px 24px !important; width:100%; margin-top:8px; }
[data-testid="stDownloadButton"] button:hover { background:#fdeedd !important; border-color:#c97b3a !important; }

/* Spinner */
.stSpinner > div { border-top-color:#c97b3a !important; }

/* Expander */
[data-testid="stExpander"] { border-color:#ede0d4 !important; background:#ffffff !important; }
[data-testid="stExpander"] summary span { color:#2e1f14 !important; }

/* Selectbox / dropdown */
[data-testid="stSelectbox"] label, [data-testid="stSelectbox"] span { color:#2e1f14 !important; }

/* Image captions */
[data-testid="stImage"] p, .stImage p { color:#b09070 !important; font-size:11px !important; }

/* Input widgets */
input, textarea, select { background:#ffffff !important; color:#2e1f14 !important; border-color:#ede0d4 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#fdf0e6 !important; }
section[data-testid="stSidebar"] * { color:#2e1f14 !important; }

/* Color picker label */
[data-testid="stColorPicker"] label { color:#2e1f14 !important; }

/* Column headers in tables */
th { color:#2e1f14 !important; background:#fdf5ee !important; }
td { color:#5a3a22 !important; }
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="brand">Ti<em>nta</em></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">AI · Personal Color Analysis</div>', unsafe_allow_html=True)
st.markdown("""<div class="card" style="text-align:center">
<p style="font-size:13px;color:#c9a882;line-height:1.75;margin:0">
Upload a <strong style="color:#b09070">clear, front-facing photo</strong> in natural daylight.<br>
No filters, no heavy makeup. White balance correction is applied automatically.
</p></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODE SELECTOR (UPDATED WITH NEW MODES)
# ═══════════════════════════════════════════════════════════════════
mode = st.radio("Mode:", [
    "📸 Single photo",
    "🎞️ Multi-photo (more accurate)",
    "🎨 Manual colour picker",
    "📋 Undertone quiz",
    "🔬 Compare two photos",
    "📷 Calibrate camera"
], horizontal=True)

# ═══════════════════════════════════════════════════════════════════
# SINGLE PHOTO MODE
# ═══════════════════════════════════════════════════════════════════
if mode == "📸 Single photo":
    opt = st.radio("Input:", ["📁  Upload","📷  Camera"], horizontal=True, key="s1")
    image = None
    if opt=="📁  Upload":
        f = st.file_uploader("Upload face photo", type=["jpg","png","jpeg"], key="s1f")
        if f: image = Image.open(f)
    else:
        cam = st.camera_input("Face the camera in natural light", key="s1c")
        if cam: image = Image.open(cam)

    if image:
        with st.spinner("Analysing…"):
            res, status, qs = analyse(image)
        render_quality_banner(qs)
        if status=="no_face":   st.error("No face detected. Use a clear front-facing photo in good lighting.")
        elif status=="no_skin": st.error("Couldn't extract skin pixels. Try better lighting or move closer.")
        elif res["conf"]<60:    render_retake()
        else:                   render_full_result(res, image)

# ═══════════════════════════════════════════════════════════════════
# MULTI-PHOTO MODE
# ═══════════════════════════════════════════════════════════════════
elif mode == "🎞️ Multi-photo (more accurate)":
    st.markdown("""<div class="card" style="padding:16px 20px">
    <div class="lbl">why multi-photo is more accurate</div>
    <div class="info-val" style="font-size:12px;margin-top:6px">
    Upload 2–3 photos of yourself in similar lighting. The app averages the skin readings across all photos
    before classifying — eliminating unlucky shadows, slight exposure differences, or camera processing quirks
    that can affect a single shot.
    </div></div>""", unsafe_allow_html=True)

    files = st.file_uploader("Upload 2–3 face photos", type=["jpg","png","jpeg"],
                              accept_multiple_files=True, key="multi")
    if files:
        if len(files) < 2:
            st.warning("Upload at least 2 photos for averaging to make a difference.")
        elif len(files) > 3:
            st.warning("Maximum 3 photos — using the first 3.")
            files = files[:3]

        if len(files) >= 2:
            images = [Image.open(f) for f in files]
            cols = st.columns(len(images))
            for i, (col, img) in enumerate(zip(cols, images)):
                with col: st.image(img, use_container_width=True, caption=f"Photo {i+1}")

            with st.spinner(f"Averaging across {len(images)} photos…"):
                res, status, qs = analyse_multi(images)

            render_quality_banner(qs)
            if status in ("no_face","no_skin"):
                st.error("Couldn't detect a face in one or more photos. Check your uploads.")
            elif res["conf"]<60:
                render_retake()
            else:
                render_full_result(res, images[0])

# ═══════════════════════════════════════════════════════════════════
# NEW FEATURE: MANUAL COLOUR PICKER MODE
# ═══════════════════════════════════════════════════════════════════
elif mode == "🎨 Manual colour picker":
    st.markdown("""<div class="card" style="margin-bottom: 20px">
    <div class="lbl">🎨 Manual Colour Selection</div>
    <div class="info-val" style="font-size: 13px">
    No good photo? No problem. Select your skin, eye, and hair colours below.
    </div>
    </div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        skin_colour = st.color_picker("Skin tone", "#D4A574", key="skin_picker")
        st.markdown(f'<div style="width:100%;height:40px;background:{skin_colour};border-radius:8px;margin-top:5px"></div>', unsafe_allow_html=True)
    with col2:
        eye_colour = st.color_picker("Eye colour", "#8B5E3C", key="eye_picker")
        st.markdown(f'<div style="width:100%;height:40px;background:{eye_colour};border-radius:8px;margin-top:5px"></div>', unsafe_allow_html=True)
    with col3:
        hair_colour = st.color_picker("Hair colour", "#5C3A1E", key="hair_picker")
        st.markdown(f'<div style="width:100%;height:40px;background:{hair_colour};border-radius:8px;margin-top:5px"></div>', unsafe_allow_html=True)
    
    if st.button("Analyze my colours", use_container_width=True):
        with st.spinner("Analysing your colour profile..."):
            res = manual_colour_picker_analysis(skin_colour, eye_colour, hair_colour)
            if res:
                placeholder_img = Image.new('RGB', (300, 300), color=skin_colour)
                draw = ImageDraw.Draw(placeholder_img)
                draw.rectangle([0, 250, 100, 300], fill=eye_colour)
                draw.rectangle([100, 250, 200, 300], fill=hair_colour)
                draw.rectangle([200, 250, 300, 300], fill=skin_colour)
                render_full_result(res, placeholder_img)

# ═══════════════════════════════════════════════════════════════════
# NEW FEATURE: UNDERTONE QUIZ MODE
# ═══════════════════════════════════════════════════════════════════
elif mode == "📋 Undertone quiz":
    st.markdown("""<div class="card" style="margin-bottom: 20px">
    <div class="lbl">📋 Professional Undertone Self-Test</div>
    <div class="info-val" style="font-size: 13px">
    Based on the same questions professional colour consultants ask during in-person draping sessions.
    No camera needed — just answer honestly.
    </div>
    </div>""", unsafe_allow_html=True)
    
    with st.form("quiz_form"):
        pred, conf = undertone_quiz()
        
        depth_q = st.radio("How would you describe your natural skin depth?",
                           ["Fair / Light (burns easily in sun)",
                            "Medium / Olive (sometimes burns, sometimes tans)",
                            "Deep / Dark (rarely burns, tans quickly or always dark)"],
                           key="depth_q", horizontal=False)

        submitted = st.form_submit_button("Get My Undertone", use_container_width=True)

        if submitted:
            if "Fair" in depth_q:
                depth = "Light"
            elif "Medium" in depth_q:
                depth = "Medium"
            else:
                depth = "Deep"
            
            info = SD.get((pred, depth), SD[("Neutral", "Medium")])
            
            st.markdown(f"""
            <div class="card-hero" style="margin-top: 20px">
            <div class="season-name">Your undertone: <span style="color:#c97b3a">{pred}</span></div>
            <div class="season-sub">Confidence: {conf:.0f}%</div>
            <div class="season-desc" style="margin-top: 16px">
            Based on your answers, you have a <strong>{pred.lower()}</strong> undertone with <strong>{depth.lower()}</strong> depth.
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            res = {"r": 180, "g": 140, "b": 100, "pred": pred, "conf": conf, 
                   "depth": depth, "info": info}
            
            placeholder_img = Image.new('RGB', (300, 300), color='#2e1f14')
            render_full_result(res, placeholder_img)

# ═══════════════════════════════════════════════════════════════════
# COMPARE MODE
# ═══════════════════════════════════════════════════════════════════
elif mode == "🔬 Compare two photos":
    st.markdown("""<div class="card" style="text-align:center;padding:16px 20px">
    <div class="info-val" style="font-size:12px">Upload two photos — different lighting, makeup, or days — and compare what the model reads for each.</div></div>""", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        fa = st.file_uploader("Photo A", type=["jpg","png","jpeg"], key="cmp_a")
        img_a = Image.open(fa) if fa else None
    with cb:
        fb = st.file_uploader("Photo B", type=["jpg","png","jpeg"], key="cmp_b")
        img_b = Image.open(fb) if fb else None

    if img_a and img_b:
        with st.spinner("Analysing both photos…"):
            res_a, st_a, qs_a = analyse(img_a)
            res_b, st_b, qs_b = analyse(img_b)

        st.markdown('<div class="sec-div">side by side</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        def mini(col, res, status, img, label):
            with col:
                st.image(img, use_container_width=True, caption=label)
                if status!="ok" or res is None:
                    st.markdown('<div class="card"><div class="info-val" style="font-size:12px;color:#a05040">Could not analyse.</div></div>', unsafe_allow_html=True)
                    return
                info = res["info"]
                st.markdown(f"""<div class="card" style="text-align:center;padding:18px 16px">
                <span class="season-pill" style="background:{info['badge_bg']};color:{info['badge_fg']};font-size:8px">{info['season']}</span>
                <div style="font-family:'Cormorant Garamond',serif;font-size:20px;color:#2e1f14;margin:6px 0 2px">{res['pred']} · {res['depth']}</div>
                <div style="font-size:11px;color:#b09070">Confidence {res['conf']:.0f}%</div>
                <div style="margin-top:12px">{swatches(info['palette'])}</div></div>""", unsafe_allow_html=True)
        mini(col1, res_a, st_a, img_a, "Photo A")
        mini(col2, res_b, st_b, img_b, "Photo B")
        if res_a and res_b:
            same = res_a["pred"]==res_b["pred"] and res_a["depth"]==res_b["depth"]
            note = "Both photos give the same season — consistent result." if same else \
                   (f"Photo A → {res_a['info']['season']}, Photo B → {res_b['info']['season']}. "
                    "This usually means the lighting conditions differ significantly. "
                    "The Multi-photo mode will average them for a single stable answer.")
            nc = "#5a8a60" if same else "#c97b3a"
            st.markdown(f"""<div class="card-alert" style="border-left-color:{nc};margin-top:14px">
            <div class="lbl">comparison note</div>
            <div class="info-val" style="margin-top:6px">{note}</div></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CALIBRATE CAMERA MODE
# ═══════════════════════════════════════════════════════════════════
elif mode == "📷 Calibrate camera":
    st.markdown("""<div class="card">
    <div class="lbl">camera calibration</div>
    <div class="info-val" style="margin-top:8px">
    Different phones process colour differently — iPhones oversaturate reds, some Androids shift cool.
    Calibrating with a white reference corrects your specific camera's bias.<br><br>
    <strong style="color:#5a3a22">How to do it:</strong><br>
    1. Find a plain white piece of paper or a white wall<br>
    2. Photograph it in the same light you'll use for your face photo<br>
    3. Upload it below — the app measures how far your camera drifts from pure white<br>
    4. This correction factor is saved and applied to your colour analysis automatically
    </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="calib-box"><div class="lbl">upload your white reference photo</div></div>', unsafe_allow_html=True)
    calib_f = st.file_uploader("White reference (plain white paper / wall)", type=["jpg","png","jpeg"], key="calib")

    if calib_f:
        calib_img = np.array(Image.open(calib_f).convert("RGB"))
        h,w = calib_img.shape[:2]
        cx,cy = w//2, h//2
        margin_x, margin_y = w//5, h//5
        patch = calib_img[cy-margin_y:cy+margin_y, cx-margin_x:cx+margin_x]
        mean_r = float(np.mean(patch[:,:,0]))
        mean_g = float(np.mean(patch[:,:,1]))
        mean_b = float(np.mean(patch[:,:,2]))

        off_r = 255 - mean_r
        off_g = 255 - mean_g
        off_b = 255 - mean_b

        st.session_state["calib_offset"] = (off_r, off_g, off_b)

        bias_r = mean_r - 128
        bias_g = mean_g - 128
        bias_b = mean_b - 128
        dominant = "Warm (yellow/orange)" if bias_r > 15 else \
                   "Cool (blue)" if bias_b > bias_r + 10 else "Neutral (minimal bias)"

        st.markdown(f"""<div class="card-alert" style="border-left-color:#5a8a60;margin-top:14px">
        <div class="lbl">calibration complete</div>
        <div class="info-val" style="margin-top:8px">
        Detected white: rgb({mean_r:.0f}, {mean_g:.0f}, {mean_b:.0f})<br>
        Camera bias: <strong style="color:#c97b3a">{dominant}</strong><br>
        Correction offset: R +{off_r:.1f} · G +{off_g:.1f} · B +{off_b:.1f}<br><br>
        This correction will be applied automatically when you switch to Single or Multi-photo mode.
        </div></div>""", unsafe_allow_html=True)

        st.image(Image.open(calib_f), caption="Your white reference", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""<div style="text-align:center;margin-top:44px;color:#e8d8c8;font-size:10px;letter-spacing:.18em">
TINTA  ·  AI PERSONAL COLOR ANALYSIS<br>
<span style="color:#e8d8c8;font-size:9px;letter-spacing:.1em">Results are a guide. Professional draping gives maximum precision.</span>
</div>""", unsafe_allow_html=True)
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
import plotly.graph_objects as go
import cv2
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from model import FusionModel

# --------------------------------
# 1. Configuration & Constants
# --------------------------------
st.set_page_config(
    page_title="Skin Disease Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis-like Lesions',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}

HIGH_RISK_CLASSES = {'mel', 'bcc', 'akiec'}

CLASS_INFO = {
    'akiec': "A rough, scaly patch on the skin caused by years of sun exposure. It can sometimes turn into skin cancer if untreated.",
    'bcc':   "The most common type of skin cancer. It grows slowly and rarely spreads, but needs treatment.",
    'bkl':   "A non-cancerous skin growth that looks like a wart or mole. It is completely harmless.",
    'df':    "A small, harmless bump under the skin, usually on the legs. It is benign (not cancer).",
    'mel':   "The most dangerous form of skin cancer. Early detection is critical for survival.",
    'nv':    "A common mole. Usually harmless, but should be monitored for changes over time.",
    'vasc':  "Lesions related to blood vessels in the skin. Usually benign but should be checked by a doctor."
}

AGE_MEAN = 51.86
AGE_STD  = 16.96

# --------------------------------
# 1b. Session State — History
# --------------------------------
# st.session_state is like a memory box that survives page reruns.
# We store a list of dicts, one per prediction made this session.
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --------------------------------
# 2. Custom CSS Styling
# --------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
        color: #e8eaf0;
    }

    .header-banner {
        background: linear-gradient(90deg, #1e3a5f 0%, #0d2137 100%);
        border-left: 5px solid #4fc3f7;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 28px;
    }
    .header-banner h1 { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin: 0 0 6px 0; }
    .header-banner p  { font-size: 1rem; color: #90caf9; margin: 0; }

    .disclaimer {
        background: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 8px; padding: 10px 16px;
        font-size: 0.82rem; color: #ffe082; margin-bottom: 24px;
    }

    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #90caf9;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 16px;
    }

    .risk-high {
        display: inline-block;
        background: linear-gradient(90deg, #b71c1c, #e53935);
        color: white; font-weight: 700; font-size: 1rem;
        padding: 10px 22px; border-radius: 50px;
        letter-spacing: 0.05em; margin-bottom: 12px;
    }
    .risk-low {
        display: inline-block;
        background: linear-gradient(90deg, #1b5e20, #43a047);
        color: white; font-weight: 700; font-size: 1rem;
        padding: 10px 22px; border-radius: 50px;
        letter-spacing: 0.05em; margin-bottom: 12px;
    }

    .prediction-box {
        background: rgba(79, 195, 247, 0.07);
        border: 1px solid rgba(79, 195, 247, 0.25);
        border-radius: 14px; padding: 20px 24px; margin-bottom: 16px;
    }
    .prediction-label    { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .prediction-code     { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #4fc3f7; }
    .prediction-confidence { font-size: 1.1rem; font-weight: 600; color: #a5d6a7; margin-top: 8px; }

    .info-card {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #4fc3f7; border-radius: 8px;
        padding: 14px 18px; font-size: 0.9rem; color: #cfd8dc;
        margin-bottom: 16px; line-height: 1.6;
    }

    .top3-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .top3-rank { font-size: 1.3rem; font-weight: 700; color: #4fc3f7; margin-right: 14px; min-width: 28px; }
    .top3-name { font-size: 0.95rem; font-weight: 500; color: #e0e0e0; flex: 1; }
    .top3-pct  { font-size: 0.95rem; font-weight: 600; color: #80cbc4; font-family: 'DM Mono', monospace; }

    .warning-box {
        background: rgba(255, 152, 0, 0.08);
        border: 1px solid rgba(255, 152, 0, 0.35);
        border-radius: 10px; padding: 14px 18px;
        color: #ffcc80; font-size: 0.9rem; margin-top: 12px;
    }
    .doctor-box {
        background: rgba(229, 57, 53, 0.08);
        border: 1px solid rgba(229, 57, 53, 0.35);
        border-radius: 10px; padding: 14px 18px;
        color: #ef9a9a; font-size: 0.9rem;
        margin-top: 12px; line-height: 1.6;
    }
    .gradcam-legend {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 14px 18px;
        font-size: 0.85rem; color: #90a4ae;
        margin-top: 10px; line-height: 1.8;
    }
    .pdf-box {
        background: rgba(129, 199, 132, 0.07);
        border: 1px solid rgba(129, 199, 132, 0.25);
        border-radius: 12px; padding: 20px 24px; margin-top: 8px;
    }

    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: #cfd8dc !important; }

    .stButton > button {
        background: linear-gradient(90deg, #1565c0, #0288d1);
        color: white !important; border: none;
        border-radius: 10px; font-weight: 600;
        font-size: 1rem; padding: 12px 0; width: 100%;
        transition: all 0.2s ease; letter-spacing: 0.03em;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1976d2, #039be5);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(2, 136, 209, 0.4);
    }

    hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# 3. Helper Functions
# --------------------------------

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(num_meta_features=19, num_classes=7)
    try:
        state_dict = torch.load("best_fusion_model_finetuned.pth", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error("❌ Model file 'best_fusion_model_finetuned.pth' not found.")
        return None, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device
    model.to(device)
    model.eval()
    return model, device


def preprocess_metadata(age, sex, localization):
    norm_age = (age - AGE_MEAN) / AGE_STD
    features = {
        'age': norm_age,
        'sex_female': 0, 'sex_male': 0, 'sex_unknown': 0,
        'localization_abdomen': 0, 'localization_acral': 0, 'localization_back': 0,
        'localization_chest': 0, 'localization_ear': 0, 'localization_face': 0,
        'localization_foot': 0, 'localization_genital': 0, 'localization_hand': 0,
        'localization_lower extremity': 0, 'localization_neck': 0, 'localization_scalp': 0,
        'localization_trunk': 0, 'localization_unknown': 0, 'localization_upper extremity': 0
    }
    if f'sex_{sex}' in features:
        features[f'sex_{sex}'] = 1
    else:
        features['sex_unknown'] = 1
    if f'localization_{localization}' in features:
        features[f'localization_{localization}'] = 1
    else:
        features['localization_unknown'] = 1

    final_vector = [features['age']]
    sex_cols = sorted([k for k in features if k.startswith('sex_')])
    final_vector.extend([features[k] for k in sex_cols])
    loc_cols = sorted([k for k in features if k.startswith('localization_')])
    final_vector.extend([features[k] for k in loc_cols])
    return torch.tensor(final_vector, dtype=torch.float32)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def get_risk_level(class_code):
    return class_code in HIGH_RISK_CLASSES


def make_plotly_chart(probabilities):
    labels = [CLASS_NAMES[c] for c in CLASSES]
    values = [round(float(p) * 100, 2) for p in probabilities]
    colors_list = ['#ef5350' if get_risk_level(CLASSES[i]) else '#26c6da' for i in range(len(CLASSES))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors_list, line=dict(color='rgba(255,255,255,0.05)', width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#e0e0e0', size=12)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#e0e0e0'),
        xaxis=dict(title='Probability (%)', range=[0, max(values) * 1.25],
                   gridcolor='rgba(255,255,255,0.06)', color='#90a4ae'),
        yaxis=dict(autorange='reversed', gridcolor='rgba(255,255,255,0.06)', color='#90a4ae'),
        margin=dict(l=10, r=60, t=20, b=40), height=320, showlegend=False
    )
    return fig


def generate_gradcam(model, img_tensor, meta_tensor, target_class_idx, device):
    activations = {}
    gradients   = {}

    def save_activation(module, input, output):
        activations['layer4'] = output.detach()

    def save_gradient(module, input, output):
        output.register_hook(lambda grad: gradients.update({'layer4': grad.detach()}))

    target_layer = model.cnn.layer4
    fwd_hook     = target_layer.register_forward_hook(save_activation)
    bwd_hook     = target_layer.register_forward_hook(save_gradient)

    model.zero_grad()
    output = model(img_tensor.to(device), meta_tensor.to(device))

    one_hot = torch.zeros_like(output)
    one_hot[0][target_class_idx] = 1
    output.backward(gradient=one_hot)

    act  = activations['layer4']
    grad = gradients.get('layer4', activations['layer4'])

    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam     = (weights * act).sum(dim=1, keepdim=True)
    cam     = F.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = (cam * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original_np = np.array(image.resize((224, 224))).astype(np.float32)
    overlay     = (0.6 * original_np + 0.4 * heatmap.astype(np.float32)).clip(0, 255).astype(np.uint8)

    fwd_hook.remove()
    bwd_hook.remove()

    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ============================================================
# PDF REPORT GENERATOR
# ============================================================
def pil_image_to_reportlab(pil_img, width_cm, height_cm):
    """
    Converts a PIL image into a ReportLab Image object.
    We save it to a bytes buffer in memory (no file on disk needed).
    """
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def generate_pdf_report(
    patient_age, patient_sex, patient_localization,
    top1_class, top1_conf, is_high_risk,
    sorted_indices, probabilities,
    original_image, gradcam_overlay
):
    """
    Builds a professional PDF report and returns it as bytes.
    The bytes can then be downloaded directly from Streamlit.
    """

    # --- Buffer: we build the PDF in memory, not on disk ---
    buffer = io.BytesIO()

    # --- Page setup ---
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    # --- Define custom styles ---
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    style_subtitle = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#455a64'),
        spaceAfter=4,
        alignment=TA_CENTER
    )
    style_section = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#0d47a1'),
        spaceBefore=16,
        spaceAfter=8,
        fontName='Helvetica-Bold',
        borderPad=4
    )
    style_body = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#263238'),
        spaceAfter=4,
        leading=16
    )
    style_disclaimer = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#b71c1c'),
        spaceAfter=4,
        leading=13,
        alignment=TA_CENTER
    )

    # ---- Start building the story (list of elements) ----
    story = []

    # ======================
    # HEADER
    # ======================
    story.append(Paragraph("AI Skin Lesion Analysis Report", style_title))
    story.append(Paragraph("Powered by ResNet50 + Metadata Fusion Deep Learning Model", style_subtitle))
    story.append(Paragraph(
        f"Report Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        style_subtitle
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1565c0'), spaceAfter=12))

    # ======================
    # PATIENT INFORMATION
    # ======================
    story.append(Paragraph("Patient Information", style_section))

    patient_data = [
        ['Field', 'Value'],
        ['Age', str(patient_age) + ' years'],
        ['Sex', patient_sex.capitalize()],
        ['Lesion Location', patient_localization.capitalize()],
    ]

    patient_table = Table(patient_data, colWidths=[5 * cm, 10 * cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0), 11),
        ('BACKGROUND',  (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('FONTNAME',    (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 1), (-1, -1), 10),
        ('TEXTCOLOR',   (0, 1), (-1, -1), colors.HexColor('#263238')),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',   (0, 0), (-1, -1), 22),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(patient_table)

    # ======================
    # DIAGNOSIS RESULT
    # ======================
    story.append(Paragraph("Diagnosis Result", style_section))

    risk_label = "HIGH RISK - Malignant" if is_high_risk else "LOW RISK - Benign"
    risk_color = colors.HexColor('#b71c1c') if is_high_risk else colors.HexColor('#1b5e20')
    risk_bg    = colors.HexColor('#ffebee') if is_high_risk else colors.HexColor('#e8f5e9')

    diagnosis_data = [
        ['Field',             'Value'],
        ['Predicted Condition', CLASS_NAMES[top1_class]],
        ['Condition Code',    top1_class.upper()],
        ['Confidence Score',  f"{top1_conf:.2f}%"],
        ['Risk Level',        risk_label],
    ]

    diagnosis_table = Table(diagnosis_data, colWidths=[5 * cm, 10 * cm])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0), 11),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('FONTNAME',    (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 1), (-1, -1), 10),
        ('TEXTCOLOR',   (0, 1), (-1, -1), colors.HexColor('#263238')),
        # Highlight risk level row (last row) specially
        ('BACKGROUND',  (0, 4), (-1, 4), risk_bg),
        ('TEXTCOLOR',   (1, 4), (1, 4), risk_color),
        ('FONTNAME',    (1, 4), (1, 4), 'Helvetica-Bold'),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',   (0, 0), (-1, -1), 22),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(diagnosis_table)

    # Description of condition
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"<b>About this condition:</b> {CLASS_INFO[top1_class]}",
        style_body
    ))

    # ======================
    # TOP 3 PREDICTIONS
    # ======================
    story.append(Paragraph("Top 3 Predictions", style_section))

    top3_data = [['Rank', 'Condition', 'Code', 'Confidence', 'Risk Level']]
    rank_labels = ['1st', '2nd', '3rd']

    for i, idx in enumerate(sorted_indices[:3]):
        cls        = CLASSES[idx]
        conf       = probabilities[idx] * 100
        risk_text  = "High Risk" if get_risk_level(cls) else "Low Risk"
        top3_data.append([
            rank_labels[i],
            CLASS_NAMES[cls],
            cls.upper(),
            f"{conf:.2f}%",
            risk_text
        ])

    top3_table = Table(top3_data, colWidths=[1.5*cm, 6*cm, 1.8*cm, 2.5*cm, 2.5*cm])
    top3_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e3f2fd'), colors.white, colors.HexColor('#f5f5f5')]),
        ('FONTSIZE',    (0, 1), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 1), (-1, -1), colors.HexColor('#263238')),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',   (0, 0), (-1, -1), 20),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',       (3, 0), (3, -1), 'CENTER'),
    ]))
    story.append(top3_table)

    # ======================
    # IMAGES SIDE BY SIDE
    # ======================
    story.append(Paragraph("Lesion Image & Grad-CAM Heatmap", style_section))
    story.append(Paragraph(
        "The left image is the original uploaded lesion. "
        "The right image is the Grad-CAM overlay showing where the AI focused "
        "during its analysis (red/yellow = high attention, blue = low attention).",
        style_body
    ))
    story.append(Spacer(1, 8))

    # Convert both PIL images to ReportLab images
    original_resized  = original_image.resize((224, 224))
    rl_original       = pil_image_to_reportlab(original_resized, 7, 7)
    rl_gradcam        = pil_image_to_reportlab(gradcam_overlay,  7, 7)

    # Put both images side by side in a table
    img_table = Table(
        [['Original Image', 'Grad-CAM Overlay'],
         [rl_original,       rl_gradcam]],
        colWidths=[8 * cm, 8 * cm]
    )
    img_table.setStyle(TableStyle([
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, 0), 10),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 1), (-1, 1), 4),
    ]))
    story.append(img_table)

    # ======================
    # ALL CLASS PROBABILITIES
    # ======================
    story.append(Paragraph("All Class Probabilities", style_section))

    prob_data = [['Condition', 'Code', 'Probability', 'Risk Level']]
    # Sort by probability descending for the table
    sorted_all = np.argsort(probabilities)[::-1]
    for idx in sorted_all:
        cls       = CLASSES[idx]
        conf      = probabilities[idx] * 100
        risk_text = "High Risk" if get_risk_level(cls) else "Low Risk"
        prob_data.append([CLASS_NAMES[cls], cls.upper(), f"{conf:.2f}%", risk_text])

    prob_table = Table(prob_data, colWidths=[7*cm, 2*cm, 3*cm, 2.5*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#1565c0')),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
        ('FONTNAME',    (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, 0), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f5f5f5'), colors.white] * 4),
        ('FONTSIZE',    (0, 1), (-1, -1), 9),
        ('TEXTCOLOR',   (0, 1), (-1, -1), colors.HexColor('#263238')),
        ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
        ('ROWHEIGHT',   (0, 0), (-1, -1), 20),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN',       (2, 0), (2, -1), 'CENTER'),
    ]))
    story.append(prob_table)

    # ======================
    # DISCLAIMER
    # ======================
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#b71c1c'), spaceAfter=8))
    story.append(Paragraph(
        "MEDICAL DISCLAIMER: This report is generated by an AI system for educational "
        "and research purposes only. It is NOT a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified dermatologist or healthcare "
        "provider for any medical concerns. The developers of this system assume no liability "
        "for clinical decisions made based on this output.",
        style_disclaimer
    ))
    story.append(Paragraph(
        "Model: ResNet50 + Metadata Fusion  |  Dataset: HAM10000  |  Overall Accuracy: 89%",
        style_disclaimer
    ))

    # --- Build the PDF ---
    doc.build(story)

    # Return the PDF as bytes
    buffer.seek(0)
    return buffer.read()


# --------------------------------
# 4. Sidebar
# --------------------------------
with st.sidebar:
    st.markdown("## 🩺 About This Tool")
    st.markdown("""
    This system uses a **Fusion Deep Learning Model** that combines:
    - 🖼️ **ResNet50** for image analysis
    - 📋 **Patient metadata** (age, sex, location)

    ---
    **7 Conditions Detected:**
    """)
    for code in CLASSES:
        risk = "🔴" if get_risk_level(code) else "🟢"
        st.markdown(f"{risk} **{code.upper()}** — {CLASS_NAMES[code]}")

    st.markdown("---")
    st.markdown("🔴 = High Risk &nbsp;&nbsp; 🟢 = Low Risk", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Model:** ResNet50 + Metadata Fusion")
    st.markdown("**Dataset:** HAM10000")
    st.markdown("**Accuracy:** 89%")
    st.markdown("---")
    st.markdown("**🔥 Grad-CAM** highlights the region the AI focused on.")
    st.markdown("**📄 PDF Report** available after each analysis.")

    # ---- SESSION HISTORY ----
    st.markdown("---")
    st.markdown("## 📜 Session History")

    if len(st.session_state['history']) == 0:
        # No predictions made yet
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px; padding:12px; font-size:0.82rem; color:#546e7a;
                    text-align:center;">
            No analyses yet.<br>Upload an image and click Analyze to get started.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show each history entry — most recent first
        for i, entry in enumerate(reversed(st.session_state['history'])):
            risk_icon  = "🔴" if entry['is_high_risk'] else "🟢"
            risk_label = "High Risk" if entry['is_high_risk'] else "Low Risk"
            # Alternate slightly different background for readability
            bg_color = "rgba(79,195,247,0.05)" if i % 2 == 0 else "rgba(255,255,255,0.02)"
            st.markdown(f"""
            <div style="background:{bg_color};
                        border:1px solid rgba(255,255,255,0.07);
                        border-radius:10px; padding:12px 14px; margin-bottom:8px;">
                <div style="font-size:0.75rem; color:#546e7a; margin-bottom:4px;">
                    #{len(st.session_state['history']) - i} &nbsp;·&nbsp; {entry['time']}
                </div>
                <div style="font-size:0.92rem; font-weight:600; color:#e0e0e0; margin-bottom:2px;">
                    {risk_icon} {entry['condition']}
                </div>
                <div style="font-size:0.8rem; color:#80cbc4;">
                    Confidence: {entry['confidence']:.1f}%
                </div>
                <div style="font-size:0.78rem; color:#90a4ae; margin-top:2px;">
                    📍 {entry['localization'].capitalize()} &nbsp;·&nbsp;
                    👤 Age {entry['age']} &nbsp;·&nbsp;
                    {entry['sex'].capitalize()}
                </div>
                <div style="font-size:0.75rem; margin-top:4px; color:{'#ef9a9a' if entry['is_high_risk'] else '#a5d6a7'};">
                    {risk_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Clear history button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state['history'] = []
            st.rerun()

# --------------------------------
# 5. Main Header
# --------------------------------
st.markdown("""
<div class="header-banner">
    <h1>🩺 AI Skin Lesion Classifier</h1>
    <p>Deep learning-powered detection across 7 dermatological conditions using image + patient data fusion</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
    It is <strong>not</strong> a substitute for professional medical advice. Always consult a qualified dermatologist.
</div>
""", unsafe_allow_html=True)

# --------------------------------
# 6. Load Model
# --------------------------------
model, device = load_model()

# --------------------------------
# 7. Input Layout
# --------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">📷 Step 1 — Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a dermoscopic skin lesion image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.markdown("""
        <div style="border: 2px dashed rgba(255,255,255,0.12); border-radius:12px;
                    padding: 40px; text-align:center; color:#546e7a; font-size:0.9rem;">
            📂 Drag and drop or click above to upload an image
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">📋 Step 2 — Patient Metadata</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">💡 Providing accurate patient details improves prediction accuracy significantly.</div>', unsafe_allow_html=True)

    age          = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    sex          = st.selectbox("Sex", ["male", "female", "unknown"])
    localization = st.selectbox(
        "Location of Lesion",
        ["back", "lower extremity", "trunk", "upper extremity", "abdomen",
         "face", "chest", "foot", "neck", "scalp", "hand", "ear", "genital", "acral"]
    )
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Lesion", type="primary", use_container_width=True)

# --------------------------------
# 8. Prediction + Grad-CAM + PDF
# --------------------------------
if predict_btn and uploaded_file is None:
    st.warning("⚠️ Please upload an image before clicking Analyze.")

if predict_btn and uploaded_file is not None and model is not None:
    with st.spinner("🔬 Analyzing lesion and generating heatmap... please wait"):

        # --- Normal prediction ---
        img_tensor  = preprocess_image(image).to(device)
        meta_tensor = preprocess_metadata(age, sex, localization).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs       = model(img_tensor, meta_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

        sorted_indices = np.argsort(probabilities)[::-1]
        top1_idx       = sorted_indices[0]
        top1_class     = CLASSES[top1_idx]
        top1_conf      = probabilities[top1_idx] * 100
        is_high_risk   = get_risk_level(top1_class)

        # --- Grad-CAM ---
        img_tensor_gc  = preprocess_image(image).to(device)
        meta_tensor_gc = preprocess_metadata(age, sex, localization).unsqueeze(0).to(device)

        try:
            gradcam_overlay, gradcam_heatmap = generate_gradcam(
                model, img_tensor_gc, meta_tensor_gc, top1_idx, device
            )
            gradcam_success = True
        except Exception as e:
            gradcam_success = False
            gradcam_error   = str(e)
            gradcam_overlay = image.resize((224, 224))  # fallback

        # --- Save to Session History ---
        # We build a small dictionary with all the info about this prediction
        # and add it to the front of our history list.
        # If the list already has 5 entries, we remove the oldest one.
        new_entry = {
            'time':        datetime.now().strftime("%H:%M:%S"),   # e.g. "14:32:01"
            'condition':   CLASS_NAMES[top1_class],               # e.g. "Melanoma"
            'code':        top1_class,                            # e.g. "mel"
            'confidence':  top1_conf,                             # e.g. 87.3
            'is_high_risk': is_high_risk,                         # True / False
            'age':         age,
            'sex':         sex,
            'localization': localization,
        }
        st.session_state['history'].append(new_entry)
        # Keep only the last 5 entries
        if len(st.session_state['history']) > 5:
            st.session_state['history'].pop(0)  # remove the oldest

    # =====================
    # RESULTS SECTION
    # =====================
    st.markdown("---")
    st.markdown('<div class="section-title">🧬 Analysis Results</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        if is_high_risk:
            st.markdown('<span class="risk-high">⚠️ HIGH RISK — Malignant</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="risk-low">✅ LOW RISK — Benign</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">{CLASS_NAMES[top1_class]}</div>
            <div class="prediction-code">Code: {top1_class.upper()}</div>
            <div class="prediction-confidence">Confidence: {top1_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="info-card">📖 {CLASS_INFO[top1_class]}</div>', unsafe_allow_html=True)

        if top1_conf < 60:
            st.markdown("""
            <div class="warning-box">
                🤔 <strong>Low Confidence Warning:</strong> The model is not very certain about
                this prediction. Please consult a dermatologist for a proper diagnosis.
            </div>
            """, unsafe_allow_html=True)

        if is_high_risk:
            st.markdown("""
            <div class="doctor-box">
                🏥 <strong>Please See a Doctor:</strong> This lesion has been flagged as potentially
                malignant. Early detection is critical. Please visit a qualified dermatologist
                as soon as possible.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏆 Top 3 Predictions</div>', unsafe_allow_html=True)

        medals = ["🥇", "🥈", "🥉"]
        for rank, idx in enumerate(sorted_indices[:3]):
            cls      = CLASSES[idx]
            conf     = probabilities[idx] * 100
            risk_dot = "🔴" if get_risk_level(cls) else "🟢"
            st.markdown(f"""
            <div class="top3-card">
                <span class="top3-rank">{medals[rank]}</span>
                <span class="top3-name">{risk_dot} {CLASS_NAMES[cls]}</span>
                <span class="top3-pct">{conf:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown('<div class="section-title">📊 All Class Probabilities</div>', unsafe_allow_html=True)
        fig = make_plotly_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        runner_up_conf = probabilities[sorted_indices[1]] * 100
        with m1:
            st.metric("Top Confidence", f"{top1_conf:.1f}%")
        with m2:
            st.metric("2nd Best", f"{runner_up_conf:.1f}%")
        with m3:
            st.metric("Margin", f"{top1_conf - runner_up_conf:.1f}%")

    # =====================
    # GRAD-CAM SECTION
    # =====================
    st.markdown("---")
    st.markdown('<div class="section-title">🔥 Grad-CAM — AI Attention Heatmap</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        🧠 <strong>What is this?</strong> Grad-CAM shows <em>exactly where</em> the AI was
        looking when it made its prediction. <strong>Red/yellow</strong> = heavily focused area.
        <strong>Blue</strong> = ignored area. This proves the AI is looking at the actual
        lesion and not random background noise.
    </div>
    """, unsafe_allow_html=True)

    if gradcam_success:
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.markdown('<div class="section-title" style="font-size:0.85rem; text-align:center;">📷 Original</div>', unsafe_allow_html=True)
            st.image(image.resize((224, 224)), use_container_width=True)
        with gc2:
            st.markdown('<div class="section-title" style="font-size:0.85rem; text-align:center;">🌡️ Raw Heatmap</div>', unsafe_allow_html=True)
            st.image(gradcam_heatmap, use_container_width=True)
        with gc3:
            st.markdown('<div class="section-title" style="font-size:0.85rem; text-align:center;">🔥 AI Focus Overlay</div>', unsafe_allow_html=True)
            st.image(gradcam_overlay, use_container_width=True)

        st.markdown("""
        <div class="gradcam-legend">
            🎨 <strong>Colour Guide:</strong> &nbsp;
            🔴 <strong>Red / Yellow</strong> = AI focused here the most &nbsp;|&nbsp;
            🟡 <strong>Green</strong> = Moderate attention &nbsp;|&nbsp;
            🔵 <strong>Blue</strong> = Low attention / ignored by AI
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Grad-CAM could not be generated: {gradcam_error}")

    # =====================
    # PDF REPORT SECTION
    # =====================
    st.markdown("---")
    st.markdown('<div class="section-title">📄 Download Report</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        📋 <strong>What's in the report?</strong> A complete professional PDF containing:
        patient information, diagnosis result, risk level, top 3 predictions,
        all class probabilities, the original lesion image, the Grad-CAM heatmap,
        and a medical disclaimer.
    </div>
    """, unsafe_allow_html=True)

    # Generate the PDF bytes
    with st.spinner("📄 Generating PDF report..."):
        pdf_bytes = generate_pdf_report(
            patient_age          = age,
            patient_sex          = sex,
            patient_localization = localization,
            top1_class           = top1_class,
            top1_conf            = top1_conf,
            is_high_risk         = is_high_risk,
            sorted_indices       = sorted_indices,
            probabilities        = probabilities,
            original_image       = image,
            gradcam_overlay      = gradcam_overlay
        )

    # File name includes date and predicted condition
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"skin_analysis_{top1_class}_{timestamp}.pdf"

    st.markdown('<div class="pdf-box">', unsafe_allow_html=True)
    st.download_button(
        label     = "⬇️ Download Full PDF Report",
        data      = pdf_bytes,
        file_name = filename,
        mime      = "application/pdf",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#546e7a; margin-top:8px; text-align:center;">
        📁 File will be saved as: <code>{filename}</code>
    </div>
    """, unsafe_allow_html=True)
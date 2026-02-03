import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
from model import FusionModel

# --------------------------------
# 1. Configuration & Constants
# --------------------------------
st.set_page_config(
    page_title="Skin Disease Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LABELS: Ensure this matches the 'label_map' order from your training notebook exactly.
# Based on standard HAM10000 alphabetical sorting, it is usually:
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# STATISTICS: (These should match your training set statistics for normalization)
# Using approximate HAM10000 stats; update if your notebook printed specific ones.
AGE_MEAN = 51.86
AGE_STD = 16.96

# --------------------------------
# 2. Helper Functions
# --------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model structure
    model = FusionModel(num_meta_features=19, num_classes=7)
    
    # Load weights
    try:
        state_dict = torch.load("best_fusion_model_finetuned.pth", map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error("Model file 'best_fusion_model_finetuned.pth' not found. Please upload it.")
        return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

    model.to(device)
    model.eval()
    return model, device

def preprocess_metadata(age, sex, localization):
    """
    Converts user input into the 19-dim tensor expected by the model.
    Order MUST match pd.get_dummies(columns=['sex', 'localization']) alphabetically.
    
    Expected Feature Order (19 total):
    0: age (normalized)
    1-3: sex_female, sex_male, sex_unknown
    4-18: localization_abdomen, localization_acral, localization_back, ..., localization_upper extremity
    """
    
    # 1. Normalize Age
    norm_age = (age - AGE_MEAN) / AGE_STD
    
    # 2. Create feature dictionary initialized to 0
    # Note: These keys must match the exact sorting used by pandas get_dummies
    features = {
        'age': norm_age,
        'sex_female': 0, 'sex_male': 0, 'sex_unknown': 0,
        'localization_abdomen': 0, 'localization_acral': 0, 'localization_back': 0,
        'localization_chest': 0, 'localization_ear': 0, 'localization_face': 0,
        'localization_foot': 0, 'localization_genital': 0, 'localization_hand': 0,
        'localization_lower extremity': 0, 'localization_neck': 0, 'localization_scalp': 0,
        'localization_trunk': 0, 'localization_unknown': 0, 'localization_upper extremity': 0
    }
    
    # 3. Set One-Hot Bits
    # Handle Sex
    if f'sex_{sex}' in features:
        features[f'sex_{sex}'] = 1
    else:
        features['sex_unknown'] = 1
        
    # Handle Localization
    if f'localization_{localization}' in features:
        features[f'localization_{localization}'] = 1
    else:
        features['localization_unknown'] = 1
        
    # 4. Convert to List in correct order (Alphabetical for categorical)
    # Pandas sorts columns alphabetically. We assume age was the first column in your dataframe concatenation.
    # If age was strictly the first column in your X array:
    final_vector = [features['age']]
    
    # Add Sex columns sorted alphabetically
    sex_cols = sorted([k for k in features if k.startswith('sex_')])
    final_vector.extend([features[k] for k in sex_cols])
    
    # Add Localization columns sorted alphabetically
    loc_cols = sorted([k for k in features if k.startswith('localization_')])
    final_vector.extend([features[k] for k in loc_cols])
    
    return torch.tensor(final_vector, dtype=torch.float32)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --------------------------------
# 3. Main App Layout
# --------------------------------
st.title("🩺 AI Skin Lesion Classifier")
st.markdown("This tool uses a **Fusion Model** combining image analysis (ResNet50) and patient metadata to classify skin lesions.")

col1, col2 = st.columns([1, 1])

model, device = load_model()

with col1:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.header("2. Patient Metadata")
    st.info("Metadata improves accuracy by providing context to the visual data.")
    
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    
    sex = st.selectbox("Sex", ["male", "female", "unknown"])
    
    localization = st.selectbox(
        "Location of Lesion", 
        ["back", "lower extremity", "trunk", "upper extremity", "abdomen", 
         "face", "chest", "foot", "neck", "scalp", "hand", "ear", "genital", "acral"]
    )

    predict_btn = st.button("Analyze Lesion", type="primary", use_container_width=True)

# --------------------------------
# 4. Prediction Logic
# --------------------------------
if predict_btn and uploaded_file is not None and model is not None:
    with st.spinner("Analyzing..."):
        # Prepare inputs
        img_tensor = preprocess_image(image).to(device)
        meta_tensor = preprocess_metadata(age, sex, localization).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor, meta_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get Top Prediction
        top_idx = np.argmax(probabilities)
        top_class = CLASSES[top_idx]
        confidence = probabilities[top_idx] * 100
        
        # Display Results
        st.divider()
        st.success(f"### Prediction: {CLASS_NAMES[top_class]} ({top_class.upper()})")
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
        
        # Display Bar Chart
        st.subheader("Class Probabilities")
        chart_data = pd.DataFrame({
            "Condition": [CLASS_NAMES[c] for c in CLASSES],
            "Probability": probabilities
        })
        st.bar_chart(chart_data, x="Condition", y="Probability")

elif predict_btn and uploaded_file is None:
    st.warning("Please upload an image first.")
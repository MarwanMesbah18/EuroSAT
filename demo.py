import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
import os
import pandas as pd

# --- Model Definition (ResNet50 from scratch) ---

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )
        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# --- Constants ---

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_resnet50_eurosat.pth')

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# --- Load Model (cached) ---

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(num_classes=10)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# --- Prediction ---

def predict(image):
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(10)}

# --- Streamlit UI ---

st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️", layout="wide")

st.title("EuroSAT Land Use Classifier")
st.caption("ResNet50 built from scratch · 97.65% accuracy · 10 land use classes")

# --- Image selection (full width, top) ---

tab_upload, tab_test, tab_dataset = st.tabs(["Upload Image", "Test Images", "Dataset Samples"])

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)

with tab_test:
    test_dir = os.path.join(BASE_DIR, 'test_images')
    if os.path.isdir(test_dir):
        test_files = sorted([f for f in os.listdir(test_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if test_files:
            test_cols = st.columns(min(len(test_files), 5))
            for i, fname in enumerate(test_files):
                with test_cols[i % len(test_cols)]:
                    fpath = os.path.join(test_dir, fname)
                    img = Image.open(fpath)
                    st.image(img, use_container_width=True)
                    if st.button(f"Select", key=f"test_{fname}"):
                        st.session_state['selected_image'] = fpath
                        st.session_state.pop('uploaded_file_key', None)
        else:
            st.info("Drop satellite images in `test_images/` folder to test on unseen data.")
    else:
        st.info("Create a `test_images/` folder and add satellite images.")

with tab_dataset:
    dataset_dir = os.path.join(BASE_DIR, 'Dataset', 'EuroSAT_RGB')
    ds_cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            sample_file = sorted(os.listdir(cls_dir))[0]
            sample_path = os.path.join(cls_dir, sample_file)
            with ds_cols[i % 5]:
                st.image(Image.open(sample_path), use_container_width=True)
                if st.button(cls, key=cls):
                    st.session_state['selected_image'] = sample_path
                    st.session_state.pop('uploaded_file_key', None)

st.divider()

# --- Prediction section (full width, below) ---

# Determine which image to predict on
img_to_predict = None
if uploaded_file is not None:
    img_to_predict = Image.open(uploaded_file)
elif 'selected_image' in st.session_state:
    img_to_predict = Image.open(st.session_state['selected_image'])

if img_to_predict is not None:
    pred_col, table_col = st.columns([1, 2])

    with pred_col:
        st.image(img_to_predict, caption="Selected Image", use_container_width=True)

    with table_col:
        with st.spinner("Classifying..."):
            results = predict(img_to_predict)

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_class = sorted_results[0][0]
        top_conf = sorted_results[0][1]

        st.metric("Predicted Class", top_class, f"{top_conf:.1%}")

        df = pd.DataFrame(sorted_results, columns=["Class", "Confidence"])
        st.dataframe(df.style.format({"Confidence": "{:.2%}"}).bar(subset=["Confidence"], color="#5f9ea0"),
                     use_container_width=True, hide_index=True)
else:
    st.info("Upload an image or select one from the tabs above to see predictions.")

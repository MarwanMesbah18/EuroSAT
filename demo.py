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
st.caption("ResNet50 built from scratch · 97.28% accuracy · 10 land use classes")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload a satellite image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("Or pick from test images")
    test_dir = os.path.join(BASE_DIR, 'test_images')
    if os.path.isdir(test_dir):
        test_files = sorted([f for f in os.listdir(test_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if test_files:
            img_cols = st.columns(min(len(test_files), 4))
            for i, fname in enumerate(test_files):
                with img_cols[i % len(img_cols)]:
                    fpath = os.path.join(test_dir, fname)
                    if st.button(fname, key=f"test_{fname}"):
                        st.session_state['sample_image'] = Image.open(fpath)
                        st.session_state.pop('uploaded_image', None)
            if 'sample_image' in st.session_state and uploaded_file is None:
                st.image(st.session_state['sample_image'], caption="Selected Test Image", use_container_width=True)
        else:
            st.caption("Drop satellite images in test_images/ folder")

    st.divider()
    st.subheader("Or try a dataset sample")
    dataset_dir = os.path.join(BASE_DIR, 'Dataset', 'EuroSAT_RGB')
    sample_cols = st.columns(5)
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            sample_file = sorted(os.listdir(cls_dir))[0]
            sample_path = os.path.join(cls_dir, sample_file)
            with sample_cols[i % 5]:
                if st.button(cls, key=cls):
                    st.session_state['sample_image'] = Image.open(sample_path)
                    st.session_state.pop('uploaded_image', None)

with col2:
    st.subheader("Prediction")
    if uploaded_file is not None or 'sample_image' in st.session_state:
        if 'sample_image' in st.session_state and uploaded_file is None:
            img = st.session_state['sample_image']
        else:
            img = Image.open(uploaded_file)

        with st.spinner("Classifying..."):
            results = predict(img)

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        top_class = sorted_results[0][0]
        top_conf = sorted_results[0][1]

        st.metric("Predicted Class", top_class, f"{top_conf:.1%}")

        df = pd.DataFrame(sorted_results, columns=["Class", "Confidence"])
        st.dataframe(df.style.format({"Confidence": "{:.2%}"}).bar(subset=["Confidence"], color="#5f9ea0"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Upload an image or click a sample class to see predictions.")

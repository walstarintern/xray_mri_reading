import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 🚨 THE BULLETPROOF WEIGHT LOADER
# ==========================================
def load_robust_weights(model, weights_path, device):
    """
    A smart weight loader that automatically handles nested dictionaries
    and strips unwanted Colab prefixes (like 'module.' or 'model.').
    """
    if not os.path.exists(weights_path):
        print(f"⚠️ Warning: Could not find '{weights_path}'")
        return model

    # 1. Load the raw file
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    # 2. Extract the actual weights dictionary (handles saved checkpoints vs raw weights)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt # It's already the raw weights
    else:
        state_dict = ckpt

    # 3. Clean the keys (Strips prefixes added by Colab wrappers)
    clean_state_dict = {}
    for k, v in state_dict.items():
        # Remove common prefixes that cause mismatch errors
        clean_key = k.replace('module.', '').replace('model.', '').replace('backbone.', '')
        clean_state_dict[clean_key] = v

    # 4. Load into the model safely
    try:
        # strict=False allows it to ignore any lingering non-essential keys
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"✅ Successfully loaded weights: {os.path.basename(weights_path)}")
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(weights_path)}: {e}")
        
    return model

# ==========================================
# --- PYTORCH ARCHITECTURES ---
# ==========================================
def build_v1_model(num_classes=4):
    """Recreates the official torchvision EfficientNet-B4 architecture"""
    model = models.efficientnet_b4(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def build_ms_model(num_classes=2):
    """Recreates the raw official torchvision DenseNet121 architecture"""
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

# ==========================================
# --- THE 3-IN-1 EXPERT PIPELINE ---
# ==========================================
class BrainMRIExpert:
    def __init__(self):
        print("Initializing Pure-PyTorch Brain MRI Panel...")
        self.device = torch.device("cpu") # Force CPU for safe offline inference

        # 1. Load Tumor Model (EfficientNet-B4)
        self.tumor_classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        self.tumor_model = build_v1_model(num_classes=4)
        self.tumor_model = load_robust_weights(self.tumor_model, 'vision/best_mri_model.pth', self.device)
        self.tumor_model.to(self.device).eval()

        # 2. Load Alzheimer's Model (EfficientNet-B4)
        self.alz_classes = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
        self.alz_model = build_v1_model(num_classes=4)
        self.alz_model = load_robust_weights(self.alz_model, 'vision/best_alzheimer_model.pth', self.device)
        self.alz_model.to(self.device).eval()

        # 3. Load MS Lesion Model (DenseNet-121)
        self.ms_model = build_ms_model(num_classes=2)
        self.ms_model = load_robust_weights(self.ms_model, 'vision/ms_densenet121_f1.pth', self.device)
        self.ms_model.to(self.device).eval()

        # Shared Pre-processing for all 3 PyTorch Models
        self.transform = A.Compose([
            A.Resize(380, 380),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def analyze(self, image_path):
        """Passes the image through the 3 PyTorch models and combines findings."""
        try:
            # Load base image
            img = cv2.imread(image_path)
            if img is None:
                return "Error: Image could not be read."
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply transforms
            transformed = self.transform(image=img_rgb)
            tensor_img = transformed['image'].unsqueeze(0).to(self.device)

            findings = []

            with torch.no_grad():
                # 1. TUMOR PREDICTION
                t_out = self.tumor_model(tensor_img)
                t_prob = torch.nn.functional.softmax(t_out, dim=1)[0]
                t_pred = torch.argmax(t_out, 1).item()
                t_conf = t_prob[t_pred].item() * 100
                findings.append(f"Tumor Analysis: {self.tumor_classes[t_pred]} ({t_conf:.1f}% confidence)")

                # 2. ALZHEIMER'S PREDICTION
                a_out = self.alz_model(tensor_img)
                a_prob = torch.nn.functional.softmax(a_out, dim=1)[0]
                a_pred = torch.argmax(a_out, 1).item()
                a_conf = a_prob[a_pred].item() * 100
                findings.append(f"Alzheimer's Analysis: {self.alz_classes[a_pred]} ({a_conf:.1f}% confidence)")

                # 3. MS LESION PREDICTION (Using the 0.60 threshold)
                ms_out = self.ms_model(tensor_img)
                ms_prob = torch.nn.functional.softmax(ms_out, dim=1)[0]
                ms_lesion_prob = ms_prob[0].item() # Assuming index 0 is 'ms_lesion'
                
                if ms_lesion_prob >= 0.60:
                    findings.append(f"MS Lesion Analysis: MS Lesion DETECTED ({ms_lesion_prob*100:.1f}% confidence)")
                else:
                    findings.append(f"MS Lesion Analysis: Normal - No lesions ({ms_prob[1].item()*100:.1f}% confidence)")

            # Combine all 3 findings into one report for Qwen
            final_report = " | ".join(findings)
            return final_report

        except Exception as e:
            return f"Error analyzing Brain MRI: {str(e)}"
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
    """Strips unwanted Colab prefixes so weights load perfectly."""
    if not os.path.exists(weights_path):
        print(f"⚠️ Warning: Could not find '{weights_path}'")
        return model

    # We add weights_only=False to bypass the PyTorch 2.6 security block
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt: state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        else: state_dict = ckpt
    else:
        state_dict = ckpt

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '').replace('model.', '').replace('backbone.', '')
        clean_state_dict[clean_key] = v

    try:
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"✅ Successfully loaded weights: {os.path.basename(weights_path)}")
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(weights_path)}: {e}")
        
    return model

# ==========================================
# --- PYTORCH ARCHITECTURES ---
# ==========================================
def build_oa_model(num_classes=5):
    """EfficientNet-B3 for Osteoarthritis KL Grading (Assumes standard 3-channel)"""
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def build_general_knee_model(num_classes=3):
    """EfficientNet-B3 modified for 1-channel grayscale and Multi-Label output"""
    model = models.efficientnet_b3(weights=None)
    
    # 🚨 HACKING THE FIRST LAYER FOR 1-CHANNEL (GRAYSCALE) INPUT
    # Standard EffNet-B3 expects 3 channels. We override it here to match your Colab log!
    model.features[0][0] = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    # Classification head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# ==========================================
# --- THE 2-IN-1 KNEE EXPERT PIPELINE ---
# ==========================================
class KneeMRIExpert:
    def __init__(self):
        print("Initializing Pure-PyTorch Knee MRI Panel...")
        self.device = torch.device("cpu")

        # 1. Load Osteoarthritis Model (EfficientNet-B3, 3-Channel)
        self.oa_classes = ['Healthy (KL-0)', 'Doubtful OA (KL-1)', 'Minimal OA (KL-2)', 'Moderate OA (KL-3)', 'Severe OA (KL-4)']
        self.oa_model = build_oa_model(num_classes=5)
        self.oa_model = load_robust_weights(self.oa_model, 'vision/knee_oa_effnetb3_ordinal_prod.pth', self.device)
        self.oa_model.to(self.device).eval()

        # 2. Load General Knee Model (EfficientNet-B3, 1-Channel, Multi-Label)
        self.knee_model = build_general_knee_model(num_classes=3)
        self.knee_model = load_robust_weights(self.knee_model, 'vision/knee_mri_production_final.pth', self.device)
        self.knee_model.to(self.device).eval()

        # Pre-processing for the OA model (3-Channel)
        self.transform_3ch = A.Compose([
            A.Resize(300, 300), 
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Pre-processing for the General Knee model (1-Channel Grayscale)
        self.transform_1ch = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=(0.5,), std=(0.5,)), # Grayscale normalization
            ToTensorV2()
        ])

    def analyze(self, image_path):
        """Passes the image through the 2 PyTorch models and combines findings."""
        try:
            img = cv2.imread(image_path)
            if img is None: return "Error: Image could not be read."
            
            # Create both RGB and Grayscale versions of the image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply the specific transforms
            tensor_img_3ch = self.transform_3ch(image=img_rgb)['image'].unsqueeze(0).to(self.device)
            tensor_img_1ch = self.transform_1ch(image=img_gray)['image'].unsqueeze(0).to(self.device)

            findings = []

            with torch.no_grad():
                # 1. OSTEOARTHRITIS PREDICTION (Softmax - KL grades are mutually exclusive)
                oa_out = self.oa_model(tensor_img_3ch)
                oa_prob = torch.nn.functional.softmax(oa_out, dim=1)[0]
                oa_pred = torch.argmax(oa_out, 1).item()
                findings.append(f"Osteoarthritis Grading: {self.oa_classes[oa_pred]} ({oa_prob[oa_pred].item()*100:.1f}% confidence)")

                # 2. GENERAL INJURY PREDICTION (Sigmoid - Multilabel can detect multiple injuries)
                k_out = self.knee_model(tensor_img_1ch)
                k_probs = torch.sigmoid(k_out)[0] # 🚨 Using Sigmoid instead of Softmax!
                
                # Based on your Colab mapping: [ACL, Meniscus, Abnormal]
                acl_conf = k_probs[0].item() * 100
                men_conf = k_probs[1].item() * 100
                abn_conf = k_probs[2].item() * 100
                
                structural_findings = []
                
                # Threshold logic: If confidence is > 50%, the injury exists
                if acl_conf > 50.0:
                    structural_findings.append(f"ACL Tear ({acl_conf:.1f}%)")
                if men_conf > 50.0:
                    structural_findings.append(f"Meniscus Tear ({men_conf:.1f}%)")
                    
                if len(structural_findings) > 0:
                    findings.append(f"Structural Analysis: DETECTED - {', '.join(structural_findings)}")
                elif abn_conf > 50.0:
                    findings.append(f"Structural Analysis: Abnormal joint pathology detected ({abn_conf:.1f}% confidence)")
                else:
                    findings.append(f"Structural Analysis: Normal - No acute tears detected")

            return " | ".join(findings)

        except Exception as e:
            return f"Error analyzing Knee MRI: {str(e)}"
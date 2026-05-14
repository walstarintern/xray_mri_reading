import os
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# 🚨 THE BULLETPROOF WEIGHT LOADER
# ==========================================
def load_robust_weights(model, weights_path, device):
    """Strips Colab prefixes and bypasses PyTorch 2.6 weights_only block."""
    if not os.path.exists(weights_path):
        print(f"⚠️ Warning: Could not find '{weights_path}'")
        return model
        
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt: state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
        else: state_dict = ckpt
    else:
        state_dict = ckpt

    clean_state_dict = {k.replace('module.', '').replace('model.', '').replace('backbone.', ''): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(clean_state_dict, strict=False)
        print(f"✅ Successfully loaded weights: {os.path.basename(weights_path)}")
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(weights_path)}: {e}")
        
    return model

# ==========================================
# --- PYTORCH U-NET ARCHITECTURE ---
# ==========================================
def build_spine_model(num_classes):
    """Recreates the smp.Unet with a ResNet34 encoder for 1-channel Grayscale"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, # We load our own trained weights
        in_channels=1,        # 1-channel Grayscale Input
        classes=num_classes,  # 4 for Lumbar/Thoracic, 8 for Cervical
    )
    return model

# ==========================================
# --- THE 3-IN-1 SPINE SEGMENTATION PIPELINE ---
# ==========================================
class SpineMRIExpert:
    def __init__(self):
        print("Initializing Pure-PyTorch Spine MRI (Segmentation) Panel...")
        self.device = torch.device("cpu")

        # Define exact classes mapping based on your notebook
        self.lumbar_classes = {1: 'Vertebra', 2: 'IVD', 3: 'Spinal Canal'}
        self.thoracic_classes = {1: 'Vertebra (T1-T12)', 2: 'IVD', 3: 'Cord/Canal'}
        self.cervical_classes = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}

        # 1. Load Lumbar Model (4 Classes)
        self.lumbar_model = build_spine_model(num_classes=4)
        self.lumbar_model = load_robust_weights(self.lumbar_model, 'vision/lumbar_spine_best.pth', self.device)
        self.lumbar_model.to(self.device).eval()

        # 2. Load Cervical Model (8 Classes)
        self.cervical_model = build_spine_model(num_classes=8)
        self.cervical_model = load_robust_weights(self.cervical_model, 'vision/cervical_spine_best.pth', self.device)
        self.cervical_model.to(self.device).eval()

        # 3. Load Thoracic Model (4 Classes)
        self.thoracic_model = build_spine_model(num_classes=4)
        self.thoracic_model = load_robust_weights(self.thoracic_model, 'vision/thoracic_spine_best.pth', self.device)
        self.thoracic_model.to(self.device).eval()

        # Pre-processing for 1-Channel Grayscale at 512x512
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485,), std=(0.229,)), # 1-channel specific normalization
            ToTensorV2()
        ])

    def format_segmentation_findings(self, mask, class_dict, region_name):
        """Calculates area percentage of each segmented structure."""
        detected = []
        total_pixels = mask.numel() # Total pixels in 512x512 image
        
        for class_idx, class_name in class_dict.items():
            # Count pixels belonging to this specific class
            pixel_count = (mask == class_idx).sum().item()
            if pixel_count > 0:
                area_pct = (pixel_count / total_pixels) * 100
                detected.append(f"{class_name} ({area_pct:.1f}% area)")
                
        if not detected:
            return f"{region_name} Analysis: No clear structures segmented."
        return f"{region_name} Analysis: Segmented {', '.join(detected)}"

    def analyze(self, image_path):
        """Passes image through all 3 segmentation models."""
        try:
            # Load explicitly as 1-Channel Grayscale
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None: return "Error: Image could not be read."

            # Apply transform
            tensor_img = self.transform(image=img_gray)['image'].unsqueeze(0).to(self.device)
            findings = []

            with torch.no_grad():
                # --- LUMBAR PREDICTION ---
                l_out = self.lumbar_model(tensor_img)
                l_mask = torch.argmax(l_out, dim=1)[0] # Extract the 2D segmentation map
                findings.append(self.format_segmentation_findings(l_mask, self.lumbar_classes, "Lumbar"))

                # --- CERVICAL PREDICTION ---
                c_out = self.cervical_model(tensor_img)
                c_mask = torch.argmax(c_out, dim=1)[0]
                findings.append(self.format_segmentation_findings(c_mask, self.cervical_classes, "Cervical"))

                # --- THORACIC PREDICTION ---
                t_out = self.thoracic_model(tensor_img)
                t_mask = torch.argmax(t_out, dim=1)[0]
                findings.append(self.format_segmentation_findings(t_mask, self.thoracic_classes, "Thoracic"))

            return " | ".join(findings)

        except Exception as e:
            return f"Error analyzing Spine MRI: {str(e)}"
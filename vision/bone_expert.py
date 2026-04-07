import os
import cv2
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. We must define the exact architecture used in Colab to load the weights
class BoneFractureModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_l",
            pretrained=False, # Offline mode, we don't need to download weights from the internet
            num_classes=0,
            global_pool="avg"
        )
        
        in_feats = self.backbone.num_features  # 1280
        
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_feats, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class BoneExpert:
    def __init__(self, weights_path="vision/bone_fracture_efficientv2l_best.pth"):
        print("Loading Production Bone Expert Model (EfficientNetV2-L)...")
        self.device = torch.device("cpu") # Forcing CPU for offline use
        
        # Initialize our custom architecture
        self.model = BoneFractureModel(num_classes=2)
        
        # Load the custom trained weights
        if os.path.exists(weights_path):
            try:
                # Load the checkpoint file
                ckpt = torch.load(weights_path, map_location=self.device)
                # Colab saved a dictionary, we only need the 'model_state_dict' piece
                self.model.load_state_dict(ckpt["model_state_dict"])
                print("✅ EfficientNetV2-L weights loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")
        else:
            print(f"⚠️ Warning: Could not find '{weights_path}'. Model will guess randomly!")
            
        self.model.eval() # Set to evaluation mode
        
        # Exact image transformations used during Colab validation (Crucial for accuracy)
        self.transform = A.Compose([
            A.Resize(384, 384),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), # The magic contrast enhancer!
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def analyze(self, image_path):
        try:
            # 1. Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return "Error: Image could not be read."
            
            # ==========================================
            # 🚨 THE RADIOLOGIST GRAYSCALE FIX
            # ==========================================
            # Strip all color (like the blue monitor tint) by forcing pure grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # EfficientNet requires 3 channels, so we convert the pure gray back to 3-channels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # ==========================================
            
            # 2. Apply Albumentations transforms
            img_tensor = self.transform(image=img)["image"].unsqueeze(0).to(self.device)

            # 3. Run CPU Inference
            with torch.no_grad():
                output = self.model(img_tensor)
                
            # 4. Interpret the 2-class output using Softmax
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            prob_normal = probs[0].item()
            prob_fracture = probs[1].item()
            
            # 5. Format output
            if prob_fracture > prob_normal:
                confidence = round(prob_fracture * 100, 2)
                return f"EfficientNet AI Analysis: ABNORMAL bone structure detected (Confidence: {confidence}%). Likely indicates a fracture, implanted hardware, or degenerative disease."
            else:
                confidence = round(prob_normal * 100, 2)
                return f"EfficientNet AI Analysis: NORMAL bone structure (Confidence: {confidence}%). No obvious fractures or acute abnormalities detected."
                
        except Exception as e:
            return f"Error analyzing bone X-ray: {str(e)}"
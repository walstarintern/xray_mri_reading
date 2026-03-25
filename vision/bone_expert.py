import torch
from torchvision import transforms
from PIL import Image
from .mura_model import densenet169  # Importing your custom Github code!

class BoneExpert:
    def __init__(self, weights_path=None):
        print("Loading Custom Bone Expert Model (MURA)...")
        # Load the custom architecture
        self.model = densenet169(pretrained=False)
        
        # If you have the downloaded .pth file, load the weights into the architecture
        if weights_path:
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
                print("Custom MURA weights loaded successfully.")
            except Exception as e:
                print(f"Error loading weights: {e}")
                
        self.model.eval() # Set to evaluation mode
        
        # MURA models require specific image sizing and normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, image_path):
        # 1. Load and format the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0) # Add batch dimension for CPU

        # 2. Run CPU Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # 3. Interpret the Sigmoid output (0 to 1)
        probability = output.item()
        
        if probability > 0.5:
            confidence = round(probability * 100, 2)
            return f"Abnormality Detected (Confidence: {confidence}%). Likely indicative of a fracture, lesion, or hardware abnormality."
        else:
            confidence = round((1 - probability) * 100, 2)
            return f"Normal Bone Structure (Confidence: {confidence}%). No significant abnormalities detected."
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image
import numpy as np

class ChestExpert:
    def __init__(self):
        # Loads the pre-trained DenseNet model for chest X-rays
        print("Loading Chest Expert Model...")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval() # Set strictly to evaluation/inference mode
        
        # Standard preprocessing for this specific model
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def analyze(self, image_path):
        # 1. Load the image and convert to grayscale
        img = Image.open(image_path).convert('L')
        img = np.array(img)
        
        # 2. Normalize pixel values for the model
        img = xrv.datasets.normalize(img, 255) 
        
        # 3. Format into a PyTorch Tensor for the CPU
        img = img[None, ...] # Add color channel
        img = self.transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0) # Add batch dimension

        # 4. Run CPU Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # 5. Extract findings and format into text
        pathologies = self.model.pathologies
        probabilities = outputs[0].detach().numpy()
        
        results = dict(zip(pathologies, probabilities))
        
        # Filter out low-probability guesses to keep the LLM focused
        significant_findings = {k: round(float(v) * 100, 2) for k, v in results.items() if v > 0.5}
        
        if not significant_findings:
            return "No significant abnormalities detected in the chest X-ray with high confidence."
            
        # Format as a clean string for the LLM
        findings_text = ", ".join([f"{k} ({v}% probability)" for k, v in significant_findings.items()])
        return findings_text
from .chest_expert import ChestExpert
from .bone_expert import BoneExpert
from .brain_mri_expert import BrainMRIExpert
from .knee_mri_expert import KneeMRIExpert # <-- Add this import
from .spine_mri_expert import SpineMRIExpert

class VisionRouter:
    def __init__(self):
        self.chest_expert = None
        self.bone_expert = None
        self.brain_mri_expert = None
        self.knee_mri_expert = None # <-- Initialize to None
        self.spine_mri_expert = None

    def route_and_analyze(self, image_path, scan_type, body_part_tag):
        # --- X-RAY ROUTING ---
        if scan_type == "X-Ray":
            if body_part_tag.lower() == "chest":
                if self.chest_expert is None: self.chest_expert = ChestExpert()
                return self.chest_expert.analyze(image_path)
            elif body_part_tag.lower() in ["arm", "leg", "hand", "bone"]:
                if self.bone_expert is None: self.bone_expert = BoneExpert()
                return self.bone_expert.analyze(image_path)
                
        # --- MRI ROUTING ---
        elif scan_type == "MRI":
            if body_part_tag.lower() == "brain":
                if self.brain_mri_expert is None: self.brain_mri_expert = BrainMRIExpert()
                return self.brain_mri_expert.analyze(image_path)
            
            # 🚨 ADD THE NEW KNEE ROUTE HERE 🚨
            elif body_part_tag.lower() == "knee":
                if self.knee_mri_expert is None: self.knee_mri_expert = KneeMRIExpert()
                return self.knee_mri_expert.analyze(image_path)

            elif body_part_tag.lower() == "spine":
                if self.spine_mri_expert is None: self.spine_mri_expert = SpineMRIExpert()
                return self.spine_mri_expert.analyze(image_path)
                
        return "Unrecognized scan type or body part."
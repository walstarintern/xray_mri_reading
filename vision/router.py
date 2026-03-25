from .chest_expert import ChestExpert
from .bone_expert import BoneExpert

class VisionRouter:
    def __init__(self):
        # We only initialize the experts when needed to save RAM
        self.chest_expert = None
        self.bone_expert = None

    def route_and_analyze(self, image_path, body_part_tag):
        if body_part_tag.lower() == "chest":
            if self.chest_expert is None:
                self.chest_expert = ChestExpert()
            return self.chest_expert.analyze(image_path)
            
        elif body_part_tag.lower() in ["arm", "leg", "hand", "bone"]:
            if self.bone_expert is None:
                self.bone_expert = BoneExpert()
            return self.bone_expert.analyze(image_path)
            
        else:
            return "Unrecognized body part. Please specify Chest or Bone."
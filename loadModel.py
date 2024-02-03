import sys
from segment_anything import sam_model_registry, SamPredictor

def sam_vit_h_4b8939_predictor_with_cuda():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    #if cuda is avliable then just replace below with device = "cuda"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor

from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_bussam.build_sam_us import bussam_model_registry


def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == "BUSSAM":
        model = bussam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model

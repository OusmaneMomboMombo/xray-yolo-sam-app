import numpy as np

def void_rate_from_masks(masks, cls_names):
    # masks: tensor/array [N, H, W] de 0/1 ; cls_names: liste des classes pour chaque masque (ex: ['chip','void',...])
    chip_area = 0
    void_area = 0
    for m, c in zip(masks, cls_names):
        area = int(np.array(m).sum())
        if c.lower() == "chip":
            chip_area += area
        elif c.lower() == "void":
            void_area += area
    return (void_area / chip_area * 100.0) if chip_area > 0 else 0.0

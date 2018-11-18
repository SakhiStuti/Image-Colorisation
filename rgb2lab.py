import skimage
import numpy as np
import torch

class RGB_to_LAB:

    def __init__(self):
        self.rgb2lab = skimage.color.rgb2lab

    def __call__(self, img):
        nd_image = np.array(img)
        lab_image = self.rgb2lab(nd_image).transpose(0,1,2)
        lab_image = torch.from_numpy(lab_image.astype(np.float32))
        return lab_image

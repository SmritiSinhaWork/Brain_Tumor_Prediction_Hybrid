import torch
import cv2
import numpy as np


def predict(model, image):
    import torch
    import numpy as np

    img = image
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    img = torch.tensor(img, dtype=torch.float32)

    output = model(img)

    probs = torch.softmax(output, dim=1)
    _, pred = torch.max(probs, 1)

    return pred.item()
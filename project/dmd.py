import numpy as np

def dmd_process(img):
    img = img.astype(float)

    img = (img - np.mean(img)) / (np.std(img) + 1e-8)

    U, S, Vt = np.linalg.svd(img, full_matrices=False)

    k = 15
    img_recon = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

    img_recon = np.nan_to_num(img_recon)

    img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min() + 1e-8)

    return img_recon
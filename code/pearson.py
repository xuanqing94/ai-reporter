import numpy as np
import random
from PIL import Image
import glob
import os
import cv2
from scipy import stats
import pickle

gradient = True


def read_img(f):
    img = cv2.imread(f)
    if gradient:
        img = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    return img


def calc():
    tar = []
    pred = []
    truth = glob.glob(f"./results/newstage_f2/truth*.png")
    truth.sort()
    print(truth)
    tarimages = []  #[read_img(img) for img in truth]
    for f in truth:
        img = read_img(f)
        save_f = f.replace('results', 'gradient')
        cv2.imwrite(save_f, img)
        tarimages.append(img)
    for img in tarimages:
        dimtar = img[:, :, 0].astype(np.float)
        tar.append(dimtar)

    prediction = glob.glob(f"./results/newstage_f2/pred*.png")
    prediction.sort()

    predimages = []  #[read_img(image) for image in prediction]
    for f in prediction:
        img = read_img(f)
        save_f = f.replace('results', 'gradient')
        cv2.imwrite(save_f, img)
        predimages.append(img)

    for image in predimages:
        dimpred = image[:, :, 0].astype(np.float)
        pred.append(dimpred)

    pairs = list(zip(tar, pred))
    # sample without replacement
    # samples = random.sample(pairs, 50)
    # scores
    R_scores = []
    for t, p in pairs:
        t = t.reshape(-1)
        p = p.reshape(-1)
        corr = stats.pearsonr(t, p)[0]
        print(corr)
        #indicator = (t < 100)
        #corr = np.sqrt(
        #    np.sum(np.abs(t - p)**2 * indicator) / np.sum(indicator))
        # R_scores.append(corr[0])
        R_scores.append(corr)
    print('Pearsons correlation: %.5f' % np.mean(R_scores))
    return R_scores


"""
train_sizes = ["305", "509"]
results = {}
for size in train_sizes:
    print(size)
    results[size] = calc(size)
"""
calc()

with open('./R-scores.pkl', 'wb') as writer:
    pickle.dump(results, writer)

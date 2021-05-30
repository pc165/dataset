import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import random


def show(im):
    plt.imshow(im, cmap='gray')
    plt.show()


def getClassLabel(name: str):
    if name[-1].isdigit():
        name = name[:-1]
    if name.find('-') != -1:
        return name.split('-')
    return [name]


LABEL_NUMBER = {}
NUMBER_LABEL = {}


def getNumber(name) -> int:
    if name not in LABEL_NUMBER:
        NUMBER_LABEL[len(LABEL_NUMBER)] = name
        LABEL_NUMBER[name] = len(LABEL_NUMBER)
    return LABEL_NUMBER[name]


def get_heuristic(bb1, bb2):
    # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner


    Returns
    -------
    float
        in [0, 1]
    """
    x, y, w, h = bb1
    b1 = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}
    x, y, w, h = bb2
    b2 = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}
    assert b1['x1'] < b1['x2']
    assert b1['y1'] < b1['y2']
    assert b2['x1'] < b2['x2']
    assert b2['y1'] < b2['y2']

    x_left = max(b1['x1'], b2['x1'])
    y_top = max(b1['y1'], b2['y1'])
    x_right = min(b1['x2'], b2['x2'])
    y_bottom = min(b1['y2'], b2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    bb1_area = (b1['x2'] - b1['x1'] + 1) * (b1['y2'] - b1['y1'] + 1)

    return intersection_area / bb1_area


def placeObject(obj, background, label):
    background = background.copy()
    w = int(obj.shape[1] * random.randint(30, 60) / 100)
    h = int(obj.shape[0] * random.randint(30, 60) / 100)
    dim = (w, h)
    obj = cv2.resize(obj, dim, interpolation=cv2.INTER_AREA)
    try:
        x_offset = random.randint(0, background.shape[1] - obj.shape[1])
        y_offset = random.randint(0, background.shape[0] - obj.shape[0])
    except:
        w = int(obj.shape[1] * random.randint(15, 30) / 100)
        h = int(obj.shape[0] * random.randint(15, 30) / 100)
        dim = (w, h)
        obj = cv2.resize(obj, dim, interpolation=cv2.INTER_AREA)
        x_offset = random.randint(0, background.shape[1] - obj.shape[1])
        y_offset = random.randint(0, background.shape[0] - obj.shape[0])

    x1, x2 = x_offset, x_offset + obj.shape[1]
    y1, y2 = y_offset, y_offset + obj.shape[0]

    back = background[y1:y2, x1:x2]
    h, w, _ = back.shape
    obj = obj[:h, :w]

    mask = obj.any(axis=-1).astype('float')
    mask = mask[:, :, np.newaxis]
    mask = np.concatenate((mask, mask, mask), axis=-1)

    background[y1:y2, x1:x2] = obj * mask + (1 - mask) * back

    h, w, _ = obj.shape
    bb = x_offset, y_offset, w, h
    return background, (getNumber(label), bb)


def noisy(image, amount=0.05, s_vs_p=0.5):
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords] = 0
    return out


ITEMS = {}
BACKGROUND = [cv2.imread(i) for i in glob("./backgrounds/*.png")]
IMGS = [(cv2.imread(i), Path(i)) for i in glob("./rawdata/*.png")]

for img, i in IMGS:
    fname = i.stem
    name = getClassLabel(fname)
    mask = img.any(axis=-1).astype('uint8')
    mask = cv2.dilate(mask, np.ones((5, 5))) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.boundingRect(i) for i in contours]
    contours.sort(key=lambda x: x[0])
    contours = [c for c in contours if c[2] > 50 and c[3] > 50]
    for (x, y, w, h), cl in zip(contours, name):
        obj = img[y:y+h, x:x+w].copy()
        if cl not in ITEMS:
            ITEMS[cl] = [obj]
        else:
            ITEMS[cl].append(obj)


def bbReduce(bb: list):
    bbpriority = {j: i for i, j in enumerate(bb)}
    bb2 = {}
    for i in bb:
        for j in bb:
            if i == j:
                continue

            v = get_heuristic(i[1], j[1])

            if v == 0:
                continue

            bb2[(i, j)] = v


    for (i, j), val in bb2.items():
        if val > 0.45:
            if bbpriority[i] < bbpriority[j]:
                try:
                    bb.remove(i)
                except:
                    pass
    # print(bb2, '\n')

    return bb


ITEM_LIST = list(ITEMS.items())
NUM_IMG = 256
random.seed(111)
for name in range(int(NUM_IMG * 1.5)):
    classes = random.sample(ITEM_LIST, random.randint(2, 6))
    objs = [(i[0], random.choice(i[1])) for i in classes]
    background = random.choice(BACKGROUND)
    bb = []
    for label, obj in objs:
        background, b = placeObject(obj, background, label)
        bb.append(b)

    bb2 = bbReduce(bb)

    path = 'train' if name > NUM_IMG == 0 else 'test'

    out = open(f"./{path}/labels/{name}.txt", 'w')
    out2 = open(f"./{path}/labels/{name}_noisy.txt", 'w')
    im_h, im_w, _ = background.shape
    anotated = background.copy()
    for label, (x, y, w, h) in bb2:
        cv2.rectangle(anotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(anotated, (f'{label} {NUMBER_LABEL[label]}'), (x, y+10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 0))

        x_center = (x + w/2) / im_w
        y_center = (y + h/2) / im_h
        w = w / im_w
        h = h / im_h
        s = f'{label} {x_center} {y_center} {w} {h}\n'
        out.write(s)
        out2.write(s)
    out.close()
    out2.close()
    cv2.imwrite(f"./{path}/labels/{name}.jpg", anotated)
    cv2.imwrite(f"./{path}/images/{name}.jpg", background)
    cv2.imwrite(f"./{path}/images/{name}_noisy.jpg", noisy(background))

l = list(LABEL_NUMBER.items())
l.sort(key=lambda x: x[1])
l = [i[0] for i in l]
s = f"""train: ../dataset/train/images/
val: ../dataset/test/images/

# number of classes
nc: {len(l)}

# class names
names: {l}"""
with open('data.yaml', 'w') as o:
    o.write(s)

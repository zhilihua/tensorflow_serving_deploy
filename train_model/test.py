import numpy as np
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

def predict(model, img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

def plot_preds(image, preds):
    index = np.argmax(preds)
    labels = ("cat", "dog")

    plt.imshow(image)
    plt.axis('off')
    plt.title(labels[index]+":"+"%s"%preds[index])
    plt.show()

if __name__ == '__main__':
    import os
    target_size = (224, 224) #(width, height)

    imagesPath = []
    imgs_path = "dataset/test"
    for root, dirs, files in os.walk(imgs_path):
        for file in files:
            path = os.path.join(imgs_path, file)
            imagesPath.append(path)

    path = 'savemodel_1fc256.h5'
    model = load_model(path)

    for p in imagesPath:
        img = Image.open(p)
        preds = predict(model, img, target_size)
        plot_preds(img, preds)
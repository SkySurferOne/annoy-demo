import os

import h5py
from annoy import AnnoyIndex
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_img_dir(name):
    dir = os.getcwd()
    return dir + '/{}'.format(name)


def get_data(filename):
    f = h5py.File(get_img_dir(filename), 'r')
    return f


def convert_to_1d(image):
    return np.array(image.ravel())


def reshape_with_denorm(image, shape):
    image = image * 255
    return image.reshape(shape).astype(int)


def show_image(arr):
    img2 = Image.fromarray(arr, 'RGB')
    img2.show()


def show_image_plt(arr):
    plt.imshow(arr)
    plt.show()


def plot_gallery(title, images, n_col=5, n_row=3, image_shape=(64, 64)):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(reshape_with_denorm(comp, image_shape))
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()


def save_annoy(filename, X, f, n_trees=10, metric='angular', force=False):
    if not os.path.isfile(get_img_dir(filename)) or force:
        t = AnnoyIndex(f, metric=metric)

        for i, v in enumerate(X):
            t.add_item(i, v)

        t.build(n_trees)
        t.save(filename)


def load_annoy(filename):
    u = AnnoyIndex(f)
    u.load(filename)

    return u


def compute_annoy(X, f, n_trees=10, metric='angular'):
    t = AnnoyIndex(f, metric=metric)

    for i, v in enumerate(X):
        t.add_item(i, v)

    t.build(n_trees)
    t.save('food.ann')

    u = AnnoyIndex(f)
    u.load('food.ann')

    return u


def get_n_closest(index, v, n, search_k=-1):
    return index.get_nns_by_vector(v, n, search_k=search_k)


if __name__ == '__main__':
    f = get_data('food_test_c101_n1000_r128x128x3.h5')

    print(list(f.keys()))

    categories = np.array(f['category'])
    cat_names = np.array(f['category_names'])
    images = np.array(f['images'])
    cat_flat = np.array([cat_names[categories[i]][0] for i in range(len(images))])

    shape = images[0].shape
    images_flatten = np.array([convert_to_1d(v) for v in images])
    images_norm = images_flatten / 255
    f = len(images_norm[0])

    X, X_test, y, y_test = train_test_split(images_norm, cat_flat, test_size=0.2, random_state=10)

    index_filename = 'food.ann'
    save_annoy(index_filename, X, f)
    u = load_annoy(index_filename)

    ex_i = np.random.randint(len(X_test))
    ex_item = X_test[ex_i]
    ex_item_label = y_test[ex_i]

    closest = get_n_closest(u, ex_item, 15)
    closest_items = [X[i] for i in closest]
    closest_items_lab = [y[i] for i in closest]

    print('Label of ex. image: ', ex_item_label)
    show_image_plt(reshape_with_denorm(ex_item, shape))

    print('Labels of closest images: ')
    for i, item in enumerate(closest_items_lab):
        print(i, ': ', item)
    plot_gallery('Closest images', closest_items, n_col=5, n_row=3, image_shape=shape)

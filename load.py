import numpy as np
import pathlib
import h5py

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# code taken from / inspired by https://github.com/XeniaOhmer/language_perception_communication_games/

def get_shape_color_labels(full_labels,
                           trait_idxs=(2, 3, 4)):
    possible_values = [[] for _ in range(len(trait_idxs))]
    # trait_names_by_idx = ['floorHue', 'wallHue', 'color', 'scale', 'shape', 'orientation']

    # default trait_idxs set to (2,3,4) corresponding to color, scale, and shape
    extracted_traits = [tuple(entry) for entry in list(full_labels[:, trait_idxs])]

    for tup in extracted_traits:
        for (idx, entry) in enumerate(tup):
            possible_values[idx].append(entry)
    for (idx, p) in enumerate(possible_values):
        possible_values[idx] = sorted(set(p))

    # since there were only 4 possible shapes, we extracted 4 approximately equally spaced
    # values from the other two traits. balance_type == 2 was used for our experiments. The first
    # list in idxes_to_keep selects values for color and the second list selects values for
    # the object scale, based on the configuration set by the extracted_traits variable

    idxes_to_keep = [[0, 2, 4, 8], [0, 3, 5, 7]]
    values_to_keep = [[], []]

    for idx in [0, 1]:
        for val_idx in idxes_to_keep[idx]:
            values_to_keep[idx].append(possible_values[idx][val_idx])
    filtered_traits = []
    keeper_idxs = []
    for (idx, traits) in enumerate(extracted_traits):
        if traits[0] in values_to_keep[0] and traits[1] in values_to_keep[1]:
            filtered_traits.append(traits)
            keeper_idxs.append(idx)
    
    extracted_traits = filtered_traits

    unique_traits = sorted(set(extracted_traits))
    labels = np.zeros((len(extracted_traits), len(unique_traits)))

    # these dictionaries are used to convert between indices for one-hot target vectors
    # and the corresponding trait combination that that entry represents, which defines the class
    # composition of the classification problem
    label2trait_map = dict()
    trait2label_map = dict()

    for (i, traits) in enumerate(unique_traits):
        trait2label_map[traits] = i
        label2trait_map[i] = traits

    # generating one-hot labels
    for (i, traits) in enumerate(extracted_traits):
        labels[i, trait2label_map[traits]] = 1
    return labels, keeper_idxs


def load_data(normalize=True,
              subtract_mean=True,
              data_path='3dshapes/3dshapes.h5'):
    try:
        print("Loading data from: ", data_path)
        dataset = h5py.File(data_path, 'r')
    except:
        raise ValueError(f"{data_path} not found. Please download the shapes3d dataset or specify the correct path.")

    img_data = dataset['images'][:]
    full_labels = dataset['labels'][:]
    labels_reg, keeper_idxs = get_shape_color_labels(full_labels)

    if keeper_idxs is not None:
        # img_data = np.array([img_data[idx] for idx in keeper_idxs])
        img_data = img_data[keeper_idxs]

    # Reshape (96000, 64, 64, 3) to (96000, 3, 64, 64) for torch conv layers
    img_data = img_data.transpose((0, 3, 1, 2))

    # Preprocess data
    # if normalize:
    #     img_data = img_data.astype("float32") / 255.0
    # if subtract_mean:
    #     tmp_data = img_data.reshape(img_data.shape[1], -1)
    #     mean = np.mean(tmp_data, axis=1)
    #     img_data = img_data - mean
    if normalize:
        img_data = img_data.astype("float32") / 255.0

    if subtract_mean:
        # Compute the mean per channel across all images and pixels
        mean = np.mean(img_data, axis=(0, 2, 3), keepdims=True)
        img_data = img_data - mean

    return img_data, labels_reg, full_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    full_data, labels_reg, _ = load_data(normalize=False, subtract_mean=False)
    print(f"Shape of full_data: {full_data.shape}")
    idx = np.random.randint(0, len(full_data))
    print(f"Label for img at {idx}: {np.argmax(labels_reg[idx])}")
    plt.imshow(full_data[idx].transpose(1, 2, 0).astype(np.uint8))
    plt.show()

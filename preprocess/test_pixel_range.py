from extract_patches import *


def test_range(pixel_url, mask_url):
    masks = load_hdf5(pixel_url)
    nodes = load_node(mask_url)
    output = np.zeros(masks.shape)

    print(np.shape(masks))
    print(np.shape(nodes))

    patch_h = 48
    patch_w = 48

    for i in range(masks.shape[0]):
        mask = masks[i][0]
        node = nodes[i]

        for j in range(np.shape(node)[0]):
            x_center = node[j][0]
            y_center = node[j][1]

            output[i][0][x_center][y_center] += 1

            # output[i:(i + 1), :, (y_center - int(patch_h / 2)):(y_center + int(patch_h / 2)),
            # (x_center - int(patch_w / 2)):(x_center + int(patch_w / 2))] += 1

    print(output.max(), output.min(), output.sum())
    output = output - 1
    output[output <= 0] = 0
    output[output >= 1] = 1
    print(output.sum())
    for i in range(output.shape[0]):
        img = Image.fromarray(output[i][0])
        img.show()


if __name__ == '__main__':
    dataset = "DRIVE"
    pixel_url = "./temp/CHASEDB1_dataset_borderMasks_train.hdf5".replace("CHASEDB1", dataset)
    mask_url = "./temp/CHASEDB1_train_patch.pickle".replace("CHASEDB1", dataset)
    test_range(pixel_url, mask_url)

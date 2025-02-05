import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def load_and_preprocess_data(data_path, batch_size):
    """
    data shape: [40000, 1, 64, 64]
    """
    dataset = torch.load(data_path)

    # random image in 8x8 grid 
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            idx = torch.randint(dataset.shape[0], (1,)).item()
            img = dataset[idx][0].numpy()  # numpy array, shape: (64,64)
            ax[i, j].imshow(img, origin="lower", cmap="gray")
            ax[i, j].axis("off")
    plt.suptitle("Random Samples from Dataset", fontsize=16)
    plt.show()

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, data_loader
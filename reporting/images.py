import matplotlib.pyplot as plt


def display_tensor(images: list, labels=[], batched=True, cols: int = 4, rows: int = 4, plot_scale: float = .3):
    fig = plt.figure(figsize=(plot_scale * cols, plot_scale * rows))
    fig.subplots_adjust(hspace=1 / plot_scale, wspace=1 / plot_scale)

    for i, ex in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = ex[0]
        if batched:
            image = image[0]
        ax.imshow(image)
        ax.grid(False)
        if len(labels) > i:
            plt.xlabel(labels[i])
        plt.xticks([], [])
        plt.yticks([], [])


def display_dataframe(images: list, labels=[], cols: int = 4, plot_scale: float = 2):
    elements = len(images)
    rows = elements / cols + 1
    fig = plt.figure(figsize=(plot_scale * cols, plot_scale * rows))
    fig.subplots_adjust(hspace=1 / plot_scale, wspace=1 / plot_scale)

    for i, ex in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        image = ex.reshape(28, 28)
        ax.imshow(image)
        ax.grid(False)
        if len(labels) > i:
            plt.xlabel(labels[i])
        plt.xticks([], [])
        plt.yticks([], [])

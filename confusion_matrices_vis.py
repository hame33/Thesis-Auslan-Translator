import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

exp01_matrix = np.array([
    [4,0,1,0,3,1,0,1,1,1],
    [0,3,0,0,2,4,0,1,0,2],
    [1,0,4,0,1,0,0,0,0,1],
    [0,0,1,5,2,0,2,0,1,0],
    [0,0,0,0,4,0,0,1,2,5],
    [2,1,0,0,1,9,0,3,0,1],
    [1,1,0,3,3,0,9,0,0,0],
    [0,0,0,0,0,1,0,8,1,0],
    [0,0,0,0,0,0,0,1,3,2],
    [0,3,0,0,3,1,1,1,0,6]
])
exp01_labels = ['GO','GOOD','HELLO','KNOW','ONE','READY','THINK','TIME','WHAT','YES']

nondet_v3_matrix = np.array([
    [6,0,0,2,0,0,2,2,0,2,0],
    [0,8,0,0,4,0,2,2,0,0,4],
    [0,0,8,0,0,2,0,0,0,0,2],
    [0,0,2,8,0,2,0,8,0,0,0],
    [0,0,0,0,6,4,0,4,0,0,8],
    [4,2,0,0,2,14,0,2,4,0,2],
    [0,2,0,6,2,0,16,0,0,0,0],
    [0,0,0,0,0,0,2,18,0,0,0],
    [0,0,0,2,0,0,0,0,8,0,2],
    [0,2,0,0,4,2,0,0,0,12,2],
    [2,4,0,0,4,2,2,0,2,0,16],
])
nondet_labels = ['GO','GOOD','HELLO','KNOW','ONE','READY','THINK','TIME','WHAT','YES','NON_DET']

def plot_confusion_matrix(ax, matrix, labels, title, normalise=True):
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display = matrix.astype(float) / row_sums
    else:
        display = matrix.astype(float)

    cmap = plt.cm.Blues
    im = ax.imshow(display, interpolation='nearest', cmap=cmap,
                   vmin=0, vmax=1 if normalise else matrix.max())

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    thresh = display.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = display[i, j]
            raw = matrix[i, j]
            if normalise:
                text = f'{val:.2f}\n({int(raw)})' if raw > 0 else ''
            else:
                text = str(int(raw)) if raw > 0 else ''
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=7)

    # Bold diagonal
    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor='black', lw=1.5
        ))

    return im

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plt.rcParams['font.family'] = 'serif'

im1 = plot_confusion_matrix(
    axes[0], exp01_matrix, exp01_labels,
    'exp01 — Clean clips only\n(101 test clips, 10 classes)',
    normalise=True
)

im2 = plot_confusion_matrix(
    axes[1], nondet_v3_matrix, nondet_labels,
    'NON\_DETECTION v3\n(235 test clips, 11 classes)',
    normalise=True
)

fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Proportion of true class')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Proportion of true class')

plt.suptitle('Confusion matrices: row-normalised (cell annotations show proportion and raw count)',
             fontsize=10, y=1.01)
plt.tight_layout()

os.makedirs('fig/assets', exist_ok=True)
plt.savefig('fig/assets/confusion_matrices.png', dpi=200, bbox_inches='tight')
plt.savefig('fig/assets/confusion_matrices.pdf', bbox_inches='tight')
print("Saved to fig/assets/")
plt.show()
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import umap
import numpy as np

def log_to_tensorboard(dataloader, model, epoch, name, writer):
    log_tsne_to_tensorboard(dataloader, model, epoch, name, writer)
    log_umap_to_tensorboard(dataloader, model, epoch, name, writer)
    calculate_silhouette_score(dataloader, model, epoch, name, writer)
    calculate_davies_bouldin_index(dataloader, model, epoch, name, writer)

# Function to plot the UMAP results and log to TensorBoard
def log_umap_to_tensorboard(data_loader, model, epoch, name, writer):
    reducer = umap.UMAP(n_components=2)
    X = []
    Y = []
    for data, labels in data_loader:
        target = model.inference(data)
        for i in range(len(labels)):
            X.append(target[i].detach().cpu().numpy())
            Y.append(labels[i].detach().cpu().numpy())

    X = np.array(X)
    X = reducer.fit_transform(X)

    # Plot the UMAP results
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='tab20')

    # Create a legend
    unique_labels = np.unique(Y)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cm.tab20(i / len(unique_labels)), markersize=10) for i in range(len(unique_labels))]
    ax.legend(handles, unique_labels, title="Labels")

    # Log the plot to TensorBoard
    writer.add_figure(f'{name}_umap/epoch_{epoch}', fig)
    plt.close(fig)

# Function to plot the t-SNE results and log to TensorBoard
def log_tsne_to_tensorboard(data_loader, model, epoch, name, writer):
    tsne = TSNE(n_components=2, perplexity=40, max_iter=300)
    X = []
    Y = []
    for data, labels in data_loader:
        target = model.inference(data)
        for i in range(len(labels)):
            X.append(target[i].detach().cpu().numpy())
            Y.append(labels[i].detach().cpu().numpy())

    X = np.array(X)
    X = tsne.fit_transform(X)

    # Plot the t-SNE results
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='tab20')

    # Create a legend
    unique_labels = np.unique(Y)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cm.tab20(i / len(unique_labels)), markersize=10) for i in range(len(unique_labels))]
    ax.legend(handles, unique_labels, title="Labels")

    # Log the plot to TensorBoard
    writer.add_figure(f'{name}_tsne/epoch_{epoch}', fig)
    plt.close(fig)

# Function to calculate silhouette score and log to TensorBoard
def calculate_silhouette_score(data_loader, model, epoch=None, name=None, writer=None):
    X = []
    Y = []
    for data, labels in data_loader:
        target = model.inference(data)
        for i in range(len(labels)):
            X.append(target[i].detach().cpu().numpy())
            Y.append(labels[i].detach().cpu().numpy())

    X = np.array(X)
    Y = np.array(Y)
    score = silhouette_score(X, Y)
    
    if epoch is None or name is None or writer is None:
        return score
    
    # Log the silhouette score to TensorBoard
    writer.add_scalar(f'{name}_silhouette_score', score, epoch)
    return score

# Function to calculate Davies-Bouldin index and log to TensorBoard
def calculate_davies_bouldin_index(data_loader, model, epoch=None, name=None, writer=None):
    X = []
    Y = []
    for data, labels in data_loader:
        target = model.inference(data)
        for i in range(len(labels)):
            X.append(target[i].detach().cpu().numpy())
            Y.append(labels[i].detach().cpu().numpy())

    X = np.array(X)
    Y = np.array(Y)
    score = davies_bouldin_score(X, Y)
    
    if epoch is None or name is None or writer is None:
        return score

    # Log the Davies-Bouldin index to TensorBoard
    writer.add_scalar(f'{name}_davies_bouldin_index', score, epoch)
    return score
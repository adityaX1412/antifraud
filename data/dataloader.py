import torch
import random
import numpy as np
import dgl
from dgl.data.fraud import FraudAmazonDataset, FraudYelpDataset
from dgl.dataloading import NeighborSampler, DataLoader 
from dgl import RowFeatNormalizer
from sklearn.model_selection import train_test_split

def load_data(data_name, seed, train_ratio, test_ratio, n_layer, batch_size):
    # Load dataset
    if data_name == 'yelp':
        graph = FraudYelpDataset().graph
        node = 'review'
        idx_unlabeled = False
    else:
        graph = FraudAmazonDataset().graph
        node = 'user'
        idx_unlabeled = 3305
        transform = RowFeatNormalizer(subtract_min=True, node_feat_names=['feature'])
        graph = transform(graph)

    # Ensure graph format remains on CPU
    graph = graph.formats("coo")  # No .to("cuda")

    # Ensure feature tensors are contiguous and remain on CPU
    features = graph.ndata["feature"].contiguous()
    labels = graph.ndata["label"].contiguous()

    # Convert labels to NumPy for sklearn train_test_split()
    labels_np = labels.numpy()

    # Data split
    index = list(range(len(labels_np)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index[idx_unlabeled:], labels_np[idx_unlabeled:], 
                                                            stratify=labels_np[idx_unlabeled:], 
                                                            train_size=train_ratio, random_state=seed, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, 
                                                            test_size=test_ratio, random_state=seed, shuffle=True)

    # Convert labels back to PyTorch tensors (CPU-only)
    graph.ndata["y"] = torch.tensor(labels_np, dtype=torch.long)  # No .to(graph.device)

    # Masking nodes
    y_mask = graph.ndata["y"].clone()
    y_mask[index[:idx_unlabeled] + idx_test + idx_valid] = 2  # Masking unknown/test nodes
    graph.ndata["y_mask"] = y_mask.contiguous()
    graph.ndata["x"] = features  # Features are already contiguous

    # Batch loader parameters
    n_sample = {e: 50 for e in graph.etypes}
    n_samples = [n_sample] * n_layer

    # Edge probabilities (for sampling)
    edge_probs = {}
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype)
        prob = torch.where(y_mask[src] == 2, 0.5, 0.9).contiguous()
        edge_probs[etype] = prob
        graph.edges[etype].data['prob'] = prob

    # Define DGL neighbor sampler (CPU-only)
    sampler = NeighborSampler(n_samples, prob='prob')
    train_loader = DataLoader(graph, idx_train, sampler, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(graph, idx_valid, sampler, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(graph, idx_test, sampler, batch_size=batch_size, shuffle=False, drop_last=False)

    return features.shape[1], train_loader, valid_loader, test_loader

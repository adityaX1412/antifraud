import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from scipy.stats import zscore
from methods.stagn.stagn_2d import stagn_2d_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from feature_engineering.data_engineering import span_data_2d
from scipy.io import loadmat
import pickle

def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


def stagn_train_2d(
    features,
    labels,
    train_idx,
    test_idx,
    g,
    num_classes: int = 2,
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    g = g.to(device)
    model = stagn_2d_model(
        time_windows_dim=features.shape[2],
        feat_dim=features.shape[1],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
        g=g,
        device=device
    )
    model.to(device)

    features = torch.from_numpy(features).to(device)
    features.transpose_(1, 2)
    labels = torch.from_numpy(labels).to(device)

    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts)*len(labels)/len(unique_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(features, g)
        loss = loss_func(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        pred = to_pred(out[train_idx])
        true = labels[train_idx].cpu().numpy()
        pred = np.array(pred)
        print(f"Epoch: {epoch}, loss: {loss:.4f}, auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")

    with torch.no_grad():
        out = model(features, g)
        pred = to_pred(out[test_idx])
        true = labels[test_idx].cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")


def stagn_main(
    features,
    labels,
    test_ratio,
    g,
    mode: str = "2d",
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 0.003,
    device="cpu",
):
    train_idx, test_idx = train_test_split(
        np.arange(features.shape[0]), test_size=test_ratio, stratify=labels)

    # y_pred = np.zeros(shape=test_label.shape)
    if mode == "2d":
        stagn_train_2d(
            features,
            labels,
            train_idx,
            test_idx,
            g,
            epochs=epochs,
            attention_hidden_dim=attention_hidden_dim,
            lr=lr,
            device=device
        )
    else:
        raise NotImplementedError("Not supported mode.")


def load_stagn_data(dataset: str, test_size: float, args: dict):
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    
    if dataset == "S-FFSD":
        # load S-FFSD dataset for base models
        data_path = os.path.join(prefix, "S-FFSD.csv")
        feat_df = pd.read_csv(data_path)
        
        # ICONIP16 & AAAI20 requires higher dimensional data
        features_path = os.path.join(prefix, "features.npy")
        labels_path = os.path.join(prefix, "labels.npy")
        
        if os.path.exists(features_path) and os.path.exists(labels_path):
            features = np.load(features_path)
            labels = np.load(labels_path)
        else:
            features, labels = span_data_2d(feat_df)
            np.save(features_path, features)
            np.save(labels_path, labels)
    
        # Filter out class 2 (if it exists)
        sampled_df = feat_df[feat_df['Labels'] != 2]
        sampled_df = sampled_df.reset_index(drop=True)
    
        # Create node encoding
        all_nodes = pd.concat([sampled_df['Source'], sampled_df['Target']]).unique()
        encoder = LabelEncoder().fit(all_nodes)  
        encoded_source = encoder.transform(sampled_df['Source'])
        encoded_tgt = encoder.transform(sampled_df['Target'])  
    
        # Create location features
        loc_enc = OneHotEncoder()
        loc_feature = np.array(loc_enc.fit_transform(
            sampled_df['Location'].to_numpy()[:, np.newaxis]).todense())
        loc_feature = np.hstack(
            [zscore(sampled_df['Amount'].to_numpy())[:, np.newaxis], loc_feature])
    
        # Create DGL graph
        g = dgl.DGLGraph()
        g.add_edges(encoded_source, encoded_tgt, data={
                    "feat": torch.from_numpy(loc_feature).to(torch.float32)})
        
    elif dataset == "yelp":
        # Load Yelp dataset
        data_file = loadmat(os.path.join(prefix, 'YelpChi.mat'))
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        
        # Convert 2D features to 3D format expected by stagn_train_2d
        # Add a time dimension (assuming single time window)
        features_2d = feat_data.to_numpy()
        features = np.expand_dims(features_2d, axis=2)  # Shape: (samples, features, 1)
        labels = labels.to_numpy()
        
        # Load the preprocessed adj_lists
        with open(os.path.join(prefix, 'yelp_homo_adjlists.pickle'), 'rb') as file:
            homo = pickle.load(file)
        
        # Create edges from adjacency lists
        src = []
        tgt = []
        for i in homo:
            for j in homo[i]:
                src.append(i)
                tgt.append(j)
        
        src = np.array(src)
        tgt = np.array(tgt)
        
        # Create DGL graph
        g = dgl.graph((src, tgt))
        g.ndata['label'] = torch.from_numpy(labels).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
        
        # Save graph
        graph_path = os.path.join(prefix, "graph-{}.bin".format(dataset))
        dgl.data.utils.save_graphs(graph_path, [g])

    elif dataset == "amazon":
        # Load Amazon dataset
        data_file = loadmat(os.path.join(prefix, 'Amazon.mat'))
        labels = pd.DataFrame(data_file['label'].flatten())[0]
        feat_data = pd.DataFrame(data_file['features'].todense().A)
        
        # For compatibility with S-FFSD format, create features array
        # Note: Using only the subset starting from index 3305 as in original code
        features_2d = feat_data.iloc[3305:].to_numpy()
        features = np.expand_dims(features_2d, axis=2)  # Shape: (samples, features, 1)
        subset_labels = labels.iloc[3305:].to_numpy()
        
        # Load the preprocessed adj_lists
        with open(os.path.join(prefix, 'amz_homo_adjlists.pickle'), 'rb') as file:
            homo = pickle.load(file)
        
        # Filter edges to only include nodes that have labels (indices >= 3305)
        valid_nodes = set(range(3305, len(labels)))
        src = []
        tgt = []
        for i in homo:
            if i in valid_nodes:
                for j in homo[i]:
                    if j in valid_nodes:
                        # Remap node indices to start from 0
                        src.append(i - 3305)
                        tgt.append(j - 3305)
        
        src = np.array(src)
        tgt = np.array(tgt)
        
        # Create DGL graph with remapped node indices
        if len(src) > 0:  # Only create graph if there are edges
            g = dgl.graph((src, tgt))
        else:
            # Create empty graph with correct number of nodes
            g = dgl.graph(([], []))
            g.add_nodes(len(subset_labels))
        
        g.ndata['label'] = torch.from_numpy(subset_labels).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.iloc[3305:].to_numpy()).to(torch.float32)
        
        # Update labels for return value
        labels = subset_labels
        
        # Save graph
        graph_path = os.path.join(prefix, "graph-{}.bin".format(dataset))
        dgl.data.utils.save_graphs(graph_path, [g])
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return features, labels, g

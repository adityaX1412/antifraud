import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import dgl
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import argparse
import time
from tqdm import tqdm

from methods.stagn.stagn_2d import stagn_2d_model

def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data(dataset_name, data_dir):
    """Load graph data and neighbor features"""
    print(f"Loading {dataset_name} dataset...")
    
    # Load DGL graph
    graph_path = "/kaggle/input/amazon-graph/graph-amazon.bin"
    graphs, _ = dgl.load_graphs(graph_path)
    graph = graphs[0]
    
    # Load neighbor features
    neighbor_feat_path = "/kaggle/input/amazon-neigh/amazon_neigh_feat.csv"
    neighbor_feats = pd.read_csv(neighbor_feat_path)
    
    print(f"Graph info: {graph}")
    print(f"Neighbor features shape: {neighbor_feats.shape}")
    
    return graph, neighbor_feats

def prepare_features(graph, neighbor_feats, device, time_windows_dim=8):
    """Prepare features for STAGN model"""
    
    # Get node features and labels
    node_features = graph.ndata['feat'].cpu().numpy()
    labels = graph.ndata['label'].cpu().numpy()
    neighbor_features = neighbor_feats.values
    
    # Combine original node features with neighbor features
    combined_features = np.concatenate([node_features, neighbor_features], axis=1)
    
    # Reshape features for time windows (simulate temporal data)
    # For simplicity, we'll duplicate features across time windows
    # In practice, you might have actual temporal features
    feat_dim = combined_features.shape[1]
    n_samples = combined_features.shape[0]
    
    # Create time-series like data by adding noise to simulate temporal variations
    temporal_features = np.zeros((n_samples, time_windows_dim, feat_dim))
    for t in range(time_windows_dim):
        noise_factor = 0.01 * t  # Small temporal variation
        temporal_features[:, t, :] = combined_features + np.random.normal(0, noise_factor, combined_features.shape)
    
    temporal_features = torch.tensor(temporal_features, device=device)
    labels = torch.tensor(labels, device=device)
    return temporal_features, labels, feat_dim

def create_edge_features(graph, device):
    """Create edge features for the graph"""
    # Simple edge features based on node degrees and labels
    src_nodes, dst_nodes = graph.edges()
    
    # Get node degrees
    in_degrees = graph.in_degrees().float()
    out_degrees = graph.out_degrees().float()
    
    # Create edge features as combination of source and destination node properties
    src_in_deg = in_degrees[src_nodes]
    src_out_deg = out_degrees[src_nodes]
    dst_in_deg = in_degrees[dst_nodes]
    dst_out_deg = out_degrees[dst_nodes]
    
    # Stack edge features
    edge_features = torch.stack([src_in_deg, src_out_deg, dst_in_deg, dst_out_deg], dim=1)
    
    return edge_features

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (features, labels, graph) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        graph = graph.to(device)
        
        optimizer.zero_grad()
        outputs = model(features, graph)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels, graph in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            graph = graph.to(device)
            
            outputs = model(features, graph)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc

class GraphDataLoader:
    """Custom data loader for graph data"""
    
    def __init__(self, features, labels, graph, batch_size, shuffle=True):
        self.features = features
        self.labels = labels
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(features)
        
    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_features = torch.FloatTensor(self.features[batch_indices])
            batch_labels = torch.LongTensor(self.labels[batch_indices])
            
            # Create subgraph for this batch
            subgraph = dgl.node_subgraph(self.graph, batch_indices)
            subgraph = subgraph.to(self.graph.device)
            
            yield batch_features, batch_labels, subgraph
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

def main():
    parser = argparse.ArgumentParser(description='Train STAGN model on fraud detection datasets')
    parser.add_argument('--dataset', type=str, choices=['yelp', 'amazon'], required=True,
                       help='Dataset to use (yelp or amazon)')
    parser.add_argument('--data_dir', type=str, default='../data/',
                       help='Directory containing the data files')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for attention')
    parser.add_argument('--time_windows', type=int, default=8,
                       help='Number of time windows')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Get absolute path to data directory
    data_dir = os.path.abspath(args.data_dir)
    
    # Load data
    graph, neighbor_feats = load_data(args.dataset, data_dir)
    
    # IMPORTANT: Move graph to device BEFORE creating edge features or initializing model
    print(f"Moving graph to device: {device}")
    graph = graph.to(device)
    
    # Prepare features
    temporal_features, labels, feat_dim = prepare_features(
        graph, neighbor_feats, device, args.time_windows
    )
    
    # Create edge features (now that graph is on the correct device)
    edge_features = create_edge_features(graph, device)
    graph.edata['feat'] = edge_features
    
    labels = labels.cpu()
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"Feature dimension: {feat_dim}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Split data
    indices = np.arange(len(temporal_features))
    train_indices, test_indices = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed, 
        stratify=labels
    )
    
    # Create data loaders
    train_loader = GraphDataLoader(
        temporal_features[train_indices], 
        labels[train_indices], 
        graph, 
        args.batch_size, 
        shuffle=True
    )
    
    test_loader = GraphDataLoader(
        temporal_features[test_indices], 
        labels[test_indices], 
        graph, 
        args.batch_size, 
        shuffle=False
    )
    
    # Initialize model (graph is already on correct device)
    print("Initializing STAGN model...")
    num_classes = len(np.unique(labels))
    model = stagn_2d_model(
        time_windows_dim=args.time_windows,
        feat_dim=feat_dim,
        num_classes=num_classes,
        attention_hidden_dim=args.hidden_dim,
        g=graph,  # Graph is already on correct device
        device=device
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Dummy forward pass to initialize LazyModules
    with torch.no_grad():
        dummy_features = torch.zeros((1, args.time_windows, feat_dim), device=device)
        dummy_graph = dgl.node_subgraph(graph, [0])  # Tiny dummy graph with 1 node
        dummy_graph = dummy_graph.to(device)

        model(dummy_features, dummy_graph)  # Forward pass

    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Test F1: {test_f1:.4f} | Test AUC: {test_auc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
    
    # Load best model and final evaluation
    model.load_state_dict(best_model_state)
    final_loss, final_acc, final_prec, final_rec, final_f1, final_auc = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print(f"Final Results on {args.dataset.upper()} Dataset:")
    print(f"{'='*60}")
    print(f"Test Accuracy:  {final_acc:.4f}")
    print(f"Test Precision: {final_prec:.4f}")
    print(f"Test Recall:    {final_rec:.4f}")
    print(f"Test F1-Score:  {final_f1:.4f}")
    print(f"Test AUC:       {final_auc:.4f}")
    print(f"{'='*60}")
    
    # Save model
    model_save_path = f"stagn_{args.dataset}_best_model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'args': args,
        'final_metrics': {
            'accuracy': final_acc,
            'precision': final_prec,
            'recall': final_rec,
            'f1': final_f1,
            'auc': final_auc
        }
    }, model_save_path)
    
    print(f"Best model saved to: {model_save_path}")

if __name__ == '__main__':
    main()

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, SchNet, radius_graph

# Step 1: Load data from CSV and create a PyG-compatible dataset
def load_labels_csv(csv_file):
    labels_df = pd.read_csv(csv_file)
    protein_ids = labels_df["protein_id"].tolist()
    solubility_values = labels_df["solubility"].tolist()
    return protein_ids, solubility_values

def extract_calpha_and_amino_acids(pdb_file_path):
    from Bio.PDB import PDBParser

    def get_residue_amino_acid_type(residue):
        try:
            return residue.get_resname()
        except:
            return None

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file_path)
    calpha_positions = {}
    amino_acid_types = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ' and residue.has_id('CA'):
                    residue_id = residue.get_id()
                    chain_id = chain.get_id()
                    calpha_positions[(chain_id, residue_id)] = residue['CA'].get_coord()
                    amino_acid_types[(chain_id, residue_id)] = get_residue_amino_acid_type(residue)

    return calpha_positions, amino_acid_types

def calculate_calpha_distances(calpha_positions):
    positions = torch.tensor(list(calpha_positions.values()), dtype=torch.float)
    distances = torch.cdist(positions, positions)  # Calculate pairwise Euclidean distances
    return distances

def calculate_edge_index(distances, r):
    # Calculate edge_index based on the distance threshold r
    edge_index = []
    num_nodes = distances.size(0)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distances[i, j] <= r:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def create_pyg_dataset(pdb_folder, protein_ids, solubility_values):
    dataset = []
    for protein_id, solubility in zip(protein_ids, solubility_values):
        pdb_file_path = os.path.join(pdb_folder, f"{protein_id}.pdb")
        calpha_positions, amino_acid_types = extract_calpha_and_amino_acids(pdb_file_path)
        distances = calculate_calpha_distances(calpha_positions)

        # Convert amino acid types to one-hot encoding
        amino_acids = [amino_acid_types[calpha_id] for calpha_id in calpha_positions.keys()]
        amino_acids_set = sorted(set(amino_acids))
        amino_acid_dict = {amino_acid: torch.tensor(i, dtype=torch.float) for i, amino_acid in enumerate(amino_acids_set)}
        amino_acids_one_hot = torch.stack([amino_acid_dict[amino_acid] for amino_acid in amino_acids], dim=0)
        positions = torch.tensor(list(calpha_positions.values()), dtype=torch.float)

        edge_index = calculate_edge_index(distances, r=20.0)

        data = Data(x=amino_acids_one_hot, edge_index=edge_index, pos=positions, solubility=torch.tensor([solubility], dtype=torch.float))
        dataset.append(data)
    return dataset


# Step 2: Prepare data loader for training
def get_data_loaders(dataset, batch_size=32, num_workers=4, split={"train": 0.8, "val": 0.2}):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split_idx = int(split["train"] * num_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader([dataset[i] for i in val_idx], batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader

#step 3: Schnet Model stuff
class CustomSchNetLayer(MessagePassing):
    def __init__(self, hidden_channels):
        super(CustomSchNetLayer, self).__init__(aggr="mean")  # Use mean aggregation for message passing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, pos):
        edge_attr = torch.cdist(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1)  # Calculate pairwise distances
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.mlp(x_j)

class SchNetModel(nn.Module):
    def __init__(self, num_amino_acids, hidden_channels):
        super(SchNetModel, self).__init__()
        self.conv = CustomSchNetLayer(hidden_channels)
        self.fc = nn.Linear(num_amino_acids * hidden_channels, 1)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos  # Use data.pos for positional information
        x = self.conv(x, edge_index, pos)  # Pass pos to the CustomSchNetLayer
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# Step 4: Train the ImprovedSchNet model
def train_schnet_model(train_loader, val_loader, model, num_epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.solubility)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.solubility)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Step 5: Main function
def main():
    pdb_folder = "data"
    csv_file = "labels.csv"

    protein_ids, solubility_values = load_labels_csv(csv_file)
    dataset = create_pyg_dataset(pdb_folder, protein_ids, solubility_values)

    amino_acids_set = set()
    for data in dataset:
        amino_acids_indices = torch.nonzero(data.x, as_tuple=False)[:, 0]
        amino_acids_set.update(amino_acids_indices.tolist())

    num_amino_acids = len(amino_acids_set)
    hidden_channels = 64

    train_loader, val_loader = get_data_loaders(dataset)
    model = SchNetModel(num_amino_acids, hidden_channels) 
    train_schnet_model(train_loader, val_loader, model)

if __name__ == "__main__":
    main()

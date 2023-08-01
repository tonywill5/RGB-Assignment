# Main coding file breakdown by function

### def load_labels_csv(csv_file):
This function takes .csv file as input and extracts the protein_id's in one column & the associated value in the other. It then returns the two as lists.

### def extract_calpha_and_amino_acids(pdb_file_path):
This function leverages BioPython's PDB file parser and extracts the calpha positions as well as the respective resiudue. It returns the calpha positions and amino acid types as dictionaries for their respective pdb file.

### def calculate_calpha_distances(calpha_positions):
This function calculates the eudclidean norm distance between calpha positions.

### def calculate_edge_index(distances, r):
This function creates edge index values if calpha positions are within some threshold value (20A).

### def create_pyg_dataset(pdb_folder, protein_ids, solubility_values):
This function creates 

### def get_data_loaders(dataset, batch_size=32, num_workers=4, split={"train": 0.8, "val": 0.2}):

### class CustomSchNetLayer

### class CustomSchNetLayer(MessagePassing):
* def __init__(self, hidden_channels):
* def forward(self, x, edge_index, pos):
* def message(self, x_j, edge_attr):

### class SchNetModel(nn.Module):
* def __init__(self, num_amino_acids, hidden_channels):
* def forward(self, data):

### def train_schnet_model(train_loader, val_loader, model, num_epochs=50, lr=0.001):

### def main():

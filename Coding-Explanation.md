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
This function creates individual data points for each pdb file which contains a one hot encoding of the amino acid type, an edge index value for the calphas that are within 20A, the coords of the calphas, and the values from the labels.txt file.

### def get_data_loaders(dataset, batch_size=32, num_workers=4, split={"train": 0.8, "val": 0.2}):
Split the dataset into training and validation sets (0.8 and 0.2 respectively). This function will then make then DataLoader types and return them.

### class CustomSchNetLayer(MessagePassing):
This function creates a custom SchNet layer from the MessagePassing class in pytorch geometric. The layer implements a graph message-passing operation using the SchNet interaction potential. It takes a single argument hidden_channels representing the number of hidden channels in the layer. The forward method performs the message-passing operation and uses a multi-layer perceptron (MLP) for message transformation.

### class SchNetModel(nn.Module):
This class defines the overall SchNet model. It takes the number of unique amino acid types (num_amino_acids) and the number of hidden channels (hidden_channels) as input. The model consists of a custom SchNet layer followed by a fully connected layer for prediction.

### def train_schnet_model(train_loader, val_loader, model, num_epochs=50, lr=0.001):
This function trains the model and tests it on the validation set. It prints the epoch number, training loss and validation.

### def main():
Calls on necessary functions to load the data, create the SchNet model, train and run the model.

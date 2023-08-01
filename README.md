# RGB-Assignment
Goal: Create a simple SchNet model to predict protein properties \

Process: \
1.) Download all necessary packages which included: pytorch, pytorch geometric, BioPython, vscode, and the data \
2.) Load the necessary data from the pdb files \
  a.) Create a .csv from the labels.txt file (easier to manage for myself) - the values associated with the labels represent the predicted output values \
    b.) parse the pdb files and extract the calpha locations along with their respective residues \
 3.) Create a function for joining all relevant data \
   a.)data will contain values for one hot encoding of the amino acid type, edge index based on being within 20A of another calpha, calpha location explicitly (x,y,z coords), and the projected output for the associated pdb file \
 4.) 

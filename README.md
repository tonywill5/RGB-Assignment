# RGB-Assignment
Goal: Create a simple SchNet model to predict protein properties 

Process: \
&nbsp; &nbsp; &nbsp; &nbsp;1.) Download all necessary packages which included: pytorch, pytorch geometric, BioPython, vscode, and the data 

&nbsp; &nbsp; &nbsp; &nbsp;2.) Load the necessary data from the pdb files \
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;a.) Create a .csv from the labels.txt file (easier to manage for myself) - the values associated with the labels represent the predicted output values \
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;b.) parse the pdb files and extract the calpha locations along with their respective residues 

 &nbsp; &nbsp; &nbsp; &nbsp;3.) Create a function for joining all relevant data \
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;a.)data will contain values for one hot encoding of the amino acid type, edge index based on being within 20A of another calpha, calpha location explicitly (x,y,z coords), and the projected output for the associated pdb file 
 
 &nbsp; &nbsp; &nbsp; &nbsp;4.) Prepare the data for training module via DataLoader from torch geometric 
 
 &nbsp; &nbsp; &nbsp; &nbsp;5.) Create the SchNet model architecture 
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;a.) create a class for the SchNet model layer \
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;b.) create a class for the entire SchNet model leveraging the previously made layer \
 
 &nbsp; &nbsp; &nbsp; &nbsp;6.) Create a function for training the model 
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;a.)use Adam optimizer and MSE loss function for 50 epochs with a learing rate of 0.001 

 &nbsp; &nbsp; &nbsp; &nbsp;7.) In main function call to all necessary functions and classes 

Notes: \
* Unable to train the model as it was crashing my computter
* all code was written on VScode
* Data folder and labels file was stored locally on computer

Issues: \
asdfasdf

Possible changes: \
asdfasdf

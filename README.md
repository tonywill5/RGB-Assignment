# RGB-Assignment
Goal: Create a simple SchNet model to predict protein properties 
 <br />
 <br />
## Process: 
1.) Download all necessary packages which included: pytorch, pytorch geometric, BioPython, vscode, and the data 

2.) Load the necessary data from the pdb files \
&nbsp; &nbsp; &nbsp; &nbsp;a.) Create a .csv from the labels.txt file (easier to manage for myself) - the values associated with the labels represent the predicted output values \
&nbsp; &nbsp; &nbsp; &nbsp;b.) Parse the pdb files and extract the calpha locations along with their respective residues 

3.) Create a function for joining all relevant data \
&nbsp; &nbsp; &nbsp; &nbsp;a.) Data will contain values for one hot encoding of the amino acid type, edge index based on being within 20A of another calpha, calpha location explicitly (x,y,z coords), and the projected output for the associated pdb file 
 
4.) Prepare the data for training module via DataLoader from torch geometric 
 
5.) Create the SchNet model architecture \
&nbsp; &nbsp; &nbsp; &nbsp;a.) Create a class for the SchNet model layer \
&nbsp; &nbsp; &nbsp; &nbsp;b.) Create a class for the entire SchNet model leveraging the previously made layer \
 
6.) Create a function for training the model \
&nbsp; &nbsp; &nbsp; &nbsp;a.) Use Adam optimizer and MSE loss function for 50 epochs with a learning rate of 0.001 

7.) In main function call to all necessary functions and classes 
 <br />
 <br />
## Notes: 
* All code was written on VScode
* Data folder and labels file were stored locally on computer
* Coding-Explanation includes written descriptions of each function

 <br />
 
## Issues: 
* Unable to train the model as it was crashing my computter

 <br />
 
## Possible changes: 
* Different loss function
* Different optimizing function
* Adjust number of epochs
* Different number of hidden channels
* More/less layers

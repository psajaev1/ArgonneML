READ ME 

(COMMANDS ARE IN LINUX ALREADY)
You will first have to install Python for this project. I have Python 3.6.8 installed
After this you can create a virtualenv, to install virtualenv run this:
	python3 -m pip install --user virtualenv

You must then create a virtual environment, use this command (env is name of your venv): 
	python3 -m venv env
To activate this enviornment, go to scripts folder within the env and then type in activate 

You must then install dependencies for this project using the requirements.txt 
	pip3 install -r requirements.txt

Make sure that your file structure is the same as in the folder I gave you or else you will have to use full path
After this you are good to go! 

Using the model: 
	You will only need to edit the finalized.py file 
	Within this file, the only thing you will have to edit is the path on line 21 
	finalized.py takes in one single graph input and returns whether it is BPM or CORR
	To run this file do: 
		py finalized.py 


Contents of folder: 
	bpmData
	corrData
	checkCounts.py 
		- Was used to check to see if equal counts of bpm and corr after deciding cutoff length 
	combineData.py 
		- Combined the individual files into one large file called fullData.csv 
	finalized.py 
		- Main file used for prediction of single sample 
	fullData.csv
		- File with all data, about 18k samples
	model_ff_net-0010.params
		- File with neural network 
	model_ff_net-symbol.json 
		- File with neural network
	requirements.txt 
		- List of independencies needed for this project 
	trainandTest.py 
		- Tested out several non Neural Network models for predictions 
	trainNet.py 
		- Created neural network used for predicting





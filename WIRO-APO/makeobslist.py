import argparse
import numpy as np
# NP Necessary imports

def names(wiro):
	'''Extracts objects' names from a list of objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-wiro: ndarray. A table containing objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-names: ndarray. An array of object names'''
	names = wiro[:,1]
	# NP Reading in names column
	return names

def ra(wiro):
	'''Extracts objects' ras from a list of objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-wiro: ndarray. A table containing objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-ra: ndarray. An array of object ras'''
	ra = wiro[:,2]
	# NP Reading in ra column
	return ra

def dec(wiro):
	'''Extracts objects' decs from a list of objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-wiro: ndarray. A table containing objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-dec: ndarray. An array of object decs'''
	dec = wiro[:,3]
	# NP Reading in dec column
	return dec

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Module to write\
		a list of targets in APO format from a list of targets\
		in WIRO format. APO format example can be found in\
		/d/zem1/hak/chip/observing/APO/ZetaOphStars.txt.\
		WIRO format example can be found in\
		/d/zem1/hak/chip/Nikhil/Nikhiltargets.cat.')
	parser.add_argument('path', type = str, help = 'Path to WIRO file.\
		Example: /d/...')
	# NP Defining parser path argument
	parser.add_argument('name', type = str, help = \
		'Name of generated file. Example: name')
	# NP Defining parser new file name argument
	args = parser.parse_args()
	# NP Adding parser to run at command line
	wirofile = args.path
	# NP Defining WIRO list
	wiro = np.loadtxt(wirofile , dtype = str, delimiter = ' ')
	# NP Reading in WIRO list
	n = names(wiro)
	# Extracting object names
	ras = ra(wiro)
	# NP Extracting object ras
	decs = dec(wiro)
	# NP Extracting objects decs
	APO = [n[i] +' ' +ras[i] +' ' +decs[i] +\
		' RotType=Horizon; RotAng=90' for i in range(len(n))]
	# NP Generating array of objects in APO format
	np.savetxt('/d/users/nikhil/observing/' +args.name +'.txt', APO,\
		fmt = '%s')
	# NP Writing APO objects to a file with new inputted file name
	print("Done.")
	# NP Printing message when conversion is complete

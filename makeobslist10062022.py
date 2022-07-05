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

def limitobjs(wiro, APO, ramax, ramin, decmax, decmin, glim):
	'''Extracts objects' decs from a list of objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-wiro: ndarray. A table containing objects in WIRO format.
	-APO: ndarray. A table containing objects in APO format
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-ii: ndarray. A boolean array corresponding to which objects
	satisfy the given constraints.'''
	G = np.array([wiro[:,8][i][2:] for i in range(len(wiro[:,8]))]\
		, dtype = float)
	# NP Defining g magnitudes from WIRO data
	decdeg = np.array([wiro[:,3][i][:3] for i in range(len(wiro[:,8]\
		))], dtype = float)
	# NP Defining declination in degrees from WIRO data
	rahour = np.array([wiro[:,2][i][:2] for i in range(len(wiro[:,8]\
		))], dtype = float)
	# NP Defining right ascension hour from WIRO data
	ii = (G > glim) & (decdeg > decmin) & (rahour < ramax) & (rahour >\
		ramin) & (decdeg < decmax)
	# NP Finding all objects that fit criteria
	print('Limited selection to ' +str(len(np.array(APO)[ii])) \
		+' objects from original '+str(len(APO)) +' object catalog.')
	return ii

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Module to write\
		a list of targets in APO format from a list of targets\
		in WIRO format. APO format example can be found in\
		/d/zem1/hak/chip/observing/APO/ZetaOphStars.txt.\
		WIRO format example can be found in\
		/d/zem1/hak/chip/Nikhil/Nikhiltargets.cat. Additionally,\
		limit selection by RA and Dec bounds and G magnitude.')
	parser.add_argument('filename', type = str, help = 'Name of WIRO file.\
		Example: blah.cat')
	# NP Defining parser path argument
	parser.add_argument('name', type = str, help = \
		'Name of generated file. Example: name')
	# NP Defining parser name argument
	parser.add_argument('RAupper', type = float, help = \
		'Upper limit of RA bound in hours. Example: 19')
	# NP Defining parser RA max argument
	parser.add_argument('RAlower', type = float, help = \
		'Lower limit of RA bound in hours. Example: 15')
	# NP Defining parser RA min argument
	parser.add_argument('Decupper', type = float, help = \
		'Upper limit of Dec bound in degrees. Example: 110')
	# NP Defining parser Dec max agrument
	parser.add_argument('Declower', type = float, help = \
		'Lower limit of Dec bound in degrees. Example: -45')
	# NP Defining parser Dec min argument
	parser.add_argument('gmag', type = float, help = \
		'G Magnitude ceiling. Selection will exclude objects'
			' brighter than this constraint. Example: 12')
	# NP Defining parser Dec min argument
	args = parser.parse_args()
	# NP Adding parser to run at command line
	wirofile = args.filename
	# NP Defining WIRO list
	wiro = np.loadtxt('/d/users/nikhil/observing/' +wirofile ,\
		dtype = str, delimiter = ' ')
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
	index = limitobjs(wiro, APO, args.RAupper, args.RAlower,\
		args.Decupper, args.Declower, args.gmag)
	dimmerAPO = np.array(APO)[index]
	np.savetxt('/d/users/nikhil/observing/' +args.name +'.txt', APO,\
		fmt = '%s')
	np.savetxt('/d/users/nikhil/observing/' +args.name +'dim.txt',\
		dimmerAPO, fmt = '%s')
	# NP Writing APO objects to a file with new inputted file name
	print("Done.")
	# NP Printing message when conversion is complete

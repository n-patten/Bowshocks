import argparse
import numpy as np
import astropy as ap
from astropy.time import Time
from astropy.coordinates import get_sun
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

def Gmags(wiro):
	'''Extracts objects' gmags from a list of objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-wiro: ndarray. A table containing objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-gmags: ndarray. An array of object gmags'''
	gmags = wiro[:,8]
	# NP Reading in dec column
	return gmags

def limitobjs(wiro, glim):
	'''Extracts objects' decs from a list of objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-wiro: ndarray. A table containing objects in WIRO format.
	-glim: int. The lower g-magnitude to filter by.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-ii: ndarray. A boolean array corresponding to which objects
	satisfy the given constraints.'''
	sunra = get_sun(Time.now()).ra.hour
	midnight = np.mod(12 +sunra, 24)
	rarange = 4
	raupper = midnight +rarange
	ralower = midnight -rarange
	print("Sun: " +str(sunra) +" hours")
	print("Midnight: " +str(midnight) +" hours")
	print("RA upper: " +str(raupper) +" hours")
	print("RA lower: " +str(ralower) +" hours")
	G = np.array([wiro[:,8][i][2:] for i in range(len(wiro[:,8]))]\
		, dtype = float)
	decdeg = np.array([wiro[:,3][i][:3] for i in range(len(wiro))]\
		, dtype = float)
	rahour = np.array([wiro[:,2][i][:2] for i in range(len(wiro))]\
		, dtype = float)
	names = np.array([wiro[:,1][i] for i in range(len(wiro))],\
		dtype = str)
	obs = np.array(np.loadtxt('/d/users/nikhil/Bowshocks/obslist/'
		'observed.txt', dtype = str))
	repeat = [obs == i for i in names]
	isrepeat = np.array([any(i) for i in repeat])
	if raupper > 24:
		a = np.mod(raupper, 24)
		b = ralower
		print('Searching between < ' +str(np.round(a,3)) \
			+" hours and > " +str(np.round(b,3)) \
			+' hours.')
		ii = (G > glim) & (~isrepeat) & (decdeg > -45)\
			& ((rahour <= a) | (rahour >= b))
	elif ralower < 0:
		a = raupper
		b = 24 +ralower
		print('Searching between > ' +str(np.round(a,3)) \
			+" hours and < " +str(np.round(b,3)) \
			+' hours.')
		ii = (G > glim) & (~isrepeat) & (decdeg > -45)\
			& ((rahour >= b) | (rahour <= a))
	else:
		print('Searching between < ' +str(np.round(raupper,3)) \
			+" hours and > " +str(np.round(ralower,3)) \
			+' hours.')
		ii = (G > glim) & (decdeg > -45) & (rahour <= raupper)\
			& (rahour >= ralower) & (~isrepeat)
	print(str(len(names[ii])) +' objects found.')
	return ii

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Module to write\
		a list of targets in APO format from a list of targets\
		in WIRO format. APO format example can be found in\
		/d/zem1/hak/chip/observing/APO/ZetaOphStars.txt.\
		WIRO format example can be found in\
		/d/zem1/hak/chip/Nikhil/Nikhiltargets.cat. Additionally,\
		limit selection by RA and Dec bounds and G magnitude.\
		\
		KOSMOS has an airmass limit of ')
	parser.add_argument('filename', type = str, help = 'Path of WIRO file.\
		Example: /d/users/nikhil/blah.cat')
	# NP Defining parser path argument
	parser.add_argument('name', type = str, help = \
		'Path of generated file. Example: /d/users/nikhil/name')
	# NP Defining parser name argument
	parser.add_argument('gmag', type = float, help = \
		'G Magnitude ceiling. Selection will exclude objects'
			' brighter than this constraint. Example: 12')
	# NP Setting g magnitude limit
	args = parser.parse_args()
	# NP Adding parser to run at command line
	wirofile = args.filename
	# NP Defining WIRO list
	wiro = np.loadtxt(wirofile , dtype = str, delimiter = ' ')
	# NP Reading in WIRO list
	n = names(wiro)
	# Extracting object names
	ras = ra(wiro)
	# NP Extracting object ras
	decs = dec(wiro)
	# NP Extracting objects decs
	gs = Gmags(wiro)
	# NP Extracting objects gmags
	APO = [n[i] +' ' +ras[i] +' ' +decs[i] +\
		' RotType=Horizon; RotAng=90'\
		for i in range(len(n))]
	# NP Generating array of objects in APO format
	index = limitobjs(wiro, args.gmag)
	dimmerAPO = np.array(APO)[index]
	np.savetxt(args.name +'.txt', APO, fmt = '%s')
	np.savetxt(args.name +'dim.txt', dimmerAPO, fmt = '%s')
	# NP Writing APO objects to a file with new inputted file name
	print("Done.")
	# NP Printing message when conversion is complete
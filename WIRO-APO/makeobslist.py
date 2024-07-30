import argparse
import numpy as np
import astropy as ap
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
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
	# NP Reading in gmag column
	mags = np.array([i[2: 7] for i in gmags])
	return mags

def limitobjs(wiro):
	'''Limits a list of objects based on inputted parameters.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-first: Boolean. Boolean indicating if observation is first half.
	-wiro: ndarray. A table containing objects in WIRO format.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-ii: ndarray. A boolean array corresponding to which objects
	satisfy the given constraints.'''
	sunra = get_sun(Time.now()).ra.hour
	# NP Finding the Sun's RA
	midnight = np.mod(12 +sunra, 24)
	# NP RA of meridian at midnight
	rarange = 6
	# NP RA range to search
	ralower = midnight -rarange
	raupper = midnight +rarange
	# NP Setting RA bounds to within 8 hours of midnight
	print("Sun: " +str(np.round(sunra, 3)) +" hours")
	print("Midnight: " +str(np.round(midnight, 3)) +" hours")
	print("RA upper: " +str(np.round(raupper, 3)) +" hours")
	print("RA lower: " +str(np.round(ralower, 3)) +" hours")
	# NP Displaying previous results
	G = np.array([wiro[:,8][i][2:] for i in range(len(wiro[:,8]))]\
		, dtype = float)
	# NP Defining G magnitudes from WIRO data
	dec = np.array([wiro[:,3][i] for i in range(len(wiro))]\
		, dtype = str)
	# NP Defining declination of each star
	ra = np.array([wiro[:,2][i] for i in range(len(wiro))]\
		, dtype = str)
	# NP Defining the Right Ascension of each star
	names = np.array([wiro[:,1][i] for i in range(len(wiro))],\
		dtype = str)
	# NP Defining name for each stars
	obs = np.array(np.loadtxt('/d/users/nikhil/Bowshocks/obslist/'
		'observed.txt', dtype = str))
	# NP Reading in already observed star names
	c = SkyCoord(ra, dec, unit = (u.hourangle, u.degree))
	# NP Making an array of SkyCoords for each target star
	ras = c.ra.hour
	# NP Defining RA's in decimal hours
	decs = c.dec.degree
	# NP Defining Dec's in decimal degrees
	repeat = [obs == i for i in names]
	isrepeat = np.array([any(i) for i in repeat])
	# NP Finding stars that are already observed
	if raupper > 24:
		a = np.mod(raupper, 24)
		b = ralower
		# NP Making more sensible bounds
		print('Searching between < ' +str(np.round(a,3)) \
			+" hours and > " +str(np.round(b,3)) \
			+' hours.')
		ii = (decs > -45)\
			& ((ras <= a) | (ras >= b)) & (~isrepeat)
		# NP Limiting selection
	# NP Making sense of upper RA bound greater than 24 hours
	elif ralower < 0:
		a = raupper
		b = 24 +ralower
		# NP Making more sensible bounds
		print('Searching between < ' +str(np.round(a,3)) \
			+" hours and > " +str(np.round(b,3)) \
			+' hours.')
		ii = (decs > -45)\
			& ((ras >= b) | (ras <= a)) & (~isrepeat)
		# NP Limiting selection
	# NP Making sense of lower RA bound lower than 0 hours
	else:
		print('Searching between < ' +str(np.round(raupper,3)) \
			+" hours and > " +str(np.round(ralower,3)) \
			+' hours.')
		ii = (decs > -45) & (ras <= raupper)\
			& (ras >= ralower) & (~isrepeat)
		# NP Limiting selection
	# NP Default case
	print(str(len(names[ii])) +' objects found.')
	# NP Printing how many stars were found in the limiting process
	return ii

def boolean_string(s):
	'''Translates string to boolean.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-s: str. Input from command line. Either true or False. Returns
	True if input is 'True'. Return False if input is 'False'.
	Returns error if none are inputted.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-out: Boolean. Boolean translation of string.'''
	if s not in {'False', 'True'}:
		raise ValueError('Not a valid boolean string')
	out = (s == 'True')
	return out

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Module to write\
		a list of targets in APO format from a list of targets\
		in WIRO format. APO format example can be found in\
		/d/zem1/hak/chip/observing/APO/ZetaOphStars.txt.\
		WIRO format example can be found in\
		/d/zem1/hak/chip/Nikhil/Nikhiltargets.cat. Additionally,\
		limit selection by RA and Dec bounds and G magnitude.\
		KOSMOS has an airmass limit of 4.5.')
	# NP Adding parser information
	parser.add_argument('name', type = str, help = \
		'Path of generated file. Example: /d/users/nikhil/name')
	# NP Defining parser name argument
	args = parser.parse_args()
	# NP Adding parser to run at command line
	wiro = np.loadtxt('/d/users/nikhil/Bowshocks/obslist/'\
		'Nikhiltargets.cat', dtype = str, delimiter = ' ')
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
		' RotType=Horizon; RotAng=90' 
		+'\n# G = ' +gs[i] for i in range(len(n))]
	# NP Generating array of objects in APO format
	test = np.array(['DispColor=#30638E'], dtype=str)
	# NP Setting color
	index = limitobjs(wiro)
	# NP Limiting selection
	dimmerAPO = np.array(APO)[index]
	# NP Whittling down list to observable targets
	full = np.concatenate((test, dimmerAPO))
	# NP Combining list with color setting
	np.savetxt(args.name +'APOtargets.txt', full, fmt = '%s')
	# NP Saving limited targets
	print("Done.")
	# NP Printing message when conversion is complete

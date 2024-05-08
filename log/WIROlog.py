import os
import argparse
import numpy as np
from astropy.io import fits
# NP Necessary imports

def findfitsfiles(directory):
	'''Finds fits files from in a given directory
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-directory: str. Path to files. Example: /d/...
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-fitsfiles: astropy.io fits.open files. Opened fits files.'''
	files = np.array([i[2] for i in os.walk(directory)])
	# NP Reading information from directory
	names = [i for i in files[0]]
	# NP Extracting the names of files in directory
	beginning = [i.rfind('.') for i in names]
	end = [len(i) for i in names]
	# NP Finding index of extension
	extensions = [names[i][beginning[i]:end[i]] for i in \
		range(len(names))]
	# NP Generating list of files with '.fits' extensions
	ii = ([i == '.fit' for i in extensions])
	# NP Boolean condition to find only fits files
	fitsfiles = np.array(names)[ii]
	# NP Creating list of files that are fits files
	print(str(len(fitsfiles)) +' fits files found out of '\
		+str(len(names)) +' files inputted.')
	# NP Printing how many fits files were found
	return fitsfiles

def generatelogtxt(fitsfiles):
	'''Generates text for a log file
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-fitsfiles: np.array. List of fits files in directory
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-log: String. A string for a log file.'''
	fitsopen = [fits.open(args.path +str(i)) for i in fitsfiles]
	# NP Opening fits files
	identifier = np.array([a[1:4] for a in fitsfiles], dtype = int)
	# NP Interpreting file number from list of fits files
	polishedindices = np.argsort(identifier)
	# NP Sorting file number from least to greatest
	dates = np.array([i[0].header['DATE-OBS'] for i in fitsopen])
	# NP Reading in dates from fits headers
	sorteddates = np.array(dates[polishedindices])
	# NP Redefining sorteddates from polishedindices
	sortednames = np.array(fitsfiles[polishedindices])
	# NP Creating a list of file names in order
	sortedobjnames = np.array([i[0].header['OBJNAME'] for i in \
		fitsopen])[polishedindices]
	# NP Creating a lit of object names in order
	ii = np.array([len(i) > 5 for i in sortedobjnames])
	# NP Creating index of obj names longer than 5 characters
	sortedobjnames[ii] = np.array([i[:5] for i in sortedobjnames[ii]])
	# NP Limiting names to 5 characters or less
	sortedexp = np.array([i[0].header['EXPTIME'] for i in fitsopen])\
		[polishedindices]
	# NP Creating a list of exptimes in order
	sortedtimes = np.array([i[11:16] for i in sorteddates])
	# NP Creating a list of TAI times in order
	sortedairmass = np.array([i[0].header['AIRMASS'] for i in \
		fitsopen])[polishedindices]
	# NP Creating a list of airmasses in order
	log = 'Day Month Year (UT)\n'\
		+str(dates[int(np.median(polishedindices))][8:10]) \
		+' '\
		+str(dates[int(np.median(polishedindices))][5:7]) +' '\
		+str(dates[int(np.median(polishedindices))][0:4])+'\n'\
		+'Observer: Nikhil Patten\n'\
		+'CONDITIONS\n'\
		+'Image\t\tObject\texp\tUT\tX\tNotes\n'
	# NP Creating header of log file
	for i in range(len(polishedindices)):
	    	log += str(sortednames[i]) \
	   		+'\t' +str(sortedobjnames[i]) +'\t' \
			+str(sortedexp[i]) \
	    		+' \t' +str(sortedtimes[i]) +'\t' \
			+str(np.round(sortedairmass[i],3)) +'\t\n'
	# NP Writing meat of log file
	return log

if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Module to write\
		a log of observations from a given night of science\
		files.')
	parser.add_argument('path', type = str, help = 'Path to files.\
		Example: /d/...')
	# NP Defining parser path argument
	args = parser.parse_args()
	# NP Adding parser to run at command line
	fitsfiles = findfitsfiles(args.path)
	# NP Finding fits files
	log = generatelogtxt(fitsfiles)
	# NP Writing log
	print(log)
	# NP Printing log
	textfile = open(args.path +'log.txt', 'w')
	textfile.write(log)
	textfile.close
	# NP Writing out log file

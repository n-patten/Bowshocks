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
	ii = ([i == '.fits' for i in extensions])
	# NP Boolean condition to find only fits files
	fitsfiles = np.array(names)[ii]
	# NP Creating list of files that are fits files
	print(str(len(fitsfiles)) +' fits files found out of '\
		+str(len(names)) +' files inputted.')
	# NP Printing how many fits files were found
	return fitsfiles

def generatelogtxt(fitsopen):
	'''Generates text for a log file
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Inputs
	-fitsopen: Astropy fitsopen. Opening fits files
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Outputs
	-log: String. A string for a log file.'''
	dates = np.array([i[0].header['DATE-OBS'] for i in fitsopen])
	# NP Reading in dates from fits headers
	sorteddates = sorted(dates)
	# NP Sorting dates
	indices = np.array([np.where(dates == i) for i in sorteddates])
	# NP Matching indices of sorteddates to regular dates
	polishedindices = [i.item() for i in indices]
	# NP Creating array of indices ordering fits files in order
	sorteddates = np.array(dates[polishedindices])
	# NP Redefining sorteddates from polishedindices
	sortednames = np.array(fitsfiles[polishedindices])
	# NP Creating a list of file names in order
	sortedobjs = np.array([i[0].header['IMAGETYP'] for i in \
		fitsopen])[polishedindices]
	# NP Creating a list of objs in order
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
	ii = np.array([(i == 'Flat') | (i == 'Bias') for i in sortedobjs])
	# NP Creating indices of non object type exposures
	sortedobjnames[ii] = np.array(["" for i in sortedobjnames[ii]])
	# NP Setting name to be blank if obj is not a star
	sortedslits = np.array([i[0].header['SLIT'] for i in \
		fitsopen])[polishedindices]
	# NP Creating a lit of filters in order
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
		+'Image\t\tObject\tName\texp\tSlit\t\tTAI\tX\tNotes\n'
	# NP Creating header of log file
	for i in range(len(polishedindices)):
	    	log += str(sortednames[i]) +'\t' +str(sortedobjs[i]) \
	   		+'\t' +str(sortedobjnames[i]) +'\t' \
			+str(sortedexp[i]) \
			+'\t' +str(sortedslits[i]) \
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
	fitsopen = [fits.open(args.path +str(i)) for i in fitsfiles]
	# NP Opening fits files
	log = generatelogtxt(fitsopen)
	# NP Writing log
	print(log)
	# NP Printing log
	textfile = open(args.path +'log.txt', 'w')
	textfile.write(log)
	textfile.close
	# NP Writing out log file

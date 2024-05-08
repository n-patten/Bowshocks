import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# NP Necessary imports

def read_sheet(direc):
	'''Reads in comparison data from spreadsheet and returns
	relevent information.
	---------------------------------------------
	Inputs
	-direc: str. Directory to spreadsheet of data.
	---------------------------------------------
	Outputs
	-names: np.array. Array of HD identifiers of each comparison
	star.
	'''
	sheet = np.loadtxt(direc, delimiter = ',', \
		skiprows = 1, dtype = str, usecols = (1, 2, 3, 4, \
		5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
	# NP Reading relevent columns from spreadsheet
	names = np.array([s[0] for s in sheet])
	# NP Extracting HD identifier from sheet
	mts = np.array([s[1] for s in sheet], dtype = float)
	# NP Extracting measured temperatures from sheet
	mt_upper = np.array([s[2] for s in sheet], dtype = float)
	# NP Extracting upper uncertainty in measured temperature
	# NP from sheet
	mt_lower = np.array([s[3] for s in sheet], dtype = float)
	# NP Extracting lower uncertainty in measured temperature
	# NP from sheet
	mgs = np.array([s[4] for s in sheet], dtype = float)
	# NP Extracting measured logg from sheet
	mg_upper = np.array([s[5] for s in sheet], dtype = float)
	# NP Extracting upper uncertainty in measured logg from 
	# NP sheet
	mg_lower = np.array([s[6] for s in sheet], dtype = float)
	mvs = np.array([s[7] for s in sheet], dtype = float)
	mv_upper = np.array([s[8] for s in sheet], dtype = float)
	mv_lower = np.array([s[9] for s in sheet], dtype = float)
	lts = np.array([s[10] for s in sheet], dtype = float)
	olt = np.array([s[11] for s in sheet], dtype = float)
	lgs = np.array([s[12] for s in sheet], dtype = float)
	olg = np.array([s[13] for s in sheet], dtype = float)
	lvs = np.array([s[14] for s in sheet], dtype = float)
	olv = np.array([s[15] for s in sheet], dtype = float)
	return names, mts, mt_upper, mt_lower, mgs, mg_upper, \
		mg_lower, mvs, mv_upper, mv_lower, lts, olt, lgs, \
		olg, lvs, olv
	
def scatter_t(mts, mt_upper, mt_lower, lts, olt, t):
	plt.figure(figsize = [12, 8], facecolor = 'white')
	ax1 = plt.subplot(1, 2, 1)
	ii = (mts != -99) & (mt_upper != -99) & \
	(mt_lower != -99) & (lts !=-99)
	true_olt = np.array([olt[i] if olt[i] !=-99 else 1000 \
		for i in range(len(olt))])
	ax1.errorbar(lts[ii]/1000, mts[ii] /1000, yerr=(mt_lower[ii]/1000, \
		mt_upper[ii]/1000), xerr = true_olt[ii]/1000, fmt = '.', \
		color = color1)
	plt.tick_params('x', labelbottom=False)
	test = np.linspace(np.min([np.min(mts[ii]), np.min(lts[ii])]), \
		np.max([np.max(mts[ii]), np.max(lts[ii])]), 1000)
	plt.plot(test/1000, test /1000, '--k')
	plt.xlabel(r'$T$ (Literature) ($kK$)')
	plt.ylabel(r'$T$ (This work) ($kK$)')
	ax2 = plt.subplot(1, 2, 2, sharey = ax1)
	plt.hist(t/1000, bins = np.arange(20, 50, 5), edgecolor = \
		'black', linewidth = 1.2, color = color1, \
		orientation = 'horizontal')
	ax2.set_xticks(np.arange(0, 22, 2))
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.subplots_adjust(hspace=0)
	plt.xlabel(r'$N$ (Number of stars)')
	plt.tight_layout()
	plt.savefig('./calibrators/temp_comparison.eps', \
		bbox_inches='tight')
	
def scatter_g(mgs, mg_upper, mg_lower, lgs, olg, g):
	plt.figure(figsize = [12, 8], facecolor = 'white')
	ax1 = plt.subplot(1, 2, 1)
	ii = (mgs != -99) & (mg_upper != -99) & \
	(mg_lower != -99) & (lgs !=-99)
	true_olg = np.array([olg[i] if olg[i] !=-99 else 0.2 \
		for i in range(len(olg))])
	ax1.errorbar(lgs[ii], mgs[ii], yerr=(mg_lower[ii], \
		mg_upper[ii]), xerr = true_olg[ii], fmt = '.', \
		color = color2)
	plt.tick_params('x', labelbottom=True)
	test = np.linspace(np.min([np.min(mgs[ii]), np.min(lgs[ii])]), \
		np.max([np.max(mgs[ii]), np.max(lgs[ii])]), 1000)
	plt.plot(test, test, '--k')
	plt.xlabel(r'$\log g$ (Literature) (cgs)')
	plt.ylabel(r'$\log g$ (This work) (cgs)')
	ax2 = plt.subplot(1, 2, 2, sharey = ax1)
	plt.hist(g, bins = np.arange(2.0, 5.5, 0.5), edgecolor = \
		'black', linewidth = 1.2, color = color2, \
		orientation = 'horizontal')
	ax2.set_xticks(np.arange(0, 30, 2))
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.subplots_adjust(hspace=0)
	plt.xlabel(r'$N$ (Number of stars)')
	plt.tight_layout()
	plt.savefig('./calibrators/g_comparison.eps', \
		bbox_inches='tight')
	
def scatter_v(mvs, mv_upper, mv_lower, lvs, olv, v):
	plt.figure(figsize = [12, 8], facecolor = 'white')
	ax1 = plt.subplot(1, 2, 1)
	ii = (mgs != -99) & (mg_upper != -99) & \
	(mg_lower != -99) & (lgs !=-99)
	true_olv = np.array([olv[i] if olv[i] !=-99 else 0.1 *lvs[i] \
		for i in range(len(olv))])
	ax1.errorbar(lvs[ii], mvs[ii], yerr=(mv_lower[ii], \
		mv_upper[ii]), xerr = true_olv[ii], fmt = '.', \
		color = color3)
	plt.tick_params('x', labelbottom=True)
	test = np.linspace(np.min([np.min(mvs[ii]), np.min(lvs[ii])]), \
		np.max([np.max(mvs[ii]), np.max(lvs[ii])]), 1000)
	plt.plot(test, test, '--k')
	plt.xlabel(r'$v\sin i$ (Literature) (km s$^{-1}$)')
	plt.ylabel(r'$v\sin i$ (This work) (km s$^{-1}$)')
	ax2 = plt.subplot(1, 2, 2, sharey = ax1)
	plt.hist(v, bins = np.arange(0, 450, 50), edgecolor = \
		'black', linewidth = 1.2, color = color3, \
		orientation = 'horizontal')
	ax2.set_xticks(np.arange(0, 18, 2))
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.subplots_adjust(hspace=0)
	plt.xlabel(r'$N$ (Number of stars)')
	plt.tight_layout()
	plt.savefig('./calibrators/v_comparison.eps')
	
def read_mass_sheet(direc):
	sheet = np.loadtxt(direc, delimiter = ',', \
		skiprows = 2, dtype = str, usecols = (8, 9, 10, 12))
	ii = [s[0] != '' and s[1] != '-99' and s[1] != '' for s in sheet]
	temps = np.array([s[0] for s in sheet[ii]], dtype = float)
	gs = np.array([s[2] for s in sheet[ii]], dtype = float)
	vs = np.array([s[3] for s in sheet[ii]], dtype = float)
	return temps, gs, vs

if (__name__ == '__main__'):
	d = './Comparisons.csv'
	# NP Defining path to data sheet
	names, mts, mt_upper, mt_lower, mgs, mg_upper, mg_lower, \
		mvs, mv_upper, mv_lower, lts, olt, lgs, olg, lvs, \
		olv = read_sheet(d)
	# NP Reading information from sheet
	color1 = '#5FAD56'
	color2 = '#F2C14E'
	color3 = '#F78154'
	md = './masslossrates.csv'
	ts, gs, vs = read_mass_sheet(md)
	scatter_t(mts, mt_upper, mt_lower, lts, olt, ts)
	# NP Plotting temperature comparison
	scatter_g(mgs, mg_upper, mg_lower, lgs, olg, gs)
	# NP Plotting logg comparison
	scatter_v(mvs, mv_upper, mv_lower, lvs, olv, vs)
	# NP Plotting vsini comparison
	
	

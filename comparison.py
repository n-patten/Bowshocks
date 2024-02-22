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
	mgs = np.array([s[4] for s in sheet], dtype = float)
	mg_upper = np.array([s[5] for s in sheet], dtype = float)
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
	
def scatter_t(mts, mt_upper, mt_lower, lts, olt):
	plt.figure(figsize = [8, 6], facecolor = 'white')
	ii = (mts != -99) & (mt_upper != -99) & \
	(mt_lower != -99) & (lts !=-99)
	true_olt = np.array([olt[i] if olt[i] !=-99 else 1000 \
		for i in range(len(olt))])
	plt.errorbar(lts[ii], mts[ii], yerr=(mt_lower[ii], \
		mt_upper[ii]), xerr = true_olt[ii], fmt = '.', \
		color = 'red')
	test = np.linspace(np.min([np.min(mts[ii]), np.min(lts[ii])]), \
		np.max([np.max(mts[ii]), np.max(lts[ii])]), 1000)
	plt.plot(test, test, '--k')
	plt.xlabel(r'Literature $T$ ($K$)')
	plt.ylabel(r'Measured $T$ ($K$)')
	plt.title(r'Comparison of $T$')
	plt.savefig('./temp_comparison.eps')
	
def scatter_g(mgs, mg_upper, mg_lower, lgs, ogt):
	plt.figure(figsize = [8, 6], facecolor = 'white')
	ii = (mgs != -99) & (mg_upper != -99) & \
	(mg_lower != -99) & (lgs !=-99)
	true_olg = np.array([olg[i] if olg[i] !=-99 else 0.10 \
		for i in range(len(ogt))])
	plt.errorbar(lgs[ii], mgs[ii], yerr=(mg_lower[ii], \
		mg_upper[ii]), xerr = true_olg[ii], fmt = '.', \
		color = 'green')
	test = np.linspace(np.min([np.min(mgs[ii]), np.min(lgs[ii])]), \
		np.max([np.max(mgs[ii]), np.max(lgs[ii])]), 1000)
	plt.plot(test, test, '--k')
	plt.xlabel(r'Literature $\log g$')
	plt.ylabel(r'Measured $\log g$')
	plt.title(r'Comparison of $\log g$')
	plt.savefig('./g_comparison.eps')
	
def scatter_v(mvs, mv_upper, mv_lower, lvs, olv):
	plt.figure(figsize = [8, 6], facecolor = 'white')
	ii = (mvs != -99) & (mv_upper != -99) & \
	(mv_lower != -99) & (lvs !=-99)
	true_olv = np.array([olv[i] if olv[i] !=-99 else 0.10 \
		*lvs[i] for i in range(len(olv))])
	plt.errorbar(lvs[ii], mvs[ii], yerr=(mv_lower[ii], \
		mv_upper[ii]), xerr = true_olv[ii], fmt = '.', \
		color = 'blue')
	test = np.linspace(np.min([np.min(mvs[ii]), np.min(lvs[ii])]), \
		np.max([np.max(mvs[ii]), np.max(lvs[ii])]), 1000)
	plt.plot(test, test, '--k')
	plt.xlabel(r'Literature $v\sin i$')
	plt.ylabel(r'Measured $v\sin i$')
	plt.title(r'Comparison of $v\sin i$')
	plt.savefig('./v_comparison.eps')

if (__name__ == '__main__'):
	d = './Comparisons.csv'
	names, mts, mt_upper, mt_lower, mgs, mg_upper, mg_lower, \
		mvs, mv_upper, mv_lower, lts, olt, lgs, olg, lvs, \
		olv = read_sheet(d)
	scatter_t(mts, mt_upper, mt_lower, lts, olt)
	scatter_g(mgs, mg_upper, mg_lower, lgs, olg)
	scatter_v(mvs, mv_upper, mv_lower, lvs, olv)
	
	

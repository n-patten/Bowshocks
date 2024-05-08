import os
import emcee
import corner
import random
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
# NP Importing necessary packages

from scipy import stats
from astropy.io import fits
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
# NP Importing specific functions from packages

def get_POWR(APO):
	'''Reads in model spectra for given instrument and sets model
	variables
	---------------------------------------------
	Inputs
	-APO: boolean. Boolean indicating whether spectrum was
	obtained on APO KOSMOS spectrograph. Example: True
	---------------------------------------------
	Outputs
	-ctemps: array. Array of temperatures from library of models.
	-cgs: array. Array of gravities from library of models.
	-vsini: array. Array of vsini from library of models.
	-cwavls: array. Array of wavelength information from models.
	-cins: array. Array of normalized flux values from models.
	'''
	if APO:
		print('Reading in APO model spectra.\n')
		cdir = '/d/hya1/nikhil/BS/model_spectra/PoWR/APO/'
		# NP Directory for convolved spectra
	else:
		print('Reading in WIRO model spectra.\n')
		cdir = '/d/hya1/nikhil/BS/model_spectra/PoWR/WIRO/'
		# NP Directory for convolved spectra
	cnames = np.array(os.listdir(cdir))
	# NP Convolved model names
	cifiles = [n[-4:] == '.txt' for n in cnames]
	# NP Limiting to only text files
	cmodel = [np.genfromtxt(cdir +i, filling_values= \
		{0:'-111'}, dtype = str, delimiter = \
		'         ') for i in cnames[cifiles]]
	# NP reading in model data
	ctemps = np.array([float(n[-12:-7]) for n in \
		cnames[cifiles]])
	# NP Finding temperature information for each model
	cgs = np.array([float(n[-7:-4]) /100 for n in \
		cnames[cifiles]])
	# NP Finding gravity information for each model
	vsini = np.array([float(n[4:-12]) for n in \
		cnames[cifiles]])
	# NP Finding vsini convolution for model
	cwavls = np.array([np.array([float(i[0:8]) for \
		i in l])for l in cmodel], dtype = object)
	# NP Reading in vacuum wavelengths
	cints = np.array([np.array([float(i[8:]) for i in \
		l])for l in cmodel], dtype = object)
	# NP Reading in convolved model intensities
	return ctemps, cgs, vsini, cwavls, cints
	print('Done!\n')

def get_Bstars(APO):
	'''Reads in model spectra for given instrument and sets model
	variables
	---------------------------------------------
	Inputs
	-APO: boolean. Boolean indicating whether spectrum was obtained
	on APO KOSMOS spectrograph. Example: True
	---------------------------------------------
	Outputs
	-None.
	'''
	if APO:
		print('Reading in APO model spectra.\n')
		cdir = '/d/hya1/nikhil/BS/model_spectra/TLUSTY/APO/BSTAR/'
		# NP Directory for convolved spectra
	else:
		print('Reading in WIRO model spectra.\n')
		cdir = '/d/hya1/nikhil/BS/model_spectra/TLUSTY/WIRO/BSTAR/'
		# NP Directory for convolved spectra
	print('Reading in B star models...\n')
	cnames = np.array(os.listdir(cdir))
	# NP Convolved model names
	cifiles = [n[-4:] == '.txt' for n in cnames]
	# NP Limiting to only text files
	cmodel = [np.genfromtxt(cdir +i, filling_values= \
		{0:'-111'}, dtype = str) for i in cnames[cifiles]]
	# NP reading in model data
	ctemps = np.array([float(n[-12:-7]) for n in \
		cnames[cifiles]])
	# NP Finding temperature information for each model
	cgs = np.array([float(n[-7:-4]) /100 for n in \
		cnames[cifiles]])
	# NP Finding gravity information for each model
	vsini = np.array([float(n[4:-12]) for n in \
		cnames[cifiles]])
	# NP Finding vsini convolution for model
	cwavls = np.array([np.array([a[0] for a in i], \
		dtype = float) for i in cmodel], dtype = object)
	# NP Reading in vacuum wavelengths
	cints = np.array([np.array([a[1] for a in i], \
		dtype = float) for i in cmodel], dtype = object)
	# NP Reading in convolved model intensities
	return ctemps, cgs, vsini, cwavls, cints
	print('Done!\n')

def get_Ostars(APO):
	'''Reads in model spectra for given instrument and sets model
	variables
	---------------------------------------------
	Inputs
	-APO: boolean. Boolean indicating whether spectrum was obtained
	on APO KOSMOS spectrograph. Example: True
	---------------------------------------------
	Outputs
	-None.
	'''
	if APO:
		print('Reading in APO model spectra.\n')
		cdir = '/d/hya1/nikhil/BS/model_spectra/TLUSTY/APO/OSTAR/'
		# NP Directory for convolved spectra
	else:
		print('Reading in WIRO model spectra.\n')
		cdir = '/d/hya1/nikhil/BS/model_spectra/TLUSTY/WIRO/OSTAR/'
		# NP Directory for convolved spectra
	print('Reading in O star models...\n')
	cnames = np.array(os.listdir(cdir))
	# NP Convolved model names
	cifiles = [n[-4:] == '.txt' for n in cnames]
	# NP Limiting to only text files
	cmodel = [np.genfromtxt(cdir +i, filling_values= \
		{0:'-111'}, dtype = str) for i in cnames[cifiles]]
	# NP reading in model data
	ctemps = np.array([float(n[-12:-7]) for n in \
		cnames[cifiles]])
	# NP Finding temperature information for each model
	cgs = np.array([float(n[-7:-4]) /100 for n in \
		cnames[cifiles]])
	# NP Finding gravity information for each model
	vsini = np.array([float(n[4:-12]) for n in \
		cnames[cifiles]])
	# NP Finding vsini convolution for model
	cwavls = np.array([np.array([a[0] for a in i], \
		dtype = float) for i in cmodel], dtype = object)
	# NP Reading in vacuum wavelengths
	cints = np.array([np.array([a[1] for a in i], \
		dtype = float) for i in cmodel], dtype = object)
	# NP Reading in convolved model intensities
	return ctemps, cgs, vsini, cwavls, cints
	print('Done!\n')

def guessmodels(btemps, bgs, bvsini, bwavls, bints, otemps, ogs, \
	ovsini, owavls, oints, wav, dat):
	'''Guesses which grid of models to interpolate over in MCMC
	fitting.
	---------------------------------------------
	Inputs
	-btemps: np.array. Temperature grid of B models
	-bgs: np.array. logg grid of B models
	-bvsini: np.array. vsini grid of B models
	-bwavls: np.array. Wavelength grid of B models.
	-bints: np.array. Normalized intensity of B models.
	-otemps: np.array. Temperature grid of O models
	-ogs: np.array. logg grid of O models
	-ovsini: np.array. vsini grid of O models
	-owavls: np.array. Wavelength grid of O models.
	-oints: np.array. Normalized intensity of O models.
	-wav: np.array. Wavelength array of data spectrum.
	-dat: np.array. Normalized data spectrum.
	---------------------------------------------
	Outputs
	-temps. np.array. Temperature grid of chosen models.
	-gs. np.array. log g grid of chosen models.
	-vsini. -np.array. vsini grid of chosen models.
	-wavls. np.array. Wavelength grid of chosen models.
	-ints. np.array. Normalized spectrum grid of chosen models.
	'''
	bspectrumspline = CubicSpline(wav, dat)
	# NP Creating a spline of the spectrum
	btestrange = np.linspace(4000, 4990, 1000)
	# NP Wavelength range to evaluate differences
	# NP between models and spectra
	bmodelsplines = [CubicSpline(bwavls[a], \
		bints[a]) for a in range(len(bints))]
	# NP Creating a spline of all model B spectra
	bdiffs = np.array([np.array([((dat[ii] \
		-bmodelsplines[i](wav[ii]))) for ii in \
		range(len(wav)) if wav[ii] > 3900]) for i in \
		range(len(bmodelsplines))])
	# NP Evaluating deviation of data from each model spectrum
	omodelsplines = [CubicSpline(owavls[a], \
		oints[a]) for a in range(len(oints))]
	# NP Creating a spline of all model O spectra
	odiffs = np.array([np.array([((dat[ii] \
		-omodelsplines[i](wav[ii]))) for ii in \
		range(len(wav)) if wav[ii] > 3900]) for i in \
		range(len(omodelsplines))])
	# NP Evaluating deviation of data from each model spectrum
	bmins = [np.min([np.sum(b **2) for b in bdiffs\
		[btemps == t]]) for t in \
		np.arange(15000, 31000, 1000)]
	# NP Finding best-fit model at each temperature in the B grid
	btempi = np.linspace(15000, 30000, 10000)
	# NP Creating B grid temperature array
	bspline = CubicSpline(np.arange(1.5e4, 3.1e4, 1e3), bmins)
	# NP Interpolating over all models to find best-fit model in
	# NP B grid
	omins = [np.min([np.sum(o **2) for o in odiffs\
		[otemps == t]]) for t in \
		np.arange(27500, 57500, 2500)]
	# NP Finding best-fit model at each temperature in the O grid
	otempi = np.linspace(27500, 55000, 10000)
	# Creating O grid temperature array
	ospline = CubicSpline(np.arange(2.75e4, 5.75e4, 2.5e3), \
		omins)
	plt.plot(otempi, ospline(otempi), color = 'blue', \
		label = 'OSTAR')
	plt.plot(btempi, bspline(btempi), color = 'lightblue', \
		label = 'BSTAR')
	plt.legend()
	# NP Interpolating over all models to find best-fit model in
	# NP O grid
	ochimin = np.min(ospline(otempi))
	# NP Finding best-fit interpolated O model
	bchimin = np.min(bspline(btempi))
	# NP Finding best-fit interpolated B model
	#print('B temp: ' +str(btemps[np.sum() == np.min(bmins)]))
	#print('O temp: ' +str(otemps[odiffs == np.min(omins)]))
	if ochimin < bchimin:
		print('Choosing O star grid\n')
		return otemps, ogs, ovsini, owavls, oints
		# NP Choosing O grid if O models are best-fit
	else:
		print('Choosing B star grid\n')
		return btemps, bgs, bvsini, bwavls, bints
		# NP Choosing B grid if B models are best-fit

def guess(wav, dat):
	'''Guesses the parameters T, vsini and log g for a Bowshock
	star and returns a residual spectrum of the best-fit model.
	---------------------------------------------
	Inputs
	-spec: str. BS identifier for the star. Example: 'BS013'.
	---------------------------------------------
	Outputs
	-t_test: float. Best fit parameter for temperaure in Kelvin.
	-g_test: float. Best fit parameter for log g.
	-v_test: float. Best fit parameter for vsini in km/s.
	-resids: array. Residual spectrum of the best-fit model
	'''
	spectrumspline = CubicSpline(wav, dat)
	# NP Creating a spline of the spectrum
	testrange = np.linspace(4000, 4990, 1000)
	# NP Wavelength range to evaluate differences
	# NP between models and spectra
	modelsplines = [CubicSpline(cwavls[a], \
		cints[a]) for a in range(len(cints))]
	# NP Creating a spline of all model spectra
	diffs = np.array([np.array([((dat[ii] \
		-modelsplines[i](wav[ii]))) for ii in \
		range(len(wav)) if wav[ii] > 3900 and wav[ii] < 5100]) for i in \
		range(len(modelsplines))])
	# NP Evaluating deviation of data from each model spectrum
	smin = np.argmin([np.sum(r **2) for r in diffs])
	# NP Finding minimum chi squared
	print('T_eff: ' +str(ctemps[smin]))
	print('log g: ' +str(cgs[smin]))
	print('v sini: ' +str(vsini[smin]) +'\n')
	# NP Printing best-fit paramteres
	dmin = np.array([dat[i] -modelsplines[smin](wav[i]) for i \
		in range(len(wav))])
	print('dmin: ' +str(dmin))
	t_test = ctemps[smin]
	g_test = cgs[smin]
	v_test = vsini[smin]
	return t_test, g_test, v_test, dmin
	# NP Returning best-fit parameters

def interpspectra(T_targ, g_targ, v_targ, plot):
	'''Interpolates between different temperature at and different
	log g model spectra. Uses a bilinear interpolation between 
	temperatures and log g to interpolate between four model
	spectra.
	---------------------------------------------
	Inputs
	-T_targ: float. Target temperature to interpolate to in
	Kelvin
	-g_targ: float. Target log g to interpolate to for models.
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-wav_new: array. Wavelength array in Angstroms.
	-spline(wav_new): float. Spline of interpolated spectrum.
	'''
	try:
		tempdiffs = ctemps -T_targ
		# NP Finding the temperature difference between all
		# NP model spectra and target temperature.
		gdiffs = cgs -g_targ
		# NP Finding the gravity difference between all
		# NP model spectra and target gravity.
		vdiffs = vsini -v_targ
		# NP Finding the rotational velocity differences
		# between all model spectra and target gravity.
		T_0 = T_targ +tempdiffs[tempdiffs < 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs < 0]))]
		# NP Finding lower temperature
		T_1 = T_targ +tempdiffs[tempdiffs > 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs > 0]))]
		# NP Finding upper temperature
		g_0 = g_targ +gdiffs[(ctemps == T_0) & (gdiffs < 0)]\
			[np.argmin(np.abs(gdiffs[(ctemps == T_0) \
			& (gdiffs < 0)]))]
		# NP Finding lower gravity
		g_1 = g_targ +gdiffs[(ctemps == T_1) & (gdiffs > 0)]\
			[np.argmin(np.abs(gdiffs[(ctemps == T_1) \
			& (gdiffs > 0)]))]
		# NP Finding upper gravity
		v_0 = v_targ +vdiffs[vdiffs < 0][np.argmin(\
			np.abs(vdiffs[vdiffs < 0]))]
		# NP Finding lower gravity
		v_1 = v_targ +vdiffs[vdiffs > 0][np.argmin(\
			np.abs(vdiffs[vdiffs > 0]))]
		# NP Finding upper gravity
		wavs0 = np.array(cwavls, dtype=object)[(cgs == g_0) \
			& (ctemps == T_0) & (vsini == v_0)][0]
		# NP Finding the lower convolution wavelengths
		wavs1 = np.array(cwavls, dtype=object)[(cgs == g_0) \
			& (ctemps == T_0) & (vsini == v_1)][0]
		# NP Finding the higher convulation wavelengths
		ints000 = np.array(cints, dtype=object)[(cgs == g_0) \
			& (ctemps == T_0) & (vsini == v_0)][0]
		# NP Finding the model with low temperature low log g
		# NP and low vsini
		ints010 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_0) & (vsini == v_0)][0]
		# NP Finding the model with low temperature high log g 
		# NP and low vsini
		ints100 = np.array(cints, dtype=object)[(cgs == g_0) \
			& (ctemps == T_1) & (vsini == v_0)][0]
		# NP Finding the model with high temperature low log g 
		# NP and low vsini
		ints001 = np.array(cints, dtype=object)[(cgs == g_0) \
			& (ctemps == T_0) & (vsini == v_1)][0]
		# NP Finding the model with low temperature low log g
		# NP and high vsini
		ints110 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_1) & (vsini == v_0)][0]
		# NP Finding the model with high temperature high log g 
		# NP and low vsini
		ints011 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_0) & (vsini == v_1)][0]
		# NP Finding the model with low temperature high log g 
		# NP and high vsini
		ints101 = np.array(cints, dtype=object)[(cgs == g_0) \
			& (ctemps == T_1) & (vsini == v_1)][0]
		# NP Finding the model with high temperature low log g
		# NP and high vsini
		ints111 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_1) & (vsini == v_1)][0]
		# NP Finding the model with high temperature high log 
		# NP g and high vsini
		T_d = (T_targ -T_0) /(T_1 -T_0)
		# NP Finding percent change in desired temperature
		g_d = (g_targ -g_0) /(g_1 -g_0)
		# NP Finding percent change in desired gravity
		v_d = (v_targ -v_0) /(v_1 -v_0)
		# NP Finding percent change in rotational velocity
		c00 = ints000 *(1 -T_d) +ints100 *T_d
		# NP Finding bilinear interpolation between log g with
		# NP low vsini and high T
		c01 = ints001 *(1 -T_d) +ints101 *T_d
		# NP Finding bilinear interpolation between log g with
		# NP high vsini and high T
		c10 = ints010 *(1 -T_d) +ints110 *T_d
		# NP Finding bilinear interpolation between log g with
		# NP low vsini and low T
		c11 = ints011 *(1 -T_d) +ints111 *T_d
		# NP Finding bilinear interpolation between log g with
		# NP high vsini and low T
		c0 = c00 *(1 -g_d) +c10 *g_d
		# NP Finding bilinear interpolation between T with
		# NP interpolated log g and low
		# NP vsini
		c1 = c01 *(1 -g_d) +c11 *g_d
		# NP Finding bilinear interpolation between T with
		# NP interpolated log g and high vsini
		wav_new = np.linspace(3850, np.max(wavl), 4096)
		# NP Creating wavelength array to evalueate splines 
		# NP over
		spl_zero = CubicSpline(wavs0, c0)
		spl_one = CubicSpline(wavs1, c1)
		# NP Creating splines of interpolated spectra at low 
		# NP vsini and high vsini to interpolate over them. 
		# NP These two sets of models have different 
		# NP resolutions
		c = spl_zero(wav_new) *(1 -v_d) +spl_one(wav_new) *v_d
		# NP Finding combined gravity interpolated and
		# NP temperature interpolated spectrum
		spline = CubicSpline(wav_new, c)
		# NP Trilinear interpolation Wikipedia page was used
		# NP extensively when writing this function. A lot of
		# NP the labels and variable names reflect this
		# NP influence. See this page for clarification and
		# NP visualization:
		# NP https://en.wikipedia.org/wiki/Trilinear_interpolation
		if(plot):
		# NP If plotting is desired:
			plt.plot(wavs1, c00, label = 'c00')
			# NP Plotting c00 interpolation
			plt.plot(wavs2, c01, label = 'c01')
			# NP Plotting c01 interpolation
			plt.plot(wavs1, c10, label = 'c10')
			# NP Plotting c10 interpolation
			plt.plot(wavs2, c11, label = 'c11')
			# NP Plotting c11 interpolation
			plt.plot(wavs2, c1, label = 'c1')
			# NP Plotting c1 interpolation
			plt.plot(wavs1, c0, label = 'c0')
			# NP Plotting c0 interpolation
			plt.plot(wav_new, spline(wav_new), \
				label ='new interpolation, ' \
				'T={0:.1f} K'.format(T_targ))
			# NP Plotting c interpolation
			plt.plot(wavl, data, label = 'data')
			# NP Plotting interpolated spectrum
			plt.legend()
			# NP Adding a legend
			plt.xlim(4000, 4200)
			# NP Limiting plot to 4000-4200 Angstroms
			plt.show()
			# NP Showing plot
		# NP Creating spline of interpolated spectrum
		return wav_new, spline
		# NP Returning wavelength array and interpolated
		# NP spectrum
	except:
		print('Could not interpolate these parameters!\n')
		print(T_targ, g_targ, v_targ)
		return np.nan, np.nan
		# NP Print this string if temperature could not be
		# NP interpolated

def noise_estimation(w, d, plot):
	'''Function used to determine the statistical noise component
	of a given spectrum. This quantifies the random deviation two
	pixel are likely to differ by and is a measure of S/N.
	---------------------------------------------
	Inputs
	-w: array. Wavelength array from stellar spectrum
	-d: array. Array of normalized flux values
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-sig: float. Statistical noise estimation
	'''
	Delt = np.array([d[i] -0.5* d[i -2] -0.5 *d[i +2] for i in \
		range(len(w)) if (i < len(d) -2) and (i > 1)])
	bounds = [(0, 0), (0.0001, 0.1)]
	fit = stats.fit(stats.norm, Delt, bounds)
	sig = fit.params[1] *(0.5 **2 +0.5 **2 +1) **-0.5
	print('Fit parameters:\n'+str(fit.params))
	print('Standard deviation: ' +str(np.std(Delt)))
	print('Mean: ' +str(np.mean(Delt)))
	color1 = '#59C3C3'
	if plot:
		plt.figure(figsize = [8, 6], facecolor = 'white')
		ax1 = plt.axes()
		print(np.max(np.abs(Delt)))
		print(fit.params[1], fit.params[0])
		x_range = np.linspace(-7 *fit.params[1], 7 \
			*fit.params[1], 1000)
		plt.hist(Delt, density =True, color = color1, \
			edgecolor = 'black', alpha = 0.7, bins = \
			np.arange(-7 *fit.params[1], 7 \
			*fit.params[1], fit.params[1]), label = \
			'Histrogram')
		plt.plot(x_range, (1 /(fit.params[1] *np.sqrt(2 \
			*np.pi)) *np.exp(-0.5 *(x_range \
			-fit.params[0]) **2 /(fit.params[1]) **2)), \
			'--', color = 'black', label = 'Fitted'\
			' Normal\nDistribution')
		plt.xlim(x_range[0], x_range[999])
		plt.ylabel('PDF')
		plt.xlabel(r'$\Delta$')
		plt.title(sname +' noise distribution')
		param_bbox = dict(alpha = 0.7,
			facecolor = 'wheat',
			boxstyle = 'round')
		plt.text(0.88, 0.75, 'Fit-parameters\n'r'$\mu$: {0}'\
			'\n$\sigma$: {1}'.format(fit.params[0], \
			np.round(fit.params[1], 5)), bbox = \
			param_bbox, ha = 'center', fontsize = 12, \
			transform = ax1.transAxes, va = 'center')
		plt.legend()
		plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname \
			+'stat_noise.pdf')
	return sig

def residuals(w, d, sig, re, plot):
	'''
	---------------------------------------------
	Inputs
	-w: array. Wavelength array from stellar spectrum.
	-d: array. Array of normalized flux values.
	-sig: float. The statistical error estimated in the
	noise_estimation function.
	-re: array. Residual array from best-fit model spectum.
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-broad_sig: array. Estimation of the entire uncertainty
	array.
	'''
	sig_sys = [np.sqrt(r **2 -sig **2) if np.abs(r) > sig \
		else 0 for r in re]
	# NP Estimating unsmoothed systematic error
	smoothed_sig_sys = gaussian_filter(sig_sys, 3)
	# NP Smoothing systematic error
	sig_tot = np.sqrt(np.array(sig_sys) **2 +sig **2)
	# NP Adding statistical and systematic error in quadrature
	# NP to get the total error
	A = 0.68
	# NP Defining a constant
	broad_sig = A *gaussian_filter(sig_tot, 3)
	# NP Reducting error array by appromate factor to encapsulate
	# NP one sigma
	if plot:
		color2 = '#52489C'
		# NP Defining pretty color
		axiskwargs = dict(fontsize = 15,
		)
		# NP Defining axis keywork agruments
		plt.figure(figsize = [24, 6], facecolor = 'white')
		# NP Making new figure
		plt.plot(w, smoothed_sig_sys, color = 'black')
		plt.fill_between(w, 0, smoothed_sig_sys, color = \
			color2)
		plt.xlim(4000, 5000)
		max_value = np.max(smoothed_sig_sys[\
			np.logical_and(w > 4000, w < 5000)])
		plt.ylim(0, max_value *1.05)
		plt.ylabel(r'Systematic uncertainty', **axiskwargs)
		plt.xlabel(r'Wavelength $\lambda$ ($\AA$)', \
			**axiskwargs)
		plt.yticks(**axiskwargs)
		plt.xticks([4000, 4050, 4100, 4150, 4200, 4250, \
			4300, 4350, 4400, 4450, 4500, 4550, 4600, \
			4650, 4700, 4750, 4800, 4850, 4900, 4950, \
			5000], **axiskwargs)
		plt.title(sname +' systematic uncertainty '\
			'spectrum', **axiskwargs)
		plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname \
			+'sys_unc.pdf')
		plt.clf()
		color3 = '#DB5461'
		plt.figure(figsize = [24, 6], facecolor = 'white')
		ax1 = plt.axes()
		plt.plot(w, broad_sig, color = 'black')
		#plt.plot(w, -1 *broad_sig, color = 'black')
		plt.fill_between(w, -0 *broad_sig, broad_sig, \
			color = color3)
		err_max = np.max(broad_sig[np.logical_and(w < \
			5000, w > 4000)])
		plt.ylim(0, 1.05 *err_max)
		plt.xlim(4000, 5000)
		plt.title(sname +' uncertainty spectrum', \
			**axiskwargs)
		plt.ylabel(r'Predicted uncertainty', **axiskwargs)
		plt.xlabel(r'Wavelength $\lambda$ ($\AA$)', \
			**axiskwargs)
		plt.yticks(**axiskwargs)
		plt.xticks([4000, 4050, 4100, 4150, 4200, 4250, \
			4300, 4350, 4400, 4450, 4500, 4550, 4600, \
			4650, 4700, 4750, 4800, 4850, 4900, 4950, \
			5000], **axiskwargs)
		plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname \
			+'unc_spec.pdf')
	return broad_sig

def log_likelihood_T(theta, x, y, yerr):
	'''
	---------------------------------------------
	Inputs
	-T_targ: float. 
	Kelvin
	-g_targ: float. 
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-wav_new: array. Wavelength array in Angstroms.
	-spline(wav_new): float. Spline of interpolated spectrum.
	'''
	T, g, vsini = theta
	# NP Defining parameters
	if 15000 < T < 56000 and 2 < g < 4.75 and \
		10 < vsini < 600:
		try:
			spec = data
			# NP Finding spectrum
			wavs = wavl
			# NP Finding wavelengths
			wavs2, smodel = interpspectra(T, g, vsini,\
				False)
			# Interpolating to desired T, log g and vsini
			#resids = spec[np.logical_and(wavs > 3900, \
			#	wavs < 5000)] -smodel(\
			#	wavs[np.logical_and(wavs > 3900, \
			#	wavs < 5000)])	
			resids = spec -smodel(wavs)
			chi_sq = np.array([(resids[i] /sig_tot[i]) \
				**2 for i in range(len(resids))])
			if (APO):
				X = np.where(np.logical_and(wavs > \
					4000, wavs < 5000), \
					chi_sq, 0)
			else:
				X = np.where(np.logical_and(wavs > \
					4200, wavs < 5000), \
					chi_sq, 0)
			dof = len(X[X != 0]) -3
			X_sq = np.sum(X) /dof
			logprobtot = chi2.logpdf(np.sum(X), dof)
			s = open('/d/hya1/nikhil/BS/analysis/' +sname \
				+'.dat', 'a')
			# NP Opening data file
			datastr = '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\t' \
				'{3:5.5f}\t{4:5.5f}\n'\
				.format(T, g, vsini, logprobtot, \
				X_sq)
			# NP Writing out parameters of fit and
			# NP indicators of fit to file
			s.write(datastr)
			# NP Updating data file
			s.close()
			# NP Closing data file
			return logprobtot
		except Exception as e:
			print('Can\'t find probability!')
			print(e)
			return -np.inf
	else:
	    	print('Skipping loop')
	    	return -np.inf

def log_probability_T(theta, x, y, yerr):
	'''
	---------------------------------------------
	Inputs
	-T_targ: float. 
	Kelvin
	-g_targ: float. 
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-wav_new: array. Wavelength array in Angstroms.
	-spline(wav_new): float. Spline of interpolated spectrum.
	'''
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood_T(theta, x, y, yerr)

def log_prior(theta):
	'''
	---------------------------------------------
	Inputs
	-T_targ: float. 
	Kelvin
	-g_targ: float. 
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-wav_new: array. Wavelength array in Angstroms.
	-spline(wav_new): float. Spline of interpolated spectrum.
	'''
	T, g, vsini = theta
	if 15000 < T < 56000 and 2 < g < 4.75 and 10 < vsini < 600:
		return 0.0
	return -np.inf
	# NP Rejecting parameters outside of model space

def mcmc(t, g, v, runs = 3000):
	'''
	---------------------------------------------
	Inputs
	-T_targ: float. 
	Kelvin
	-g_targ: float. 
	-plot: Boolean. Whether to plot the interpolated spectrum.
	---------------------------------------------
	Outputs
	-wav_new: array. Wavelength array in Angstroms.
	-spline(wav_new): float. Spline of interpolated spectrum.
	'''
	if t == 15000:
		T_true = 15100
	elif t == 30000:
		T_true = 29900 
	else:
		T_true = t
	if g == 4.75:
		g_true = 4.65
	else:
		g_true = g
	if v == 10:
		vsini_true = 15
	else:
		vsini_true = v
	# NP Setting best-fit temperature, log g, vsini starting
	# NP point
	nll = lambda *args: -log_likelihood_T(*args)
	initial = np.array([T_true, g_true, vsini_true])\
		+0.1 * np.random.randn(3)
	testing = True
	soln = minimize(nll, initial, args=(cints[1][0:3], \
		cints[1][0:3], cints[1][0:3]))
	pos = soln.x + 1e-4 * np.random.randn(32, 3)
	nwalkers, ndim = pos.shape
	sampler = emcee.EnsembleSampler(nwalkers, ndim, \
		log_probability_T, args=(cints[1][0:3], \
		cints[1][0:3], cints[1][0:3]))
	sampler.run_mcmc(pos, runs, progress=True);
	# NP Running MCMC fitting on spectrum

	if runs == 5000:
		tau = sampler.get_autocorr_time()
		print('Autocorrelation time: {0}'.format(tau))
	flat_samples = sampler.get_chain(discard=100, thin=15,\
		flat=True)
	labels = ["T", "logg", "vsini"]
	txt = ""
	for i in range(ndim):
		mcmc = np.percentile(flat_samples[:, i], [\
			15.865525, 50, 84.134475])
		q = np.diff(mcmc)
		txt += "{3} = {0:.5f} -{1:.5f} +{2:.5f}\n"\
			.format(mcmc[1], q[0], q[1], labels[i])
	print(txt)
	s = open('/d/hya1/nikhil/BS/analysis/' +sname+'_params.dat', 'w')
	s.write(txt)
	s.close()
	# NP Printing and saving best-fit parameters and 
	# NP uncertainties
	#ds9 -geometry 800x1057 -wcs fk5 -wcs skyformat degrees  -rgb -red W4.fits -scale linear -scale limits 38 72 -green W3.fits -scale linear -scale limits 55 105 -blue W1.fits -scale linear -scale limits 0.5 12  -match frame wcs  -pan to 029.1883 0.5550 -wcs galactic -zoom 8.4  -grid load coordFK5.grd -grid yes -view panner yes  -view colorbar no &   

	Ts = np.percentile(flat_samples[:, 0], [15.865525, 50, \
		84.134475])
	# NP Getting  -1, 0 and 1 sigma values for temperature
	Tguess = Ts[1]
	# NP Getting most-probable temperature
	qT = np.diff(Ts)
	# NP Calculating deviations from most-probable temperature
	# NP for the -1 and 1 sigma levels
	gs = np.percentile(flat_samples[:, 1], [15.865525, 50, \
		84.134475])
	# NP Getting -1, 0 and 1 sigma values for log g
	gguess = gs[1]
	# NP Getting most-probable log g
	qg = np.diff(gs)
	# NP Calculating deviations from most-probable log g for the
	# NP -1 and 1 sigma levels
	vs = np.percentile(flat_samples[:, 2], [15.865525, 50, \
		84.134475])
	# NP Getting -1, 0 and 1 sigma values for vsini
	vguess = vs[1]
	# NP Getting most-probable vsini
	qv = np.diff(vs)
	# NP Calculating deviations from most-probable vsini for the
	# NP -1 and 1 sigma levels

	f = plt.figure(facecolor = 'white', figsize = [24, 12])
	# NP Creating figure for data spectrum and best fit model
	# NP plot and residual spectrum
	wmodel, model = interpspectra(Tguess, gguess, vguess, False)
	# NP Interpolating best fit parameters

	ax1 = plt.subplot(2, 1, 1)
	# NP Creating first subplot for data spectrum and best-fit
	# NP model spectrum
	plt.title('Stellar spectrum with best-fit interpolated\n'
		'model and residual spectrum', fontsize = 20)
	# NP Setting title
	plt.plot(wavl, data, color = 'black', label = 'Data')
	# NP Plotting data spectrum
	plt.plot(wavl[wavl > 3800], model(wavl[wavl > 3800]), \
		'--', color = 'darkred', label = 'Best-fit model')
	# NP Plotting best-fit model spectrum
	plt.vlines(x=4009, color = 'k', linewidth = 1, ymax = 1.03, \
		ymin = 1.01)
	plt.text(4009, 1.038, r'He I 4009', rotation = 90, \
		ha = 'center')
	# NP Labeling He I 4009 line
	plt.vlines(x=4026, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4026, 1.038, r'He I+II 4026', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4058, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4058, 1.038, r'N IV', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4068, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4069, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4070, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4068, xmax = 4070, color = 'k'\
		, linewidth = 1)
	plt.text(4069, 1.038, r'C III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4070, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4072, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4076, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4070, xmax = 4076, color = 'k'\
		, linewidth = 1)
	plt.text(4073, 1.038, r'O II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4089, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4089, 1.038, r'Si IV', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4101, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4101, 1.038, r'H$\delta$', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4116, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4116, 1.038, r'Si IV', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4121, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4121, 1.038, r'He I 4121', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4128, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4130, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4128, xmax = 4130, color = 'k'\
		, linewidth = 1)
	plt.text(4129, 1.038, r'Si II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4144, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4144, 1.038, r'He I 4144', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4200, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4200, 1.038, r'He II 4200', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4267, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4267, 1.038, r'C II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4326, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4326, 1.038, r'C III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4340, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4340, 1.038, r'H$\gamma$', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4349, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4349, 1.038, r'O II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4379, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4379, 1.038, r'N III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4388, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4388, 1.038, r'He I 4388', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4415, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4417, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4415, xmax = 4417, color = 'k'\
		, linewidth = 1)
	plt.text(4416, 1.038, r'O II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4420, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4436, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4420, xmax = 4436, color = 'k'\
		, linewidth = 1)
	plt.text(4428, 1.038, r'DIB', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4471, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4471, 1.038, r'He I 4471', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4481, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4481, 1.038, r'Mg II 4481', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4511, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4515, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4511, xmax = 4515, color = 'k'\
		, linewidth = 1)
	plt.text(4513, 1.038, r'N III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4541, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4541, 1.038, r'He II 4541', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4552, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4568, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4575, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4552, xmax = 4575, color = 'k'\
		, linewidth = 1)
	plt.text(4563.5, 1.038, r'Si III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4604, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4620, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4604, xmax = 4620, color = 'k'\
		, linewidth = 1)
	plt.text(4612, 1.038, r'N V', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4634, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4640, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4642, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4634, xmax = 4642, color = 'k'\
		, linewidth = 1)
	plt.text(4638, 1.038, r'N III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4640, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4650, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4640, xmax = 4650, color = 'k'\
		, linewidth = 1)
	plt.text(4645, 1.038, r'O II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4631, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4631, 1.038, r'N II', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4647, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4652, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4647, xmax = 4652, color = 'k'\
		, linewidth = 1)
	plt.text(4649.5, 1.038, r'C III', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4654, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4654, 1.038, r'Si IV', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4658, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4658, 1.038, r'C IV', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4686, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4686, 1.038, r'He II 4686', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4713, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4713, 1.038, r'He I 4713', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4762, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4765, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4762, xmax = 4765, color = 'k'\
		, linewidth = 1)
	plt.text(4763.5, 1.038, r'DIB', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4861, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4861, 1.038, r'H$\beta$', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4880, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.vlines(x=4887, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.hlines(y = 1.03, xmin = 4880, xmax = 4887, color = 'k'\
		, linewidth = 1)
	plt.text(4883.5, 1.038, r'DIB', rotation = 90, \
		ha = 'center')
	plt.vlines(x=4922, color= 'k', linewidth = 1, ymax = 1.03\
		, ymin = 1.01)
	plt.text(4922, 1.038, r'He I 5922', rotation = 90, \
		ha = 'center')
	legend_font = {'size': 15,}
	param_bbox = dict(alpha = 0.7,
		facecolor = 'wheat',
		boxstyle = 'round')
	param_string = r'$T = {0}^{{+{1}}}_{{-{2}}}$ K$'+'\n'+r'\log g '\
		r'= {3}^{{+{4}}}_{{-{5}}}$' +'\n' +r'$v\sin{{i}} = '\
		r'{6}^{{+{7}}}_{{-{8}}}$ km s$^{{-1}}$'.format(\
		int(np.round(Tguess)), int(np.round(qT[1])), \
		int(np.round(qT[0])), np.round(gguess, 2), \
		np.round(qg[1], 2), np.round(qg[0], 2), int(\
		np.round(vguess)), int(np.round(qv[1])), \
		int(np.round(qv[0])))
	plt.text(0.6, 0.15, r'$T = {0}^{{+{1}}}_{{-{2}}}$ K, $ \log g'\
		r'= {3}^{{+{4}}}_{{-{5}}}$, $v\sin{{i}} = '\
		r'{6}^{{+{7}}}_{{-{8}}}$ km s$^{{-1}}$'.format(\
		int(np.round(Tguess)), int(np.round(qT[1])), \
		int(np.round(qT[0])), np.round(gguess, 2), \
		np.round(qg[1], 2), np.round(qg[0], 2), int(\
		np.round(vguess)), int(np.round(qv[1])), \
		int(np.round(qv[0]))), bbox = param_bbox, ha = \
		'center', fontsize = 20, transform = \
		ax1.transAxes, va = 'center')
	plt.legend(loc = 'upper center', prop = \
		legend_font, ncol = 2, \
		fancybox = True, shadow = True, \
		framealpha = 1.0)
	minimum = np.min(data[np.logical_and(wavl > 4000, wavl < 5000)])
	axiskwargs = dict(fontsize = 15,
		)
	plt.yticks(**axiskwargs)
	plt.xlim(4000, 5000)
	plt.ylim(minimum -0.03, 1.16)
	plt.ylabel('Normalized spectrum', fontsize = 20)
	plt.setp(ax1.get_xticklabels(), visible=False)

	ax2 = plt.subplot(2, 1, 2, sharex = ax1)
	resids = data -model(wavl)
	plt.fill_between(wavl, -1 *sig_tot, sig_tot, \
		color = 'darkorange', alpha = 0.6, label = r'1 $\sigma$')
	plt.fill_between(wavl, -2 *sig_tot, 2 \
		*sig_tot, color = 'darkorange', alpha = 0.4, label = \
		r'2 $\sigma$')
	plt.fill_between(wavl, -3 *sig_tot, 3 \
		*sig_tot, color = 'darkorange', alpha = 0.2, label = \
		r'3 $\sigma$')
	plt.plot(wavl, resids, linewidth = 0.7, color = \
		'black')
	plt.ylabel('Residuals', fontsize = 20)
	plt.xlabel(r'Wavelength $\lambda$ $(\AA)$', fontsize = 20)
	plt.xlim(4000, 5000)
	max_err = np.max(sig_tot[np.where(np.logical_and(wavl > \
		4000, wavl < 5000))])
	print('max. error: ' +str(max_err))
	plt.ylim(-3 *max_err -0.02, 3.0 *max_err +0.02)
	plt.legend(loc = 'upper right', prop = \
		legend_font, ncol = 1, \
		fancybox = True, shadow = True, \
		framealpha = 1.0)
	plt.yticks(**axiskwargs)
	plt.xticks([4000, 4050, 4100, 4150, 4200, 4250, 4300, \
		4350, 4400, 4450, 4500, 4550, 4600, 4650, 4700, \
		4750, 4800, 4850, 4900, 4950, 5000], **axiskwargs)
	plt.tight_layout()
	plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname \
		+'emceefittedparams.png', bbox_inches ='tight')
	plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname \
		+'emceefittedparams.pdf', bbox_inches ='tight')
	# NP Plotting best-fit parameters

	flat_samples = sampler.get_chain(discard=100, thin = 15,\
		flat=True)
	CORNER_KWARGS = dict(
		label_kwargs = dict(fontsize=16),
		smooth = 1,
		quantiles=[0.15865525, 0.5, 0.84134475],
		levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 \
			-np.exp(-9 / 2.)),
		plot_density=True,
		plot_datapoints=True,
		fill_contours=False,
		show_titles=True,
		max_n_ticks = 6,
		labels = ('T', 'log g', 'vsini'),
		range = [0.999, 0.999, 0.999],
	)
	fig = corner.corner(flat_samples, **CORNER_KWARGS)
	plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname +'corners.png')
	plt.savefig('/d/hya1/nikhil/BS/analysis/' +sname +'corners.pdf')
	
def plot_grid(t, g, n):
	plt.figure(figsize = [8, 6], facecolor = 'white')
	plt.plot(t/1000, g, 'ok')
	plt.title(n +' parameters')
	plt.xlabel(r'Temperature ($kK$)')
	plt.ylabel(r'$\log g$ (cgs)')
	plt.yticks(np.arange(np.min(g), np.max(g) +0.2, 0.2))
	t_range = np.arange(np.floor(np.min(t /1000)), np.max(t \
		/1000) +1, 1)
	tlabels = [str(t) if t % 5 == 0 else "" for t in t_range]
	plt.xticks(t_range, tlabels)
	plt.savefig('/d/hya1/nikhil/BS/model_spectra' +n +'.pdf')

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Program to\
		run MCMC fitting on spectra to find temperature, \
		log g  and vsini with uncertainties.')
	# NP Adding description of program
	parser.add_argument('spec', type = str, help = 'Directory'
		' of the normalized reduced spectrum. Example:'
		' /d/car1/.../BS013.fits')
	# NP Adding description of arguments
	args = parser.parse_args()
	# NP Adding parsers
	print('Reading in spectrum...\n')
	hdu = fits.open(args.spec)
	# NP Opening spectrum
	data= hdu[0].data
	# NP Reading in data
	hdr = hdu[0].header
	# NP Reading in header
	crval = hdr['CRVAL1']
	# NP Finding starting wavelength
	cdelt = hdr['CD1_1']
	# NP Finding wavelength scale in A per pix
	crpix = hdr['CRPIX1']
	# NP Finding starting pixel
	wavl = crval +cdelt *(np.arange(0,len(data)))
	# NP Creating wavelength array from header
	goodindex = args.spec.rfind('/')
	# NP Guessing a name by looking for the last '/' in the input
	# NP file name
	sname = hdr['OBJNAME']
	# NP Choosing a name by looking at the header
	print('Name of object: ' +sname +'\n')
	# NP Printing program's guess at the name
	global APO 
	APO = hdr['OBSERVAT'] == 'APO'
	# NP Defining a global boolean indicating whether this is an
	# NP APO KOSMOS spectrum
	print('Determining which models to use.\n')
	global ctemps, cgs, vsini, cwavls, cints
	#btemps, bgs, bvsini, bwavls, bints = get_Bstars(APO)
	otemps, ogs, ovsini, owavls, oints = get_Ostars(APO)
	#ptemps, pgs, pvsini, pwavls, pints = get_POWR(APO)
	#plot_grid(btemps, bgs, 'BSTAR2006')
	#plot_grid(otemps, ogs, 'OSTAR2002')
	#plot_grid(ptemps, pgs, 'POWR')
	#ctemps, cgs, vsini, cwavls, cints = guessmodels(btemps, \
#		bgs, bvsini, bwavls, bints, otemps, ogs, ovsini, \
#		owavls, oints, wavl, data)
	ctemps, cgs, vsini, cwavls, cints = otemps, ogs, ovsini, \
		owavls, oints
	#ctemps, cgs, vsini, cwavls, cints = btemps, bgs, bvsini, \
	#	bwavls, bints
	#ctemps, cgs, vsini, cwavls, cints = ptemps, pgs, pvsini, \
	#	pwavls, pints
	# NP Defining global variables for models used throughout the
	# NP program
	print('Estimating best fit parameters.\n')
	bestT, bestg, bestv, resids = guess(wavl, data)
	# NP Getting best-fit paramaters
	sig = noise_estimation(wavl, data, True)
	print('Statistical sigma estimation: ' +str(sig))
	global sig_tot
	sig_tot = residuals(wavl, data, sig, resids, True)
	print('Total sigma estimation: ' +str(sig_tot))
	if(~np.isnan(bestT +bestg +bestv)):
		s = open('/d/hya1/nikhil/BS/analysis/' +sname \
			+'.dat', 'w')
		# NP Creating data file
		s.write(str(sname) +" params:\n")
		# NP Writing initial paramter guesses to file
		s.close()
		# NP Closing file
		mcmc(bestT, bestg, bestv)
		# NP Running MCMC on the desired spectrum
		chis = np.loadtxt('/d/hya1/nikhil/BS/analysis/'+sname \
			+'.dat', usecols = [4], skiprows =1)
		# NP Reading in chis from data file
		bestchi = chis[np.argmin(np.abs(chis))]
		# NP Reading best reduced chi-squared
		print('best chi2: ' +str(bestchi) +'\n')
		sig_tot = np.sqrt(np.round(bestchi, 3)) *sig_tot
		mcmc(bestT, bestg, bestv, 6000)
		newchis = np.loadtxt('/d/hya1/nikhil/BS/analysis/' +sname \
			+'.dat', usecols = [4], skiprows = 1)
		newbestchi = newchis[np.argmin(np.abs(newchis))]
		print('new best chi2: ' +str(newbestchi) +'\n')
		print('Done!\n')




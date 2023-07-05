import os
import emcee
import corner
import random
import argparse
import traceback
import numpy as np
from astropy.io import fits
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
# NP Necessary imports

def getmodels(APO):
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
		cdir = '/d/hya1/BS/model_spectra/conv/'
		# NP Directory for convolved spectra
		cnames = np.array(os.listdir(cdir))
		# NP Convolved model names
		cifiles = [n[-4:] == '.txt' for n in cnames]
		# NP Limiting to only text files
		cmodel = [np.loadtxt(cdir +i, usecols = (0, 1)) for i in \
			cnames[cifiles]]
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
		cwavls = [i.T[0] for i in cmodel]
		# NP Reading in vacuum wavelengths
		cints = [i.T[1] for i in cmodel]
		# NP Reading in convolved model intensities
	else:
		print('Reading in WIRO model spectra.\n')
		cdir = '/d/hya1/BS/model_spectra/wiro_conv/'
		# NP Directory for convolved spectra
		cnames = np.array(os.listdir(cdir))
		# NP Convolved model names
		cifiles = [n[-4:] == '.txt' for n in cnames]
		# NP Limiting to only text files
		cmodel = [np.loadtxt(cdir +i, usecols = (0, 1)) for i in \
			cnames[cifiles]]
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
		cwavls = [i.T[0] for i in cmodel]
		# NP Reading in vacuum wavelengths
		cints = [i.T[1] for i in cmodel]
		# NP Reading in convolved model intensities
	return ctemps, cgs, vsini, cwavls, cints
	print('Done!\n')

def guess(wav, dat):
	'''Guesses the parameters T, vsini and log g for a Bowshock
	star.
	---------------------------------------------
	Inputs
	-spec: str. BS identifier for the star. Example: 'BS013'.
	---------------------------------------------
	Outputs
	-t_test: float. Best fit parameter for temperaure in Kelvin.
	-g_test: float. Best fit parameter for log g.
	-v_test: float. Best fit parameter for vsini in km/s.
	'''
	spectrumspline = CubicSpline(wav, dat)
	# NP Creating a spline of the spectrum
	testrange = np.linspace(4000, 4990, 1000)
	# NP Wavelength range to evaluate differences
	# NP between models and spectra
	modelsplines = [CubicSpline(cwavls[a], \
		cints[a]) for a in range(len(cints))]
	# NP Creating a spline of all model spectra
	diffs = [np.sum(np.abs(d(testrange) \
		-spectrumspline(testrange)) **2) \
		for d in modelsplines]
	# NP Evaluating rough chi squared for each model
	smin = np.argmin(diffs)
	# NP Finding minimum chi squared
	print('T_eff: ' +str(ctemps[smin]))
	print('log g: ' +str(cgs[smin]))
	print('v sini: ' +str(vsini[smin]) +'\n')
	# NP Printing best-fit paramteres
	t_test = ctemps[smin]
	g_test = cgs[smin]
	v_test = vsini[smin]
	return t_test, g_test, v_test
	# NP Returning best-fit parameters

def interpspectra(T_targ, g_targ, v_targ, plot):
	'''Interpolates between different temperature at and different
	log g model spectra. Uses a bilinear interpolation between 
	temperatures and log g to interpolate between four model
	spectra.
	---------------------------------------------
	Inputs
	-T_targ: float. Target temperature to interpolate to in Kelvin.
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
		T_1 = T_targ +tempdiffs[tempdiffs < 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs < 0]))]
		# NP Finding lower temperature
		T_2 = T_targ +tempdiffs[tempdiffs > 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs > 0]))]
		# NP Finding upper temperature
		g_1 = g_targ +gdiffs[(ctemps == T_1) & (gdiffs < 0)]\
			[np.argmin(np.abs(gdiffs[(ctemps == T_1) \
			& (gdiffs < 0)]))]
		# NP Finding lower gravity
		g_2 = g_targ +gdiffs[(ctemps == T_2) & (gdiffs > 0)]\
			[np.argmin(np.abs(gdiffs[(ctemps == T_2) \
			& (gdiffs > 0)]))]
		# NP Finding upper gravity
		v_1 = v_targ +vdiffs[vdiffs < 0][np.argmin(\
			np.abs(vdiffs[vdiffs < 0]))]
		# NP Finding lower gravity
		v_2 = v_targ +vdiffs[vdiffs > 0][np.argmin(\
			np.abs(vdiffs[vdiffs > 0]))]
		# NP Finding upper gravity
		wavs1 = np.array(cwavls, dtype=object)[(cgs == g_1) \
			& (ctemps == T_1) & (vsini == v_1)][0]
		# NP Finding the lower convolution wavelengths
		wavs2 = np.array(cwavls, dtype=object)[(cgs == g_1) \
			& (ctemps == T_1) & (vsini == v_2)][0]
		# NP Finding the higher convulation wavelengths
		ints111 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_1) & (vsini == v_1)][0]
		# NP Finding the model with low temperature low log g
		# NP and low vsini
		ints121 = np.array(cints, dtype=object)[(cgs == g_2) \
			& (ctemps == T_1) & (vsini == v_1)][0]
		# NP Finding the model with low temperature high log g 
		# NP and low vsini
		ints211 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_2) & (vsini == v_1)][0]
		# NP Finding the model with high temperature low log g 
		# NP and low vsini
		ints221 = np.array(cints, dtype=object)[(cgs == g_2) \
			& (ctemps == T_2) & (vsini == v_1)][0]
		# NP Finding the model with high temperature high log g 
		# NP and low vsini
		ints112 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_1) & (vsini == v_2)][0]
		# NP Finding the model with low temperature low log g
		# NP and high vsini
		ints122 = np.array(cints, dtype=object)[(cgs == g_2) \
			& (ctemps == T_1) & (vsini == v_2)][0]
		# NP Finding the model with low temperature high log g 
		# NP and high vsini
		ints212 = np.array(cints, dtype=object)[(cgs == g_1) \
			& (ctemps == T_2) & (vsini == v_2)][0]
		# NP Finding the model with high temperature low log g
		# NP and high vsini
		ints222 = np.array(cints, dtype=object)[(cgs == g_2) \
			& (ctemps == T_2) & (vsini == v_2)][0]
		# NP Finding the model with high temperature high log 
		# NP g and high vsini
		T_d = (T_targ -T_1) /(T_2 -T_1)
		# NP Finding percent change in desired temperature
		g_d = (g_targ -g_1) /(g_2 -g_1)
		# NP Finding percent change in desired gravity
		v_d = (v_targ -v_1) /(v_2 -v_1)
		# NP Finding percent change in rotational velocity
		c00 = ints211 *(1 -g_d) +ints221 *g_d
		# NP Finding bilinear interpolation between log g with
		# NP low vsini and high T
		c01 = ints212 *(1 -g_d) +ints222 *g_d
		# NP Finding bilinear interpolation between log g with
		# NP high vsini and high T
		c10 = ints111 *(1 -g_d) +ints121 *g_d
		# NP Finding bilinear interpolation between log g with
		# NP low vsini and low T
		c11 = ints112 *(1 -g_d) +ints122 *g_d
		# NP Finding bilinear interpolation between log g with
		# NP high vsini and low T
		c0 = c00 *(1 -T_d) +c10 *T_d
		# NP Finding bilinear interpolation between T with
		# NP interpolated log g and low
		# NP vsini
		c1 = c01 *(1 -T_d) +c11 *T_d
		# NP Finding bilinear interpolation between T with
		# NP interpolated log g and high vsini
		wav_new = np.linspace(3700, 5800, 5000)
		# NP Creating wavelength array to evalueate splines 
		# NP over
		spl_zero = CubicSpline(wavs1, c0)
		spl_one = CubicSpline(wavs2, c1)
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
		# NP visualization
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
			plt.plot(wav_new, spline(wav_new), label ='c')
			# NP Plotting c interpolation
			plt.plot(wavl[index], data[index])
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
	except TypeError:
		print('Could not interpolate these parameters!\n')
		print(T_targ, g_targ, v_targ)
		# NP Print this string if temperature could not be
		# NP interpolated
	except ValueError:
		print('Could not interpolate these parameters!\n')
		print(T_targ, g_targ, v_targ)
		# NP Print this string if temperature could not be
		# NP interpolated

def log_likelihood_T(theta, x, y, yerr):
	T, g, vsini = theta
	# NP Defining parameters
	try:
		if 15000 < T < 55000 and 2 < g < 5 and 10 < vsini < 600:
			spec = data
			# NP Finding spectrum
			wavs = wavl
			# NP Finding wavelengths
			specspline = CubicSpline(wavs, spec)
			# NP Creating cubic spline of spectrum
			mask1 = wavs > 4030
			# NP Limiting blue continuum to greater than
			# NP 4256 Angstroms
			mask2 = wavs < 4060
			# NP Limiting blue continuum to less than 4264
			# NP Angstroms
			mask3 = np.logical_and(mask1, mask2)
			# NP Combining limitations
			bluewavs = spec[mask3]
			# NP Defining blue continuum
			mask1 = wavs > 4945
			# NP Limiting red continuum to greater than
			# NP 4945 Angstorms
			mask2 = wavs < 4954
			# NP Limiting red continuum to less than 4954
			# NP Angstroms
			mask3 = np.logical_and(mask1, mask2)
			# NP Combining limitations
			redwavs = spec[mask3]
			# NP Definining red continuum
			wavs2, smodel = interpspectra(T, g, vsini,\
				False)
			# Interpolating to desired T and log g

			bsigma = args.chi *np.std(bluewavs) **2
			# NP Estimating uncertainty from standard
			# NP deviation in blue continuum

			if APO:
				N1, chi_1 = line_evaluate(smodel, \
					specspline, wavs, 4009, bsigma)		
				# NP Comparing He I 4009

				N2, chi_2 = line_evaluate(smodel, \
					specspline, wavs, 4026, bsigma)			
				# NP Comparing He I+II 4026

				N3, chi_3 = line_evaluate(smodel, \
					specspline, wavs, 4089, bsigma)			
				# NP Comparing Si IV 4089

				N4, chi_4 = line_evaluate(smodel, \
					specspline, wavs, 4101, bsigma)			
				# NP Comparing H delta 4101

				N5, chi_5 = line_evaluate(smodel, \
					specspline, wavs, 4116, bsigma)			
				# NP Comparing Si IV 4116

				N6, chi_6 = line_evaluate(smodel, \
					specspline, wavs, 4121, bsigma)			
				# NP Comparing He I 4121

				N7, chi_7 = line_evaluate(smodel, \
					specspline, wavs, 4144, bsigma)			
				# NP Comparing He I 4121

				N8, chi_8 = line_evaluate(smodel, \
					specspline, wavs, 4200, bsigma)			
				# NP Comparing He II 4200

				N9, chi_9 = line_evaluate(smodel, \
					specspline, wavs, 4340, bsigma)			
				# NP Comparing H gamma 4340

				N10, chi_10 = line_evaluate(smodel, \
					specspline, wavs, 4387, bsigma)	
				# NP Comparing He I 4387

				N11, chi_11 = line_evaluate(smodel, \
					specspline, wavs, 4471, bsigma)			
				# NP Comparing He I 4471

				N12, chi_12 = line_evaluate(smodel, \
					specspline, wavs, 4481, bsigma)			
				# NP Comparing Mg II 4481

				N13, chi_13 = line_evaluate(smodel, \
					specspline, wavs, 4541, bsigma)			
				# NP Comparing He II 4541

				N14, chi_14 = line_evaluate(smodel, \
					specspline, wavs, 4552, bsigma)			
				# NP Comparing Si IV 4552

				chi_total = chi_1 +chi_2 +chi_3 +chi_4 \
					+chi_5 +chi_6 +chi_7 +chi_8 \
					+chi_9 +chi_10 +chi_11 \
					+chi_12 +chi_13 +chi_14
				# NP Summing chi squares of all lines
				Ntot = N1 +N2 +N3 +N4 +N5 +N6 +N7 +N8 \
					+N9 +N10 +N11 +N12 +N13 +N14 -3
				# NP Summing 
				logprobtot = chi2.logpdf(chi_total, Ntot)
			else:
				N1, chi_1 = line_evaluate(smodel, \
					specspline, wavs, 4200, bsigma)			
				# NP Comparing He II 4200

				N2, chi_2 = line_evaluate(smodel, \
					specspline, wavs, 4340, bsigma)			
				# NP Comparing H gamma

				N3, chi_3 = line_evaluate(smodel, \
					specspline, wavs, 4387, bsigma)			
				# NP Comparing He I 4387

				N4, chi_4 = line_evaluate(smodel, \
					specspline, wavs, 4471, bsigma)			
				# NP Comparing He I 4471

				N5, chi_5 = line_evaluate(smodel, \
					specspline, wavs, 4481, bsigma)			
				# NP Comparing Mg II 4481

				N6, chi_6 = line_evaluate(smodel, \
					specspline, wavs, 4541, bsigma)			
				# NP Comparing He II 4541

				N7, chi_7 = line_evaluate(smodel, \
					specspline, wavs, 4552, bsigma)			
				# NP Comparing Si III 4552

				N8, chi_8 = line_evaluate(smodel, \
					specspline, wavs, 4568, bsigma)			
				# NP Comparing Si III 4568

				N9, chi_9 = line_evaluate(smodel, \
					specspline, wavs, 4575, bsigma)			
				# NP Comparing Si III 4575

				N10, chi_10 = line_evaluate(smodel, \
					specspline, wavs, 4686, bsigma)			
				# NP Comparing He II 4686

				N11, chi_11 = line_evaluate(smodel, \
					specspline, wavs, 4713, bsigma)			
				# NP Comparing He I 4713

				N12, chi_12 = line_evaluate(smodel, \
					specspline, wavs, 4813, bsigma)			
				# NP Comparing Si III 4813

				N13, chi_13 = line_evaluate(smodel, \
					specspline, wavs, 4820, bsigma)			
				# NP Comparing Si III 4820

				N14, chi_14 = line_evaluate(smodel, \
					specspline, wavs, 4829, bsigma)			
				# NP Comparing Si III 4829

				N15, chi_15 = line_evaluate(smodel, \
					specspline, wavs, 4861, bsigma)			
				# NP Comparing H Beta

				N16, chi_16 = line_evaluate(smodel, \
					specspline, wavs, 4922, bsigma)			
				# NP Comparing He I 4922

				chi_total = chi_1 +chi_2 +chi_3 \
					+chi_4 +chi_5 +chi_6 +chi_7 \
					+chi_8 +chi_9 +chi_10 +chi_11\
					+chi_12 +chi_13 +chi_14 \
					+chi_15 +chi_16
				# NP Summing chi squares of all lines
				Ntot = N1 +N2 +N3 +N4 +N5 +N6 +N7 +N8 \
					+N9 +N10 +N11 +N12 +N13 +N14 \
					+N15 +N16 -3
				# NP Summing 
				logprobtot = chi2.logpdf(chi_total, \
					Ntot)
			s = open('./' +sname+'.dat', 'a')
			datastr = '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\t' \
				'{3:5.5f}\t{4:5.5f}\t{5:5.5f}\n'\
				.format(T, g, vsini, logprobtot, \
				bsigma**0.5, (chi_total) /Ntot)
			# NP Writing out parameters of fit and
			# NP indicators of fit to file
			s.write(datastr)
			s.close()
			return logprobtot
		else:
		    	print('Skipping loop')
		    	return -np.inf
	except:
		traceback.print_exc()
		return -np.inf

def line_evaluate(model, spec, w, lmbda, sigma):
	mask1 = w < lmbda +4
	mask2 = w > lmbda -4
	mask3 = np.logical_and(mask1, mask2)
	# NP Searching within 4 angstroms left and right of identified
	# NP feature
	wavelengths = w[mask3]
	chi2 = np.sum((model(wavelengths) -spec(wavelengths)) \
		**2 /sigma)
	# NP Evaluating chi squared for the line profile
	return len(wavelengths), chi2

def spec_evaluate(model, spec, w, sigma):
	mask1 = w < 4800
	mask2 = w > 4000
	mask3 = np.logical_and(mask1, mask2)
	wavelengths = w[mask3]
	chi2 = np.sum((model(wavelengths) -spec(wavelengths)) \
		**2 /sigma)
	# NP Evaluating chi squared for the entire spectrum, 4000-6000
	# NP Angstroms
	return len(wavelengths), chi2

def log_probability_T(theta, x, y, yerr):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood_T(theta, x, y, yerr)

def log_prior(theta):
	T, g, vsini = theta
	if 15000 < T < 55000 and 2 < g < 5 and 10 < vsini < 600:
		return 0.0
	return -np.inf
	# NP Rejecting parameters outside of model space

def mcmc(t, g, v):
	T_true = t
	g_true = g
	vsini_true = v
	# NP Setting best-fit temperature, log g, vsini and radial
	# NP velocity presets.
	nll = lambda *args: -log_likelihood_T(*args)
	initial = np.array([T_true, g_true, vsini_true])\
		+0.1 * np.random.randn(3)
	soln = minimize(nll, initial, args=(cints[1][0:3], cints[1][0:3]\
		,cints[1][0:3]))
	pos = soln.x + 1e-4 * np.random.randn(32, 3)
	nwalkers, ndim = pos.shape
	sampler = emcee.EnsembleSampler(nwalkers, ndim, \
		log_probability_T, args=(cints[1][0:3], cints[1][0:3], \
		cints[1][0:3]))
	if args.chi == 1:
		sampler.run_mcmc(pos, 1000, progress=True);
	else:
		sampler.run_mcmc(pos, 5000, progress=True);
	# NP Running MCMC fitting on spectrum

	flat_samples = sampler.get_chain(discard=200, thin=15,\
		flat=True)
	labels = ["T", "logg", "vsini"]
	txt = ""
	for i in range(ndim):
		mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
		q = np.diff(mcmc)
		txt += "{{{3}}} = {0:.5f} -{1:.5f} +{2:.5f}\n"\
			.format(mcmc[1], q[0], q[1], labels[i])
	print(txt)
	s = open('./' +sname+'_params.dat', 'w')
	s.write(txt)
	s.close()
	# NP Printing and saving best-fit parameters and uncertainties

	Tguess = np.percentile(flat_samples[:, 0], [50])
	gguess = np.percentile(flat_samples[:, 1], [50])
	vguess = np.percentile(flat_samples[:, 2], [50])

	f = plt.figure(facecolor = 'white', figsize = [24, 6])
	wmodel, model = interpspectra(Tguess, gguess, vguess, False)
	colors = ['red', 'orange', 'green', 'blue', 'purple']
	cindex = int(len(colors) *random.random())

	plt.plot(wavl, data, color = \
		colors[cindex], label = sname)
	plt.plot(wmodel, model(wmodel), '--k', label = \
		'Best fit params')
	plt.axvline(x=4009, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4009, 1.035, r'He I')
	plt.axvline(x=4026, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4026, 1.035, r'He I+II')
	plt.axvline(x=4058, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4058, 1.035, r'N IV')
	plt.axvline(x=4069, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4071, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4069, xmax = 4071, color = 'k'\
		, linewidth = 1)
	plt.text(4072, 1.035, r'C III')
	plt.axvline(x=4089, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4089, 1.035, r'Si IV')
	plt.axvline(x=4101, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4101, 1.035, r'H$\delta$')
	plt.axvline(x=4116, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4116-6, 1.035, r'Si IV')
	plt.axvline(x=4121, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4121, 1.035, r'He I')
	plt.axvline(x=4128, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4130, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4128, xmax = 4130, color = 'k'\
		, linewidth = 1)
	plt.text(4129+2, 1.035, r'Si II')
	plt.axvline(x=4144, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4144, 1.035, r'He I')
	plt.axvline(x=4200, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4188, 1.035, r'He II')
	plt.xlabel(r'$\AA$')
	plt.axvline(x=4340, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4340, 1.035, r'H$\gamma$')
	plt.axvline(x=4350, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4350, 1.035, r'O II')
	plt.axvline(x=4379, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4375, 1.035, r'N III')
	plt.axvline(x=4387, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4387, 1.035, r'He I')
	plt.axvline(x=4415, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4417, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4415, xmax = 4417, color = 'k'\
		, linewidth = 1)
	plt.text(4412, 1.035, r'O II')
	plt.axvline(x=4420, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4440, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4420, xmax = 4440, color = 'k'\
		, linewidth = 1)
	plt.text(4422, 1.035, r'IS band')
	plt.axvline(x=4471, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4471-10, 1.035, r'He I')
	plt.axvline(x=4481, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4481-10, 1.035, r'Mg II')
	plt.axvline(x=4486, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4504, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4486, xmax = 4504, color = 'k'\
		, linewidth = 1)
	plt.text(4490, 1.035, r'Si IV')
	plt.xlabel(r'$\AA$')
	plt.axvline(x=4511, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4515, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4511, xmax = 4515, color = 'k'\
		, linewidth = 1)
	plt.text(4506, 1.035, r'N III')
	plt.axvline(x=4541, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4541-12, 1.035, r'He II')
	plt.axvline(x=4552, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4568, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4575, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4552, xmax = 4575, color = 'k'\
		, linewidth = 1)
	plt.text(4555, 1.035, r'Si III')
	plt.axvline(x=4604, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4620, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4604, xmax = 4620, color = 'k'\
		, linewidth = 1)
	plt.text(4602, 1.035, r'N V')
	plt.axvline(x=4634, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4640, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4642, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4634, xmax = 4642, color = 'k'\
		, linewidth = 1)
	plt.text(4632, 1.035, r'N III')
	plt.axvline(x=4640, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.axvline(x=4650, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.hlines(y = 1.03, xmin = 4640, xmax = 4650, color = 'k'\
		, linewidth = 1)
	plt.text(4648, 1.035, r'O II')
	plt.axvline(x=4631, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4631-14, 1.035, r'N II')
	plt.axvline(x=4647, color= 'k', linewidth = 1, ymax = 0.60\
		, ymin = 0.55)
	plt.axvline(x=4652, color= 'k', linewidth = 1, ymax = 0.60\
		, ymin = 0.55)
	plt.hlines(y = 0.87, xmin = 4647, xmax = 4652, color = 'k'\
		, linewidth = 1)
	plt.text(4630, 0.87, r'C III')
	plt.axvline(x=4654, color= 'k', linewidth = 1, ymax = 0.60\
		, ymin = 0.55)
	plt.text(4656, 0.87, r'Si IV')
	plt.axvline(x=4658, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4662, 1.035, r'C IV')
	plt.axvline(x=4686, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4686, 1.035, r'He II')
	plt.axvline(x=4713, color= 'k', linewidth = 1, ymax = 0.95\
		, ymin = 0.90)
	plt.text(4713, 1.035, r'He I')
	plt.xlim(4000, 4900)
	plt.ylim(0.65, 1.05)
	plt.xlabel(r'$\AA$')
	plt.savefig('./' +sname +'emceefittedparams.png', \
		bbox_inches ='tight')
	plt.show()
	# NP Plotting best-fit parameters

	flat_samples = sampler.get_chain(discard=600, thin=15,\
		flat=True)
	fig = corner.corner(flat_samples, labels = labels)
	fig.set_facecolor('white')
	fig.show()
	plt.savefig('./' +sname +'fullcorners.png')
	plt.show()

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Program to\
		run MCMC fitting on spectra to find temperature, \
		log g  and vsini with uncertainties.')
	# NP Adding description of program
	parser.add_argument('spec', type = str, help = 'Directory'
		' of the normalized reduced spectrum. Example:'
		' /d/car1/.../BS013.fits')
	parser.add_argument('-o', '--chi', type = float, help = \
		'Multiplicative factor to multiply uncertainties by. '
		'This is to reduce the minimum reduced chi^2 to 1, to '
		'get more accurate uncertaines on parameters as well '
		'as escape local minima. Example: 1.0000', default \
		= 1.000)
	# NP Adding description of parsers
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
	goodindex = args.spec.rfind('/')
	sname = args.spec[goodindex+1:goodindex+6]
	print('Name of object: ' +sname +'\n')
	global APO 
	APO = hdr['OBSERVAT'] == 'APO'
	print('Determining which models to use.\n')
	global ctemps, cgs, vsini, cwavls, cints
	ctemps, cgs, vsini, cwavls, cints = getmodels(APO)
	print('Estimating best fit parameters.\n')
	bestT, bestg, bestv = guess(wavl, data)
	# NP Getting best-fit paramaters
	if(~np.isnan(bestT +bestg +bestv)):
		s = open('./' +sname+'.dat', 'w')
		s.write(str(sname) +" params:\n")
		s.close()
		mcmc(bestT, bestg, bestv)
		# NP Running MCMC on the desired spectrum
		chis = np.loadtxt('./'+sname +'.dat', usecols = [5], \
			skiprows =1)
		bestchi = chis[np.argmin(chis)]
		# NP Reading best reduced chi-squared
		print('best chi2: ' +str(bestchi) +'\n')
		sigguess = bestchi *args.chi
		print('Guess at uncertainty factor: ' +str(sigguess) \
			+'\n')
		print('Done!\n')


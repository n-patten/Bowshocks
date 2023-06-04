import os
import emcee
import corner
import random
import argparse
import traceback
import numpy as np
import scipy as sp
from astropy.io import fits
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from PyAstronomy.pyasl import fastRotBroad, rotBroad
# NP Necessary imports

# rvsao
# xcsao BS013.fits template=test.fit st_l=3800 end_l=6000 low_bin=10 top_low=50 top_nrun=4096 pkfrac=-0.5 report_mode= 2
# hedit BS013.fits fiel=VRAD value=-15.862 add+
# dopcor BS013.fits BS013shift.fits redsh=VRAD isvel+ add- disper+ verbose+ flux-

dir = '/d/hya1/BS/spectra/'
# NP Directory of processed spectra
fnames = os.listdir(dir)
# NP Reading in the names of the files in the spectra directory
beginning = [i.rfind('.') for i in fnames]
# NP Finding the first instance of . to read file extensions
end = [len(i) for i in fnames]
# NP Index of the last character in each file name
extensions = [fnames[i][beginning[i]:end[i]] for i in \
	range(len(fnames))]
# NP Using the two previous indices to read the file extensions
ii = ([i == '.fits' for i in extensions])
# Limiting search to fits files
snames = np.array([f[0:-5] for f in fnames])[ii]
# NP All fits files in spectra directory
print(str(len(snames)) +' fits files found out of ' +str(len(fnames))\
	+' files in directory.\n')
# NP reading in model data
hdu = [fits.open(dir +n +'.fits') for n in snames]
# NP Opening spectrum
data= np.array([h[0].data[0][0] if len(h[0].data) != 4096 else \
	h[0].data for h in hdu])
# NP Reading in data
hdr = [h[0].header for h in hdu]
# NP Reading in header
crval =np.array([h['CRVAL1'] for h in hdr])
# NP Finding starting wavelength
cdelt = np.array([h['CD1_1'] for h in hdr])
# NP Finding plate scale in A per pix
crpix = np.array([h['CRPIX1'] for h in hdr])
# NP Finding starting pixel
wavl = np.array([crval[i] +cdelt[i] *(np.arange(0,len(data[i])) \
	-crpix[i]) for i in range(len(snames))])
# NP Creating wavelength range over whole spectrum
exp = np.array([h['EXPTIME'] for h in hdr])
# NP Exposure times for spectra
targs = np.loadtxt('/d/users/nikhil/Bowshocks/obslist/Nikhiltargets.'\
	'cat', dtype = str, delimiter=' ')
# NP Reading in target file
G = np.array([targs[:,8][i][2:] for i in range(len(targs[:,8]))],\
	dtype = float)
# NP G magnitude for each bowshock
targnames = np.array([targs[:,1][i] for i in range(len(targs))], \
	dtype = str)
# NP BS Identifiers name

dir = '/d/hya1/BS/model_spectra/norm/'
# NP Reading in normalized model spectra convolved to KOSMOS resolution
names = np.array(os.listdir(dir))
# NP Names of model spectra
ifiles = [n[-4:] == '.txt' for n in names]
# NP Limiting search to txt files
model = [np.loadtxt(dir +i, usecols = (0, 1)) for i in names[ifiles]]
# NP Reading in model data
temps = np.array([float(n[-12:-7]) for n in names[ifiles]])
# NP Finding temperature information for each model
gs = np.array([float(n[-7:-4]) /100 for n in names[ifiles]])
# NP Finding gravity information for each model
wavlsvac = [i.T[0] for i in model]
# NP Reading in vacuum model wavelenghts
ints = [i.T[1] for i in model]
# NP Reading in model intensities
s = [1e4 /w for w in wavlsvac]
# NP Temporary variable to convert to air wavelengths
nstp = [1 +0.0000834254 +0.02406147 /(130 -es**2) +0.00015998 \
	/(38.9 -es**2) for es in s]
# NP Finding index of refraction for air at standard temp. and pressure
n_tp = [1 + (76000 * n /(96095.43)) *(1e-8 *76000 *(0.601-0.00972 \
	*20) /(1 +0.003661 *20)) for n in nstp]
wavls = [wavlsvac[i] /nstp[i] for i in range(len(nstp))]
# NP Converting wavelengths to air wavelengths
# NP Conversion to air wavelengths was done by a procedure outlined in
# this link: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

cdir = '/d/hya1/BS/model_spectra/conv/'
# NP Directory for convolved spectra
cnames = np.array(os.listdir(cdir))
# NP Convolved model names
cifiles = [n[-4:] == '.txt' for n in cnames]
# NP Limiting to only text files
cmodel = [np.loadtxt(cdir +i, usecols = (0, 1)) for i in \
	cnames[cifiles]]
# NP reading in model data
ctemps = np.array([float(n[-12:-7]) for n in cnames[cifiles]])
# NP Finding temperature information for each model
cgs = np.array([float(n[-7:-4]) /100 for n in cnames[cifiles]])
# NP Finding gravity information for each model
vsini = np.array([float(n[4:-12]) for n in cnames[cifiles]])
# NP Finding vsini convolution for model
cwavlsvac = [i.T[0] for i in cmodel]
# NP Reading in vacuum wavelengths
cints = [i.T[1] for i in cmodel]
# NP Reading in convolved model intensities
s = [1e4 /w for w in cwavlsvac]
# NP Temporary variable for convesion to air wavelengths
nstp = [1 +0.0000834254 +0.02406147 /(130 -es**2) +0.00015998 \
	/(38.9 -es**2) for es in s]
# NP Index of refraction in air for each wavelength
cwavls = [cwavlsvac[i] /nstp[i] for i in range(len(nstp))]
# NP Convolved model spectra wavelengths in air
# NP Conversion to air wavelengths was done by a procedure outlined in
# this link: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

index = 0

def guess(spec):
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
	for i in range(len(snames)):
		if (snames[i] == spec):
			print('Spectrum found!\n')
			mask1 = wavl[i] > 4101-5
			mask2 = wavl[i] < 4101+5
			mask3 = np.logical_and(mask1, mask2)
			hd = wavl[i][mask3][np.argmin(data[i][mask3])]
			# NP Searching around the Hd region to find
			# NP Hd line in spectrum
			mask1 = cwavls[i] > 4101-5
			mask2 = cwavls[i] < 4101+5
			mask3 = np.logical_and(mask1, mask2)
			modelhd = cwavls[i+1][mask3][np.argmin(\
				cints[i][mask3])]
			# NP Searching around the Hd region to find
			# NP Hd line in model
			shift = modelhd -hd
			# NP Determing redshift guess from the
			# NP difference in Hd lines between the spectrum
			# NP and the model.
			spectrumspline = CubicSpline(wavl[i], data[i])
			# NP Creating a spline of the spectrum
			testrange = np.linspace(3800, 5000, 10000)
			# NP Wavelength range to evaluate differences
			# NP between models and spectra
			modelsplines = [CubicSpline(cwavls[a], \
				cints[a]) for a in range(len(cints))]
			# NP Creating a spline of all model spectra
			diffs = [np.sum(np.abs(d(testrange) \
				-spectrumspline(testrange-\
				shift)) **2) for d in modelsplines]
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
	print('Spectrum not found!\n')
	# NP Printing errror if spectrum not found
	return np.nan, np.nan, np.nan
	# NP Returning invalid parameters if no spectrum found

def interpspectra(T_targ, g_targ, plot):
	'''Interpolates between different temperature at and different
	log g model spectra. Uses a bilinear interpolation between 
	temperatures and log g to interpolate between four model spectra.
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
		tempdiffs = temps -T_targ
		# NP Finding the temperature difference between all
		# NP model spectra and target temperature.
		gdiffs = gs -g_targ
		# NP Finding the gravity difference between all
		# NP model spectra and target gravity.
		T_1 = T_targ +tempdiffs[tempdiffs < 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs < 0]))]
		# NP Finding lower temperature
		T_2 = T_targ +tempdiffs[tempdiffs > 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs > 0]))]
		# NP Finding upper temperature
		g_1 = g_targ +gdiffs[gdiffs < 0][np.argmin(\
			np.abs(gdiffs[gdiffs < 0]))]
		# NP Finding lower gravity
		g_2 = g_targ +gdiffs[gdiffs > 0][np.argmin(\
			np.abs(gdiffs[gdiffs > 0]))]
		# NP Finding upper gravity
		wavs1 = np.array(wavls)[(gs == g_1) & (temps == \
			T_1)][0]
		# NP Finding the wavelength array corresponding to the
		# NP lower temperature and lower gravity model
		wavs2 = np.array(wavls)[(gs == g_1) & (temps == \
			T_2)][0]
		# NP Finding the wavelength array corresponding to the
		# NP upper temperature and lower gravity model
		wavs3 = np.array(wavls)[(gs == g_2) & (temps == \
			T_1)][0]
		# NP Finding the wavelength array corresponding to the
		# NP higher gravity and lower temperature model
		wavs4 = np.array(wavls)[(gs == g_2) & (temps == \
			T_2)][0]
		# NP Finding the wavelength array corresponding to the
		# NP higher temperature and higher gravity model
		ints1 = np.array(ints)[(gs == g_1) & (temps == \
			T_1)][0]
		# NP Finding the spectrum of the lower temperature model
		ints2 = np.array(ints)[(gs == g_1) & (temps == \
			T_2)][0]
		# NP Finding the spectrum of the higher temperature 
		# NP and lower gravity model
		ints3 = np.array(ints)[(gs == g_2) & (temps == \
			T_1)][0]
		# NP Finding the spectrum of the higher gravity and
		# lower temperature modle
		ints4 = np.array(ints)[(gs == g_2) & (temps == \
			T_2)][0]
		# NP Finding the spectrum of the higher gravity model
		# NP and higher temperature
		S_T_g1 = ((T_2 -T_targ) /(T_2 -T_1)) *ints1 \
			+((T_targ -T_1) /(T_2 -T_1)) *ints2
		# NP Finding temperature interpolated spectrum at low
		# NP gravity
		S_T_g2 = ((T_2 -T_targ) /(T_2 -T_1)) *ints3 \
			+((T_targ -T_1) /(T_2 -T_1)) *ints4
		# NP Finding temperature interpolated spectrum at low
		# NP gravity
		S_T_g = ((g_2 -g_targ) /(g_2 -g_1)) *S_T_g1 \
			+((g_targ -g_1) /(g_2 -g_1)) *S_T_g2
		# NP Finding combined gravity interpolated and
		# NP temperature interpolated spectrum
		wav_new = np.linspace(3700, 5010, 15000)
		# NP Creating wavelength array to interpolate over
		if(plot):
		# If plotting is desired:
			#plt.plot(wavs1, ints1, label = 'T =' +str(T_1) \
			#	+', g =' +str(g_1))
			# NP Plotting first model spectrum
			#plt.plot(wavs4, ints4, label = 'T =' +str(T_2) \
			#	+', g =' +str(g_2))
			# NP Plotting fourth model spectrum
			#plt.plot(wavs2, ints2, label = 'T =' +str(T_2) \
			#	+', g =' +str(g_1))
			# NP Plotting second model spectrum
			#plt.plot(wavs3, ints3, label = 'T =' +str(T_1) \
			#	+', g =' +str(g_2))
			# NP Plotting third model spectrum
			interp = CubicSpline(wavs1, S_T_g)
			broadinterp = rotBroad(wav_new, \
				interp(wav_new), 0.3, 360)
			plt.plot(wav_new, broadinterp, label = 'T =' +\
				str(T_targ) +', g =' +str(g_targ))
			plt.plot(wavl[index], data[index])
			# NP Plotting interpolated spectrum
			plt.legend()
			# NP Adding a legend
			plt.xlim(4000, 4200)
			# NP Limiting plot to 4000-4200 Angstroms
			plt.show()
			# NP Showing plot
		spline = CubicSpline(wavs1, S_T_g)
		# NP Creating spline of interpolated spectrum
		return wav_new, spline
		# NP Returning wavelength array and interpolated
		# NP spectrum
	except Exception:
		print('Could not interpolate this temperature!\n')
		traceback.print_exc()
		# NP Print this string if temperature could not be
		# NP interpolated

def log_likelihood_T(theta, x, y, yerr):
	T, g, vsini, log_f = theta
	# NP Defining parameters
	print(T, g, vsini, log_f)
	# NP Printing parameters
	try:
		if((15000 < T < 50000) & (2 < g < 5) & (0 < vsini < \
			800) & (-10 < log_f < -1)):
			spec = data[index]
			# NP Finding spectrum
			wavs = wavl[index]
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
			wavs2, smodel = interpspectra(T, g, False)
			# Interpolating to desired T and log g

			scale = len(wavs2) /(np.max(wavs2) -np.min(wavs))
			convmodel = gaussian_filter(smodel, sigma = \
				4250 *scale *vsini /2 /3e5)
			convmodelspline = CubicSpline(wavs2, smodel(wavs2))
			bsigma = np.std(bluewavs) **2 +np.exp(2 *log_f)
			n, chi_1 = line_evaluate(convmodelspline, \
				specspline, wavs, 4100, bsigma)			
			N1 = n -3
			n, chi_2 = line_evaluate(convmodelspline, \
				specspline, wavs, 4009, bsigma)			
			N2 = n -3

			n, chi_3 = line_evaluate(convmodelspline, \
				specspline, wavs, 4026, bsigma)			
			N3 = n -3

			n, chi_4 = line_evaluate(convmodelspline, \
				specspline, wavs, 4121, bsigma)			
			N4 = n -3

			n, chi_5 = line_evaluate(convmodelspline, \
				specspline, wavs, 4144, bsigma)			
			N5 = n -3

			n, chi_6 = line_evaluate(convmodelspline, \
				specspline, wavs, 4129, bsigma)			
			N6 = n -3

			n, chi_7 = line_evaluate(convmodelspline, \
				specspline, wavs, 4200, bsigma)			
			N7 = n -3

			n, chi_8 = line_evaluate(convmodelspline, \
				specspline, wavs, 4340, bsigma)			
			N8 = n -3

			n, chi_9 = line_evaluate(convmodelspline, \
				specspline, wavs, 4387, bsigma)			
			N9 = n -3

			n, chi_10 = line_evaluate(convmodelspline, \
				specspline, wavs, 4340, bsigma)			
			N10 = n -3

			n, chi_11 = line_evaluate(convmodelspline, \
				specspline, wavs, 4471, bsigma)			
			N11 = n -3

			n, chi_12 = line_evaluate(convmodelspline, \
				specspline, wavs, 4481, bsigma)			
			N12 = n -3

			n, chi_13 = line_evaluate(convmodelspline, \
				specspline, wavs, 4486, bsigma)			
			N13 = n -3

			n, chi_14 = line_evaluate(convmodelspline, \
				specspline, wavs, 4504, bsigma)			
			N14 = n -3

			n, chi_15 = line_evaluate(convmodelspline, \
				specspline, wavs, 4541, bsigma)			
			N15 = n -3

			n, chi_16 = line_evaluate(convmodelspline, \
				specspline, wavs, 4568, bsigma)			
			N16 = n -3

			n, chi_17 = line_evaluate(convmodelspline, \
				specspline, wavs, 4575, bsigma)			
			N17 = n -3

			n, chi_18 = line_evaluate(convmodelspline, \
				specspline, wavs, 4656, bsigma)			
			N18 = n -3

			n, chi_19 = line_evaluate(convmodelspline, \
				specspline, wavs, 4686, bsigma)			
			N19 = n -3

			n, chi_20 = line_evaluate(convmodelspline, \
				specspline, wavs, 4713, bsigma)			
			N20 = n -3

			#mask1 = wavs2 > 4300
			# NP Limiting blue model to greater than 3990
			
			logprob1 = chi2.logpdf(chi_1, N1)
			logprob2 = chi2.logpdf(chi_2, N2)
			logprob3 = chi2.logpdf(chi_3, N3)
			logprob4 = chi2.logpdf(chi_4, N4)
			logprob5 = chi2.logpdf(chi_5, N5)
			logprob6 = chi2.logpdf(chi_6, N6)
			logprob7 = chi2.logpdf(chi_7, N7)
			logprob8 = chi2.logpdf(chi_8, N8)
			logprob9 = chi2.logpdf(chi_9, N9)
			logprob10 = chi2.logpdf(chi_10, N10)
			logprob11 = chi2.logpdf(chi_11, N11)
			logprob12 = chi2.logpdf(chi_12, N12)
			logprob13 = chi2.logpdf(chi_13, N13)
			logprob14 = chi2.logpdf(chi_14, N14)
			logprob15 = chi2.logpdf(chi_15, N15)
			logprob16 = chi2.logpdf(chi_16, N16)
			logprob17 = chi2.logpdf(chi_17, N17)
			logprob18 = chi2.logpdf(chi_18, N18)
			logprob19 = chi2.logpdf(chi_19, N19)
			logprob20 = chi2.logpdf(chi_20, N20)

			logprobtot = logprob1 +logprob2 +logprob3 \
				+logprob4 +logprob5 +logprob6 \
				+logprob7 +logprob8 +logprob9 \
				+logprob10 +logprob11 +logprob12 \
				+logprob13 +logprob14 +logprob15 \
				+logprob16 +logprob17 +logprob18 \
				+logprob19 +logprob20
			s = open('/d/hya1/BS/emcee/temp/' +snames[index]\
				+'.dat', 'a')
			datastr = '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\t' \
				'{3:5.5f}\t{4:5.5f}\t{5:5.5f}\n'\
				.format(T, vsini, chi_1, \
				logprobtot, bsigma **0.5, (chi_1 \
				+0) /N1)
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
	mask1 = w < lmbda +5
	mask2 = w > lmbda -5
	mask3 = np.logical_and(mask1, mask2)
	wavelengths = w[mask3]
	chi2 = np.sum((model(wavelengths) -spec(wavelengths)) \
		**2 /sigma)
	return len(wavelengths), chi2

def log_probability_T(theta, x, y, yerr):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood_T(theta, x, y, yerr)

def log_prior(theta):
	T, g, vsini, log_f = theta
	if 15000 < T < 50000 and 2 < g < 5 and 0 < vsini < 800:
		return 0.0
	return -np.inf

def mcmc(t, g, v):
	T_true = t
	g_true = g
	vsini_true = v
	log_f_true = -2
	# NP Setting best-fit temperature, log g, vsini and radial
	# NP velocity presets.
	nll = lambda *args: -log_likelihood_T(*args)
	initial = np.array([T_true, g_true, vsini_true, log_f_true])\
		+0.1 * np.random.randn(4)
	soln = minimize(nll, initial, args=(ints[1][0:3], ints[1][0:3]\
		,ints[1][0:3]))
	pos = soln.x + 1e-4 * np.random.randn(32, 4)
	nwalkers, ndim = pos.shape
	sampler = emcee.EnsembleSampler(nwalkers, ndim, \
		log_probability_T, args=(ints[1][0:3], ints[1][0:3], \
		ints[1][0:3]))
	sampler.run_mcmc(pos, 5000, progress=True);

	from IPython.display import display, Math
	flat_samples = sampler.get_chain(discard=600, thin=15,\
		flat=True)
	labels = ["T", "logg", "vsini", "log_f"]
	txt = ""
	for i in range(ndim):
		mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
		q = np.diff(mcmc)
		txt += "{{{3}}} = {0:.5f} -{1:.5f} +{2:.5f}\n"\
			.format(mcmc[1], q[0], q[1], labels[i])
	print(txt)
	# NP Printing best-fit parameters and uncertainties

	Tguess = np.percentile(flat_samples[:, 0], [50])
	gguess = np.percentile(flat_samples[:, 1], [50])
	vsiniguess = np.percentile(flat_samples[:, 2], [50])
	logfguess = np.percentile(flat_samples[:, 3], [50])

	f = plt.figure(facecolor = 'white', figsize = [24, 6])
	wmodel, model = interpspectra(Tguess, gguess, False)
	splspec = CubicSpline(wavl[index], data[index])
	colors = ['red', 'orange', 'green', 'blue', 'purple']
	cindex = int(len(colors) *random.random())
	convmodel = rotBroad(wmodel, model(wmodel), 0.3, vsiniguess)
	conmodelspl = CubicSpline(wmodel, convmodel)

	plt.subplot(1, 3, 1)
	plt.plot(wmodel, splspec(wmodel), color = \
		colors[cindex], label = snames[index])
	plt.plot(wmodel, conmodelspl(wmodel), '--k', label = \
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
	plt.xlim(4000, 4205)
	plt.ylim(0.65, 1.05)
	plt.xlabel(r'$\AA$')

	plt.subplot(1, 3, 2)
	plt.plot(wmodel, splspec(wmodel), color = \
		colors[cindex], label = snames[index])
	plt.plot(wmodel, conmodelspl(wmodel), '--k', label = \
		'Best fit params')
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
	plt.xlim(4300, 4505)
	plt.ylim(0.65, 1.05)
	plt.xlabel(r'$\AA$')

	plt.subplot(1, 3, 3)
	plt.plot(wmodel, splspec(wmodel), color = \
		colors[cindex], label = snames[index])
	plt.plot(wmodel, conmodelspl(wmodel), '--k', label = \
		'Best fit params')
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
	plt.xlim(4500, 4800)
	plt.ylim(0.65, 1.05)
	plt.xlabel(r'$\AA$')
	#plt.text(5010, 1, txt, fontsize = 'x-large')
	plt.savefig('/d/www/nikhil/public_html/research/emcee/temp'\
		'/' +snames[index] +'fullemceefittedparams.png', \
		bbox_inches ='tight')
	plt.show()
	# NP Plotting best-fit parameters

	flat_samples = sampler.get_chain(discard=600, thin=15,\
		flat=True)
	fig = corner.corner(flat_samples, labels = labels)
	fig.set_facecolor('white')
	fig.show()
	plt.savefig('/d/www/nikhil/public_html/research/emcee/temp/'\
		+snames[index] +'fullcorners.png')
	plt.show()

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Program to\
		run MCMC fitting on spectra to find temperature, \
		log g  and vsini with uncertainties.')
	# NP Adding description of program
	parser.add_argument('spec', type = str, help = 'BS \
		identifier of the bowshock object. Example: \
		BS013.')
	# NP Adding description of parsers
	args = parser.parse_args()
	# NP Adding parsers
	bestT, bestg, bestv = guess(args.spec)
	# NP Getting best-fit paramaters
	if(~np.isnan(bestT +bestg +bestv)):
		index = np.argwhere(snames == args.spec)[0][0]
		# NP Finding index of desired spectrum
		#mcmc(bestT, bestg, bestv)
		# NP Running MCMC on the desired spectrum
		chis = np.loadtxt('/d/hya1/BS/emcee/temp/' \
			+snames[index] +'.dat', usecols = [5])
		bestchi = chis[np.argmin(np.abs(chis -1))]
		# NP Reading best reduced chi-squared
		print('best chi2: ' +str(bestchi))

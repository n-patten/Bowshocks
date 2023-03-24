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
from PyAstronomy.pyasl import fastRotBroad
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
# NP Necessary imports

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

def interpspectra_T(T_targ, g_targ):
	'''Interpolates between different temperature model spectra at
	constant log g. Uses a linear interpolation between temperature
	difference to interpolate between two model spectra.
	---------------------------------------------
	Inputs
	-T_targ: float. Target temperature to interpolate to in Kelvin.
	-g_targ: float. Best fit log g result for models.
	---------------------------------------------
	Outputs
	-t_test: float. Best fit parameter for temperaure in Kelvin.
	-g_test: float. Best fit parameter for log g.
	'''
	try:
		tempdiffs = temps -T_targ
		# NP Finding the temperature difference between all
		# NP model spectra and target temperature.
		T_1 = T_targ +tempdiffs[tempdiffs < 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs < 0]))]
		# NP Finding lower temperature model
		T_2 = T_targ +tempdiffs[tempdiffs > 0][np.argmin(\
			np.abs(tempdiffs[tempdiffs > 0]))]
		# NP Finding upper temperature model
		wavs1 = np.array(wavls)[(gs == g_targ) & (temps == \
			T_1)][0]
		# NP Finding the wavelength array corresponding to the
		# NP lower temperature model
		wavs2 = np.array(wavls)[(gs == g_targ) & (temps == \
			T_2)][0]
		# NP Finding the wavelength array corresponding to the
		# NP upper temperature model
		ints1 = np.array(ints)[(gs == g_targ) & (temps == \
			T_1)][0]
		# NP Finding the spectrum of the lower temperature model
		ints2 = np.array(ints)[(gs == g_targ) & (temps == \
			T_2)][0]
		# NP Finding the spectrum of the higher temperature 
		# NP model
		spline1 = CubicSpline(wavs1, ints1)
		# NP Fidning spline for lower temeperature model
		spline2 = CubicSpline(wavs2, ints2)
		# NP Finding spline for higher temperature model
		wav_new = np.linspace(3700, 5000, 15000)
		# NP Creating wavelength array to interpolate over
		interp = spline1(wav_new) +(spline2(wav_new) -\
			spline1(wav_new)) *(T_targ -T_1) /(T_2 -T_1)
		# NP Linearlly interpolating between different
		# NP temperature spectra
		return wav_new, interp
		# NP Returning wavelength array and interpolated
		# NP spectrum
	except Exception:
		print('Could not interpolate this temperature!\n')
		traceback.print_exc()
		# NP Print this string if temperature could not be
		# NP interpolated

def log_likelihood_T(theta, x, y, yerr):
	T, vsini, z, l = theta
	try:
		if((15000 < T < 50000) & (0 < vsini < 800) & (-5 < z < \
			5) & (0 < l < 1)):
			spec = data[index]
			wavs = wavl[index]
			specspline = CubicSpline(wavs, spec)
			mask1 = wavs > 4256
			mask2 = wavs < 4264
			mask3 = np.logical_and(mask1, mask2)
			bluewavs = spec[mask3]
			mask1 = wavs > 4945
			mask2 = wavs < 4954
			mask3 = np.logical_and(mask1, mask2)
			redwavs = spec[mask3]
			wavs2, model = interpspectra_T(T, bestg)
			scale = len(wavs2) /(6200 -3700)
			mask1 = wavs > 4000
			mask2 = wavs < 4210
			mask3 = np.logical_and(mask1, mask2)
			#convmodel = gaussian_filter(model, sigma = \
			#	4400 *vsini *scale /(3e5) /2)
			convmodel = fastRotBroad(wavs2, model, l, \
				vsini)
			convmodelspline = CubicSpline(wavs2, convmodel)
			model = convmodelspline(wavs[mask3])
			y = specspline(wavs[mask3] +z)
			bsigma = np.std(bluewavs) **2
			chi_b = np.sum((y -model) **2 /bsigma)
			logprob1 = chi2.logpdf(chi_b, len(wavs[mask3]) \
				-3)
			mask1 = wavs > 4465
			mask2 = wavs < 5000
			mask3 = np.logical_and(mask1, mask2)
			rsigma = np.std(redwavs) **2
			chi_r = np.sum((specspline(wavs[mask3] \
				+z) -convmodelspline(wavs[mask3])) \
				**2 /rsigma)
			logprobtot = logprob1
			s = open('/d/hya1/BS/emcee/temp/' +snames[index]\
				+'.dat', 'a')
			datastr = '{0:5.5f}\t{1:5.5f}\t{2:5.5f}\t' \
				'{3:5.5f}\t{4:5.5f}\t{5:5.5f}'\
				'\t{6:5.5f}\t{7:5.5f}\n'\
				.format(T, vsini, z, chi_b, chi_r, \
				logprobtot, bsigma, rsigma)
			s.write(datastr)
			s.close()
			return logprobtot
		else:
		    	print('Skipping loop')
		    	return -np.inf
	except:
		traceback.print_exc()
		return -np.inf

def log_probability_T(theta, x, y, yerr):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood_T(theta, x, y, yerr)

def log_prior(theta):
	T, vsini, z, l = theta
	if 15000 < T < 50000 and 0 < vsini < 800 and -5 < z < 5 \
		and 0 < l < 1:
		return 0.0
	return -np.inf

def mcmc_T(t, g, v, z):
	T_true = t
	vsini_true = v
	z_true = z
	nll = lambda *args: -log_likelihood_T(*args)
	initial = np.array([T_true, vsini_true, z_true, 0.1]) \
		+0.1 * np.random.randn(4)
	soln = minimize(nll, initial, args=(ints[1][0:3], ints[1][0:3], \
		ints[1][0:3]))

	pos = soln.x + 1e-4 * np.random.randn(32, 4)
	nwalkers, ndim = pos.shape
	sampler = emcee.EnsembleSampler(nwalkers, ndim, \
		log_probability_T, args=(ints[1][0:3], ints[1][0:3], \
		ints[1][0:3]))
	sampler.run_mcmc(pos, 5000, progress=True);

	from IPython.display import display, Math
	flat_samples = sampler.get_chain(discard=600, thin=15, flat=True)
	labels = ["T", "vsini", "z", "l"]
	txt = ""
	for i in range(ndim):
		mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
		q = np.diff(mcmc)
		txt += "{{{3}}} = {0:.5f} -{1:.5f} +{2:.5f}\n"\
			.format(mcmc[1], q[0], q[1], labels[i])
	print(txt)

	Tguess = np.percentile(flat_samples[:, 0], [50])
	vsiniguess = np.percentile(flat_samples[:, 1], [50])
	zguess = np.percentile(flat_samples[:, 2], [50])
	f = plt.figure(facecolor = 'white', figsize = [16, 10])
	plt.xlim(3800, 5000)
	wmodel, model = interpspectra_T(Tguess, g)
	splspec = CubicSpline(wavl[index], data[index])
	plt.plot(wmodel, splspec(wmodel -zguess), 'r', label = \
		snames[index])
	scale = len(wmodel) /(6200 -3700)
	convmodel = gaussian_filter(model, sigma = 4400 *vsiniguess \
		*scale /2 /(3e5))
	conmodelspl = CubicSpline(wmodel, convmodel)
	plt.plot(wmodel, conmodelspl(wmodel), '--k', label = \
		'Best fit params')
	plt.ylim(0.65, 1.05)
	plt.legend()
	plt.xlabel(r'$\AA$')
	plt.text(5010, 1, txt, fontsize = 'x-large')
	#plt.savefig('/d/www/nikhil/public_html/research/emcee/temp\
		#/BS013fullemceefittedparams.png', bbox_inches ='tight')
	plt.show()

	flat_samples = sampler.get_chain(discard=600, thin=15, flat=True)
	fig = corner.corner(flat_samples, labels = labels)
	fig.set_facecolor('white')
	fig.show()
	#plt.savefig('/d/www/nikhil/public_html/research/emcee/temp/\
		#BS013fullcorners.png')
	plt.show()

if(__name__ == '__main__'):
	parser = argparse.ArgumentParser(description = 'Program to run\
		run MCMC fitting on spectra to find temperature, log g\
		vsini and redshift/blueshift with uncertainties.')
	# NP Adding description of program
	parser.add_argument('spec', type = str, help = 'BS identifier of\
		the bowshock object. Example: \'BS013\'.')
	# NP Adding description of parsers
	args = parser.parse_args()
	# NP Adding parsers
	bestT, bestg, bestv = guess(args.spec)
	# NP Getting best-fit paramaters
	if(~np.isnan(bestT +bestg +bestv)):
		index = np.argwhere(snames == args.spec)[0][0]
		mcmc_T(bestT, bestg, bestv, 0)

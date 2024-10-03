import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
# NP Necessary imports

def gauss(x, mu, t_1, t_2):
	'''
	---------------------------------------------
	Inputs
	-x: array. Wavelength array in Angstroms.
	-mu: float. Mean of the special Gaussian.
	-t_1: float. Inverse of the left dispersion in recipricol
	Angstroms.
	-t_2: float. Inverse of the right dispersion in recipricol
	Angstroms.
	---------------------------------------------
	Outputs
	-g: Array. Special Gaussian function evaluated at all of the
	inputted Wavelengths.
	'''
	g = np.array([np.exp(-t_1 **2 *(i -mu) **2 /2) if i < mu \
		else np.exp(-t_2 **2 *(i -mu) **2 /2) for i in x])
	# NP Defining a Gaussian with different widths about the mean
	return g
	# NP Returning special Gaussian function

def plot_dispersion(wavs, intensities, name):
	lam = wavs
	# NP Setting wavelengths based on inputted array
	X = 1.056 *gauss(lam, 5998, 0.00264, 0.00323) +0.362 \
		*gauss(lam, 4420, 0.00624, 0.00374) -0.065 \
		*gauss(lam, 5011, 0.00490, 0.00382)
	Y = 0.821 *gauss(lam, 5688, 0.00213, 0.00247) +0.286 \
		*gauss(lam, 5309, 0.00613, 0.00322)
	Z = 1.217 *gauss(lam, 4370, 0.00845, 0.00278) +0.681 \
		*gauss(lam, 4590, 0.00385, 0.00725)
	# NP Defining CIE 1931 XYZ color matching parameters. See
	# NP Wikipedia for more details:
	# NP https://en.wikipedia.org/wiki/CIE_1931_color_space
	x = X /(X +Y +Z)
	y = Y /(X +Y +Z)
	z = Z /(X +Y +Z)
	r = 2.36461385 *X -0.89654057 *Y -0.46807328 *Z
	g = -0.51516621 *X +1.4264081 *Y +0.0887571 *Z
	b = 0.0052037 *X -0.1440816 *Y +1.00920446 *Z
	A = np.max([np.max(r), np.max(g), np.max(b)])
	A_r = np.trapz(r, lam)
	A_g = np.trapz(g, lam)
	A_b = np.trapz(b, lam)
	R = r /A
	G = g /A
	B = b /A
	for i in range(len(R)):
		if R[i] < 0:
			white = -1 *R[i]
			if(R[i] +white <1):
				R[i] += white
			else:
				R[i] =1
			if(G[i] +white < 1):
				G[i] += white
			else:
				G[i] = 1
			if(B[i] +white < 1):
				B[i] += white
			else:
				B[i] = 1
		if G[i] < 0:
			white = -1 *G[i]
			if(R[i] +white <1):
				R[i] += white
			else:
				R[i] =1
			if(G[i] +white < 1):
				G[i] += white
			else:
				G[i] = 1
			if(B[i] +white < 1):
				B[i] += white
			else:
				B[i] = 1
		if B[i] < 0:
			white = -1 *B[i]
			if(R[i] +white <1):
				R[i] += white
			else:
				R[i] =1
			if(G[i] +white < 1):
				G[i] += white
			else:
				G[i] = 1
			if(B[i] +white < 1):
				B[i] += white
			else:
				B[i] = 1
	#R_clip = np.array([i if i > 0 else 0 for i in R])
	#G_clip = np.array([i if i > 0 else 0 for i in G])
	#B_clip = np.array([i if i > 0 else 0 for i in B])
	gamma = 2.4
	R_gamma = [1.055 *r **(1 /gamma) -0.055 if r > 0.0031308 \
		else 12.92 *r for r in R]
	G_gamma = [1.055 *r **(1 /gamma) -0.055 if r > 0.0031308 \
		else 12.92 *r for r in G]
	B_gamma = [1.055 *r **(1 /gamma) -0.055 if r > 0.0031308 \
		else 12.92 *r for r in B]
	scaled_intensities = intensities /np.max(intensities)
	norm_intensities = np.array([i if i > 0 else 0 for i in \
		scaled_intensities])
	R_spec = R_gamma *norm_intensities
	G_spec = G_gamma *norm_intensities
	B_spec = B_gamma *norm_intensities
	colors = [(R_spec[i], G_spec[i], B_spec[i], 1) for i in \
		range(len(R_gamma))]
	#plt.scatter(lam, [1 for i in lam], color = colors)
	plt.figure(figsize = [16, 4], layout = 'tight')
	for i in range(len(lam) -1):
		plt.fill_between([lam[i], lam[i+1]], 0, 512, color \
		= colors[i])
	plt.axis('off')
	plt.tight_layout()
	plt.margins(False, tight = True)
	plt.savefig('/d/users/nikhil/Bowshocks/dispersions/' +name \
		+'.png', bbox_inches = 'tight', pad_inches = 0, \
		transparent = True)
	
def get_fits_names(d):
	files = np.array(os.listdir(d))
	ii = [f[-4:] == 'fits' for f in files]
	fits_files = files[ii]
	names = np.array([f[:-5] for f in fits_files])
	return names

def read_spectrum(name):
	direc = '/d/hya1/nikhil/BS/spectra/'
	hdu = fits.open(direc +name +'.fits')
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
	return wavl, data
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Program to'\
		' turn a 1D spectrum to a visual dispersion of'\
		' colors')
	# NP Adding description of program
	parser.add_argument('spec', type = str, help = 'Name of '\
		'spectrum to make dispersion of. Example: BS013')
	# NP Adding description of arguments
	args = parser.parse_args()
	directory = '/d/hya1/nikhil/BS/spectra/'
	# NP Defining spectra directory
	names = get_fits_names(directory)
	# NP Getting fits files in spectra disrectory
	if any(names == args.spec):
		wavs, intensities = read_spectrum(args.spec)
		# NP Reading wavelengths and intensities from fits
		# NP file
		plot_dispersion(wavs, intensities, args.spec)
		# NP Mapping wavelengths to rgb colors and creating
		# NP dispersion plot
	else:
		print('Spectrum not found!')
	# NP Checking to see if inputted spectrum exists in spectra
	# NP directory. If True, proceed as usual. If not, terminate
	# NP program and print error message.
		
	
	

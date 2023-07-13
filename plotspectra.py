import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

if __name__ == '__main__':
	hdu1 = fits.open('/d/car1/nikhil/Data/proc/20230524/BS300.fits')
	# NP Opening spectrum
	data1= hdu1[0].data
	# NP Reading in data
	hdr1 = hdu1[0].header
	# NP Reading in header
	crval1 = hdr1['CRVAL1']
	# NP Finding starting wavelength
	cdelt1 = hdr1['CD1_1']
	# NP Finding plate scale in A per pix
	crpix1 = hdr1['CRPIX1']
	# NP Finding starting pixel
	wavl1 = crval1 +cdelt1 *(np.arange(0,len(data1))-crpix1)

	hdu2 = fits.open('/d/car1/nikhil/Data/proc/20230524/BS300.fits')
	# NP Opening spectrum
	data2= hdu2[0].data
	# NP Reading in data
	hdr2 = hdu2[0].header
	# NP Reading in header
	crval2 = hdr2['CRVAL1']
	# NP Finding starting wavelength
	cdelt2 = hdr2['CD1_1']
	# NP Finding plate scale in A per pix
	crpix2 = hdr2['CRPIX1']
	# NP Finding starting pixel
	wavl2 = crval2 +cdelt2 *(np.arange(0,len(data2)))

	hdu3 = fits.open('/d/car1/nikhil/Data/proc/20230524/BS300.fits')
	# NP Opening spectrum
	data3= hdu3[0].data
	# NP Reading in data
	hdr3 = hdu3[0].header
	# NP Reading in header
	crval3 = hdr3['CRVAL1']
	# NP Finding starting wavelength
	cdelt3 = hdr3['CD1_1']
	# NP Finding plate scale in A per pix
	crpix3 = hdr3['CRPIX1']
	# NP Finding starting pixel
	wavl3 = crval3 +cdelt3 *(np.arange(0,len(data3))-crpix3)

	plt.figure(figsize = [8, 3.6], facecolor = 'white')
	plt.plot(wavl1, data1, color = 'blue', label = 'BS300')
	#plt.plot(wavl2, data2, color = 'green', label = 'no minus')
	#plt.plot(wavl3, data3, color = 'red', label = 'minus')
	plt.legend()
	plt.xlim(4549.8, 4557.1)
	plt.ylim(0.9, 0.998)
	plt.show()
	

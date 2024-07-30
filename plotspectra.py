import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

if __name__ == '__main__':
	files = os.listdir('/d/hya1/nikhil/BS/spectra/')
	#files = os.listdir('/d/hya1/nikhil/BS/temp/')
	# NP Finding files in directory
	ispec = np.array([f[-4:] == 'fits' for f in files])
	# NP Limiting selection to only fits files
	for i in np.array(files)[ispec]:
		print('Working on spectrum: ' +i)
		hdu1 = fits.open('/d/hya1/nikhil/BS/spectra/' +i)
		#hdu1 = fits.open('/d/hya1/nikhil/BS/temp/' +i)
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
		name = hdr1['OBJNAME']
		wavl1 = crval1 +cdelt1 *(np.arange(0,len(data1))-crpix1)

		f, (ax1, ax2, ax3) = plt.subplots(1, 3, \
			figsize = [20, 8], facecolor = 'white', \
			gridspec_kw = {'width_ratios': [4, 9, 3]}, \
			sharey = True)
		#f.suptitle(name +' spectrum', fontsize = 24)
		minimum = np.min(data1[np.logical_and(wavl1 > \
			4000, wavl1 < 4900)])
		nearest_tenth = np.round(minimum *10) /10
		axiskwargs = dict(fontsize = 15,
			)

		f.supylabel('Relative intensity', fontsize = 15)
		ax1.set_ylim(minimum -0.03, 1.14)
		ax1.set_xlim(4000, 4210)
		ax1.set_yticks(np.arange(nearest_tenth, 1.15, 0.05))
		ax1.set_xticks(np.arange(4000, 4250, 50))
		ax1.set_yticklabels(np.round(np.arange(\
			nearest_tenth, 1.15, 0.05), decimals = 2), \
			**axiskwargs)
		ax1.set_xticklabels(['4000', '4050', '4100', \
			'4150',	'4200'], **axiskwargs)
		ax1.plot(wavl1, data1, color = 'darkred', label = \
			name)		
		ax1.vlines(x=4009, color = 'k', linewidth = 1, ymax = 1.03, \
			ymin = 1.01)
		ax1.text(4009, 1.038, r'He I 4009', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4026, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4026, 1.038, r'He I+II 4026', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4058, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4058, 1.038, r'N IV', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4068, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.vlines(x=4069, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.vlines(x=4070, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.hlines(y = 1.03, xmin = 4068, xmax = 4070, color = 'k'\
			, linewidth = 1)
		ax1.text(4069, 1.038, r'C III', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4070, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.vlines(x=4072, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.vlines(x=4076, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.hlines(y = 1.03, xmin = 4070, xmax = 4076, color = 'k'\
			, linewidth = 1)
		ax1.text(4073, 1.038, r'O II', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4089, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4089, 1.038, r'Si IV', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4101, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4101, 1.038, r'H$\delta$', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4116, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4116, 1.038, r'Si IV', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4121, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4121, 1.038, r'He I 4121', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4128, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.vlines(x=4130, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.hlines(y = 1.03, xmin = 4128, xmax = 4130, color = 'k'\
			, linewidth = 1)
		ax1.text(4129, 1.038, r'Si II', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4144, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4144, 1.038, r'He I 4144', rotation = 90, \
			ha = 'center')
		ax1.vlines(x=4200, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax1.text(4200, 1.038, r'He II 4200', rotation = 90, \
			ha = 'center')

		ax2.set_ylim(ax1.get_ylim())
		ax2.set_xticks(np.arange(4300, 4800, 50))
		ax2.set_xticklabels(['4300', '4350', '4400', \
			'4450', '4500', '4550', '4600', '4650', \
			'4700', '4750'], \
			**axiskwargs)
		f.supxlabel(r'Wavelength $\lambda$ $(\AA)$', \
			fontsize = 20)
		ax2.plot(wavl1, data1, color = 'darkred', label = \
			name)
		ax2.set_xlim(4300, 4750)
		ax2.vlines(x=4326, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4326, 1.038, r'C III', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4340, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4340, 1.038, r'H$\gamma$', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4349, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4349, 1.038, r'O II', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4379, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4379, 1.038, r'N III', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4388, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4388, 1.038, r'He I 4388', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4415, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4417, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4415, xmax = 4417, color = 'k'\
			, linewidth = 1)
		ax2.text(4416, 1.038, r'O II', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4420, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4436, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4420, xmax = 4436, color = 'k'\
			, linewidth = 1)
		ax2.text(4428, 1.038, r'DIB', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4471, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4471, 1.038, r'He I 4471', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4481, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4481, 1.038, r'Mg II 4481', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4511, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4515, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4511, xmax = 4515, color = 'k'\
			, linewidth = 1)
		ax2.text(4513, 1.038, r'N III', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4541, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4541, 1.038, r'He II 4541', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4552, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4568, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4575, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4552, xmax = 4575, color = 'k'\
			, linewidth = 1)
		ax2.text(4563.5, 1.038, r'Si III', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4604, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4620, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4604, xmax = 4620, color = 'k'\
			, linewidth = 1)
		ax2.text(4612, 1.038, r'N V', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4634, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4640, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4642, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4634, xmax = 4642, color = 'k'\
			, linewidth = 1)
		ax2.text(4638, 1.038, r'N III', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4640, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4650, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4640, xmax = 4650, color = 'k'\
			, linewidth = 1)
		ax2.text(4645, 1.038, r'O II', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4631, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4631, 1.038, r'N II', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4647, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.vlines(x=4652, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.hlines(y = 1.03, xmin = 4647, xmax = 4652, color = 'k'\
			, linewidth = 1)
		ax2.text(4649.5, 1.038, r'C III', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4654, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4654, 1.038, r'Si IV', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4658, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4658, 1.038, r'C IV', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4686, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4686, 1.038, r'He II 4686', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4713, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4713, 1.038, r'He I 4713', rotation = 90, \
			ha = 'center')
		ax2.vlines(x=4727, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax2.text(4727, 1.038, r'DIB', rotation = 90, \
			ha = 'center')
		#ax2.vlines(x=4762, color= 'k', linewidth = 1, ymax = 1.03\
		#	, ymin = 1.01)
		#ax2.vlines(x=4765, color= 'k', linewidth = 1, ymax = 1.03\
		#	, ymin = 1.01)
		#ax2.hlines(y = 1.03, xmin = 4762, xmax = 4765, color = 'k'\
		#	, linewidth = 1)
		#ax2.text(4763.5, 1.038, r'DIB', rotation = 90, \
		#	ha = 'center')
			
		ax3.set_ylim(ax1.get_ylim())
		ax3.set_xticks(np.arange(4850, 5050, 50))
		ax3.set_xticklabels(['4850', '4900', '4950', \
			'5000'], **axiskwargs)
		ax3.set_yticklabels(np.round(np.arange(\
			nearest_tenth, 1.15, 0.05), decimals = 2), \
			**axiskwargs)
		f.supxlabel(r'Wavelength $\lambda$ $(\AA)$', \
			fontsize = 20)
		ax3.plot(wavl1, data1, color = 'darkred', label = \
			name)
		ax3.set_xlim(4840, 5000)
		ax3.vlines(x=4861, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		plt.text(4861, 1.038, r'H$\beta$', rotation = 90, \
			ha = 'center')
		ax3.vlines(x=4880, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax3.vlines(x=4887, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		ax3.hlines(y = 1.03, xmin = 4880, xmax = 4887, color = 'k'\
			, linewidth = 1)
		plt.text(4883.5, 1.038, r'DIB', rotation = 90, \
			ha = 'center')
		ax3.vlines(x=4922, color= 'k', linewidth = 1, ymax = 1.03\
			, ymin = 1.01)
		plt.text(4922, 1.038, r'He I 4922', rotation = 90, \
			ha = 'center')
		
		ax1.spines['right'].set_visible(False)
		ax2.spines['left'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		ax3.spines['left'].set_visible(False)
		ax1.yaxis.tick_left()
		ax1.tick_params(labelright = False, labelleft = \
			True, right = False, left = True)
		ax2.tick_params(labelleft = False, labelright = \
			False, left = False, right = False)
		ax3.tick_params(labelleft = False, labelright = \
			True, left = False, right = True)
		ax3.yaxis.tick_right()

		d = 0.015
		kwargs = dict(transform=ax1.transAxes, color = 'k', \
			clip_on= False)
		ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
		ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
		kwargs.update(transform=ax2.transAxes)
		ax2.plot((-d*4/9, +d*4/9), (1-d, 1+d), **kwargs)
		ax2.plot((-d*4/9, +d*4/9), (-d, +d), **kwargs)
		ax2.plot((1-d*4/9, 1+d*4/9), (1-d, 1+d), **kwargs)
		ax2.plot((1-d*4/9, 1+d*4/9), (-d, +d), **kwargs)
		kwargs.update(transform=ax3.transAxes)
		ax3.plot((-d*4/3, +d*4/3), (1-d, 1+d), **kwargs)
		ax3.plot((-d*4/3, +d*4/3), (-d, +d), **kwargs)
		
		f.tight_layout(pad=1.6)
		plt.savefig('/d/www/nikhil/public_html/research/'
			'spectra/newspectra/' +name +'.png', \
			bbox_inches = 'tight')
		plt.close()
	

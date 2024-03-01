import healpy as H
import numpy as np
import scipy.optimize

from CRAToPy.coordinates import Lambert_pix2ang

__all__ = [
	"EWderivative",
	"EWdipole",
	"EWderivative_Lambert",
]

def EWderivative(nside,ntimes,nbins,thetamax,CRmap) :
	
	npix = H.nside2npix(nside)

	if not (ntimes % nbins) == 0 :
		print("Error: Parameter nbins need to be a divisor of ntimes.")
		exit(3)
		
	group = ntimes // nbins
	
	da1E = np.zeros(nbins, dtype=np.double)
	da2E = np.zeros(nbins, dtype=np.double)
	da1W = np.zeros(nbins, dtype=np.double)
	da2W = np.zeros(nbins, dtype=np.double)
		
	pixlist = np.arange(0,npix)
	theta,phi = H.pix2ang(nside,pixlist)
	
	phi = 2.0*np.pi-phi # healpy convention 
	
	condtheta = theta <= thetamax
	
	cond1 = phi > 0
	cond2 = phi < np.pi
	eastpixels = pixlist[np.logical_and(np.logical_and(cond1,cond2),condtheta)] 
	
	cond1 = phi > np.pi
	cond2 = phi < 2*np.pi
	westpixels = pixlist[np.logical_and(np.logical_and(cond1,cond2),condtheta)] 
	
	totevents = 0.0
	
	for timeidx2 in range(0,ntimes) :
		
		timeidx = timeidx2//group
		
		totevents += sum(CRmap[timeidx2])
		
		da1E[timeidx] += sum(np.sin(phi[eastpixels])*np.sin(theta[eastpixels])*CRmap[timeidx2][eastpixels]) 
		da2E[timeidx] += sum(CRmap[timeidx2][eastpixels]) 
			
		da1W[timeidx] += sum(np.sin(phi[westpixels])*np.sin(theta[westpixels])*CRmap[timeidx2][westpixels]) 
		da2W[timeidx] += sum(CRmap[timeidx2][westpixels]) 
		
	totda1E = sum(da1E)	
	totda1W = sum(da1W)
	totda2E = sum(da2E)	
	totda2W = sum(da2W)
	
	dalpha = (totda1E/totda2E-totda1W/totda2W)/2.
	
	EWint = []
		
	temp = (da2E - da2W)/(da2E + da2W)	
	total = sum(temp)/dalpha*(np.pi*2/nbins)
	EW = temp/dalpha
	
	#normalize
	bgr = sum(EW)/nbins
	EW = EW - bgr
	
	#uncertainty
	dEW = 2.*np.sqrt(da2E*da2W)/np.sqrt((da2E+da2W)**3)/dalpha
	
	#integrate (for traditional plots)
	total = 0.0
	for i in range(0,nbins) :
		EWint.append(total)
		total += EW[i]*(np.pi*2/nbins)
	
	bgr = sum(EWint)/nbins
	for i in range(0,nbins) :
		EWint[i] = EWint[i]-bgr
	
	return np.array(EW),np.array(dEW),np.array(EWint),dalpha
	
def EWderivative_Lambert(ndec,nbins,CRmap) :
	
	npix = 2*ndec*ndec
	ntimes = 2*ndec
	pixlist = np.arange(0,npix)

	if not (ntimes % nbins) == 0 :
		print("Error: Parameter nbins need to be a divisor of ntimes.")
		exit(3)
		
	group = ntimes // nbins
	
	da1E = np.zeros(nbins, dtype=np.double)
	da2E = np.zeros(nbins, dtype=np.double)
	da1W = np.zeros(nbins, dtype=np.double)
	da2W = np.zeros(nbins, dtype=np.double)
		
	theta,phi = Lambert_pix2ang(ndec,pixlist)
	
	cond1 = phi > 0
	cond2 = phi < np.pi
	eastpixels = pixlist[np.logical_and(cond1,cond2)] 
	
	cond1 = phi > np.pi
	cond2 = phi < 2*np.pi
	westpixels = pixlist[np.logical_and(cond1,cond2)] 
	
	totevents = 0.0
	
	for timeidx2 in range(0,ntimes) :
		
		timeidx = timeidx2//group
		
		totevents += sum(CRmap[timeidx2])
		
		da1E[timeidx] += sum(np.sin(phi[eastpixels])*np.cos(theta[eastpixels])*CRmap[timeidx2][eastpixels]) 
		da2E[timeidx] += sum(CRmap[timeidx2][eastpixels]) 
			
		da1W[timeidx] += sum(np.sin(2*np.pi-phi[westpixels])*np.cos(theta[westpixels])*CRmap[timeidx2][westpixels])
		da2W[timeidx] += sum(CRmap[timeidx2][westpixels]) 
		
	totda1E = sum(da1E)	
	totda1W = sum(da1W)
	totda2E = sum(da2E)	
	totda2W = sum(da2W)
	
	dalpha = (totda1E/totda2E + totda1W/totda2W)/2.
	
	EWint = []
		
	temp = (da2E - da2W)/(da2E + da2W)	
	total = sum(temp)/dalpha*(np.pi*2/nbins)
	EW = temp/dalpha
	
	#normalize
	bgr = sum(EW)/nbins
	EW = EW - bgr
	
	#uncertainty
	dEW = 2.*np.sqrt(da2E*da2W)/np.sqrt((da2E+da2W)**3)/dalpha
	
	#integrate (for traditional plots)
	total = 0.0
	for i in range(0,nbins) :
		EWint.append(total)
		total += EW[i]*(np.pi*2/nbins)
	
	bgr = sum(EWint)/nbins
	for i in range(0,nbins) :
		EWint[i] = EWint[i]-bgr
	
	return np.array(EW),np.array(dEW),np.array(EWint),dalpha
	
def EWdipole(nbins,EW,dEW,nmax=1) :
	
	def func(x,*a):
		temp =  0.0 
		for i in range(1,2) :
			temp += a[2*(i-1)]*np.cos(i*(x-a[2*(i-1)+1]))
		return temp
	
	hours = (0.5+np.arange(0,nbins))/nbins*2*np.pi
	
	x0 = 	np.zeros(2*nmax, dtype=np.double)
	result = scipy.optimize.curve_fit(func, hours, EW, x0, dEW)
	
	Amp = result[0][0]
	dAmp = np.sqrt(result[1][0][0])
	phi = (result[0][1])/2./np.pi*360 + 90.0
	dphi = (np.sqrt(result[1][1][1]))/2./np.pi*360
	
	if Amp < 0.0 :
		Amp = -Amp
		phi = (phi + 180.0) % 360.0
		
	return Amp,dAmp,phi,dphi
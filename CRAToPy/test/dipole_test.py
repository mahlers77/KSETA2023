import sys
sys.path.append('/Users/mahlers/Documents/git/CRAPy')

import CRAPy
from CRAPy.coordinates import *
from CRAPy.datatools import Lambertview

import numpy as np

import healpy as H
import numpy as np
from pylab import *
import pickle
import scipy.optimize
from scipy.special import gammainc
from scipy.special import gammaincc
from scipy.special import gamma
from scipy.special import erfcinv
from scipy.special import erfinv
from scipy.special import erf
import copy
	
if __name__ == "__main__":
		
	my_cmap = copy.copy(cm.get_cmap("RdBu_r"))
	my_cmap.set_under('w') # sets background to white
	my_cmap.set_over('w')
	my_cmap.set_bad('w')
	#my_cmap.colorbar_extend = True
	
	THETAMAX = 30./180*np.pi 
	LATITUDE = 49.1/180*np.pi
	TOPHAT = 45.0/180*np.pi
	NITER = 21 # 101
	
	ndec = 180
	npix= 2*ndec*ndec
	ntimes = 2*ndec
	
	pixlist = np.arange(0,npix,1)
	
	dec, RA = Lambert_pix2ang(ndec,pixlist)
	
	vx = np.cos(RA)*np.cos(dec)
	vy = np.sin(RA)*np.cos(dec)
	vz = np.sin(dec)
	
	#deltaI = 0.01*np.sin(RA)*np.cos(dec)
	deltaI = 0.001*np.sin(RA)*np.cos(dec)
	
	norm = np.random.poisson(100,ntimes)*1e9/ntimes/100
	
	vLCx,vLCy,vLCz = EQ2LC_vector(vx,vy,vz,0/180*np.pi,LATITUDE)
	
	vLC = np.transpose([vLCx,vLCy,vLCz])
	theta, phi = H.vec2ang(vLC)

	pixlistFOV = pixlist[theta < THETAMAX]
	
	Emap = np.zeros(npix, dtype=np.double)
	Emap[pixlistFOV] = np.cos(theta[pixlistFOV])*(1.0+1.0*np.sin(phi[pixlistFOV]-45./180.*np.pi)*np.sin(theta[pixlistFOV]))
	
	Emap = Emap/np.sum(Emap)
	 
	#Lambertview(ndec,deltaI,cmap= my_cmap)
	#show()
	
	#Lambertview(ndec,Emap,cmap= my_cmap)
	#show()
	
	CRmap = CRAPy.mockdata_Lambert(ndec,deltaI,Emap,norm)
	
	mymask = np.zeros(npix, dtype=np.double)
	
	for i in range(0,ntimes) :
		mymask += CRmap[i]
	
	print(sum(mymask))
	
	#Lambertview(ndec,mymask)
	#show()
	
	mymask[mymask > 0.0] = 1.0
	
	#Lambertview(ndec,mymask)
	#show()
	
	bestfit = CRAPy.INA_Lambert(ndec,LATITUDE,THETAMAX,maskLC=mymask)
	bestfit.firstguess(CRmap)
	out = CRAPy.extract_dipole_Lambert(CRmap,bestfit,NITER)
	print(out[:7])
	print(out[4]/out[5])
	tempmap = bestfit.I -np.ones(npix)
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	Lambertview(ndec,tempmap,cmap=my_cmap)
	show()
	#exit(3)
	
	bestfit = CRAPy.INA_Lambert(ndec,LATITUDE,THETAMAX,maskLC=mymask)
	
	bestfit.firstguess(CRmap)
	
	chi2list = CRAPy.extract_allsky_Lambert(CRmap,bestfit,NITER)
	
	tempmap = bestfit.I-np.ones(npix)
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	Lambertview(ndec,tempmap,vmin=-mymax,vmax=mymax,cmap=my_cmap)
	show()
	
	smoothdeltaI,smoothsignificance,trialmap = bestfit.tophat(CRmap,TOPHAT)
	
	mymax=max(np.amax(smoothdeltaI),-np.amin(smoothdeltaI))	
	Lambertview(ndec,smoothdeltaI,vmin=-mymax,vmax=mymax,cmap=my_cmap)
	show()
	
	tempmap = Lambert_to_healpy(ndec,64,smoothdeltaI)
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	H.mollview(tempmap,rot=180,cmap=my_cmap,min=-mymax,max=mymax)
	H.graticule()
	show()
	
	tempmap = np.sqrt(smoothsignificance)*np.sign(smoothdeltaI)
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	Lambertview(ndec,tempmap,cmap=my_cmap,vmin=-mymax,vmax=mymax)
	show()
	
	tempmap = Lambert_to_healpy(ndec,64,np.sqrt(smoothsignificance)*np.sign(smoothdeltaI))
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	H.mollview(tempmap,rot=180,cmap=my_cmap,min=-mymax,max=mymax)
	H.graticule()
	show()
	
	tempmap = np.sqrt(2)*erfinv(erf(np.sqrt(smoothsignificance)/np.sqrt(2))**trialmap)*np.sign(smoothdeltaI)
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	Lambertview(ndec,tempmap,cmap=my_cmap,vmin=-mymax,vmax=mymax)
	show()
	
	tempmap = Lambert_to_healpy(ndec,64,tempmap)
	mymax=max(np.amax(tempmap),-np.amin(tempmap))	
	H.mollview(tempmap,rot=180,cmap=my_cmap,min=-mymax,max=mymax)
	H.graticule()
	show()
	
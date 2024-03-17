import healpy as H
import numpy as np
from CRAToPy.coordinates import Pixel_Lambert
from CRAToPy.coordinates import Lambert_pix2ang
from pylab import *

__all__ = [
	"mockdata",
	"mockdata_Lambert",
	"Lambertview",
	"smoothdata",
	"rebindata",
]

def rebindata(ndec,ndec2,mymap) :

	step = np.array(ndec/ndec2)
	
	npix = 2*ndec*ndec
	
	npix2 = 2*ndec2*ndec2
	mymap2 = np.zeros(npix2)
	
	for i in range(0,npix) :
	
		pix_x = i % (2*ndec)
		pix_y = i // (2*ndec) 
		pix_x = pix_x // step.astype(int)
		pix_y = pix_y // step.astype(int)
		
		j = pix_x + 2*ndec2*pix_y

		mymap2[j] += mymap[i]
	
	return mymap2

def Lambertview(ndec,mymap,vmin=None,vmax=None,cmap=None) :
	
	npix = 2*ndec*ndec
	
	pixlist = np.arange(0,npix,1)
	
	dec, RA = Lambert_pix2ang(ndec,pixlist)
	
	dec = dec.reshape((ndec, 2*ndec))
	RA = RA.reshape((ndec, 2*ndec))
	val = mymap.reshape((ndec, 2*ndec))

	fig = figure(figsize=(12,5))
	ax = fig.add_subplot(1,1,1)
	ax.invert_xaxis()

	xticks((0.0/180.*np.pi,45/180.*np.pi,90/180.*np.pi,135/180.*np.pi,\
	180/180.*np.pi,225/180.*np.pi,270/180.*np.pi,315/180.*np.pi,360/180.*np.pi), ('0','3','6','9','12','15','18','21','24'))
	
	yticks((-1,-0.5,0.0,0.5,1), ('-1.0','-0.5','0','0.5','1.0'))
	
	if vmin == None :
		vmin = np.min(mymap)
		
	if vmax == None :
		vmax = np.max(mymap)
		
	if cmap == None :
		im = plt.pcolormesh(RA, np.sin(dec), val,vmin=vmin,vmax=vmax,shading='nearest')
	else :	
		im = plt.pcolormesh(RA, np.sin(dec), val,vmin=vmin,vmax=vmax,cmap = cmap,shading='nearest')
		
	fig.colorbar(im)
	
	
def mockdata_Lambert(ndec,deltaI,A,N) :

	npix = 2*ndec*ndec
	ntimes = 2*ndec
	pixlist = np.arange(0,npix)

	LC = Pixel_Lambert(ndec,pixlist)
	
	ISOmap = np.ones(npix, dtype=np.double)
	I = deltaI + ISOmap
	
	CRmap = np.zeros((ntimes,npix), dtype=np.double)
			
	for timeidx in range(0,ntimes) :
		
		pixlistEQ = LC.LC2EQ(timeidx)
		
		lam = N[timeidx]*A[pixlist]*I[pixlistEQ]
			
		CRmap[timeidx] = np.random.poisson(lam=lam)
		
	return CRmap
	
def mockdata(nside,ntimes,thetamax,latitude,deltaI,A,N) :

	npix = H.nside2npix(nside)
	pixlist = np.arange(0,npix)

	vx,vy,vz = H.pix2vec(nside,pixlist)
	
	ISOmap = np.ones(npix, dtype=np.double)
	I = deltaI + ISOmap
	
	CRmap = np.zeros((ntimes,npix), dtype=np.double)
			
	for timeidx in range(0,NTIMES) :
			
		#hour = (0.5+timeidx)/ntimes*np.pi*2
		
		#randomize arrival time within time window bin
		hour = (np.random.rand(len(pixlist))+timeidx)/ntimes*np.pi*2
		vpx,vpy,vpz = LC2EQ_vector(vx,vy,vz,hour,latitude)
		
		pixlistEQ = H.vec2pix(nside,vpx,vpy,vpz) 
		
		lam = N[timeidx]*A[pixlist]*I[pixlistEQ]
			
		CRmap[timeidx] = np.random.poisson(lam=lam)
		
	return CRmap

def smoothdata(nside,ntimes,CRmap,thetamax,fwhm) :

	npix = H.nside2npix(nside)
	
	CRmapsmooth = np.zeros((ntimes,npix), dtype=np.double)
	
	pixlist = np.arange(0,npix)
	
	vx,vy,vz = H.pix2vec(nside,pixlist)
	theta,phi = H.pix2ang(nside,pixlist)
	
	pixlistNOTFOV = pixlist[theta > thetamax]
	
	for timeidx in range(0,ntimes) : 
	
		CRmapsmooth[timeidx] = H.smoothing(CRmap[timeidx],fwhm=fwhm,verbose=False)
	
		CRmapsmooth[timeidx][pixlistNOTFOV] = 0.0
		
		pixlistNEG = pixlist[CRmapsmooth[timeidx] < 0.0]
		
		CRmapsmooth[timeidx][pixlistNEG] = 0.0
		
	return CRmapsmooth
import sys
sys.path.append('/Users/mahlers/Documents/git/CRAToPy')

import CRAToPy
from CRAToPy.coordinates import *
from CRAToPy.datatools import Lambertview
from CRAToPy.datatools import rebindata
from astropy.time import Time

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
from scipy.special import sph_harm

import copy
	
if __name__ == "__main__":
		
	my_cmap = copy.copy(cm.get_cmap("RdBu_r"))
	my_cmap.set_under('w') # sets background to white
	my_cmap.set_over('w')
	my_cmap.set_bad('w')
	my_cmap.colorbar_extend = True
	
	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Palatino']})
	
	#maximal zenith (in degree)
	THETAMAX = 80.
	
	#detector location (in degree)
	LATITUDE = -35.2
	LONGITUDE = -69.5
	
	#smoothing scale
	TOPHAT = 45.0/180*np.pi
	
	#number of iterations
	NITER = 21
	
	#number of decliation steps
	ndec = 180
	npix= 2*ndec*ndec
	
	#number of time steps
	ntimes = 2*ndec

	time0 = Time(0.0,format='unix',location=(LONGITUDE,LATITUDE))
	sidereal0 = time0.sidereal_time('mean')
	hourangle0 = sidereal0.hourangle
	
	omegaorbit = 2*np.pi/(365.24217*24.0*60.0*60.0) 
	omegasol = 2*np.pi/(24.0*60.0*60.0)
	omega = omegasol + omegaorbit
	
	lines = [line.strip() for line in open("../resources/Auger2017/Arrival_directions_8EeV_Science_2017.dat")]
	Augerdata = np.array([line.split() for line in lines[33:]])
	
	dec = np.array([float(x)/180.*np.pi for x in Augerdata.transpose()[2]])
	ra = np.array([float(x)/180.*np.pi for x in Augerdata.transpose()[3]])
	time = np.array([float(x) for x in Augerdata.transpose()[6]])
	
	deltaRA = (ra - omega*time - hourangle0/24.*2.0*np.pi) % (2.*np.pi)
	pixels = Lambert_ang2pix(ndec,dec,deltaRA)
	
	hourangle = (omega*time + hourangle0/24.*2.0*np.pi) % (2.*np.pi)
	timebins = np.digitize(hourangle,np.linspace(0.0, 2.*np.pi, num=ntimes+1))-1
	
	CRmap = np.zeros((ntimes,npix), dtype=np.double)
	testmap = np.zeros(npix, dtype=np.double)
	
	for i in range(0,len(Augerdata)) :
		
		CRmap[timebins[i]][pixels[i]] += 1.0 
		
		testmap[pixels[i]] += 1.0
	
	bestfit = CRAToPy.INA_Lambert(ndec,LATITUDE/180.*np.pi,THETAMAX/180.*np.pi)
	
	bestfit.firstguess(CRmap)
	
	#Lambertview(ndec,bestfit.I)
	#show()
	
	chi2list = CRAToPy.extract_allsky_Lambert(CRmap,bestfit,NITER)
	
	#Lambertview(ndec,bestfit.I)
	#show()
	
	#nside = 64
	#nside_fine = 1024
	
	#mapHP = Lambert_to_healpy(ndec,nside,bestfit.I)
	
	#high_res_map = H.ud_grade(mapHP,nside_fine)
	#LMAX = 4*nside_fine
		
	#out  = H.anafast(high_res_map,alm=True,lmax=LMAX)	
	
	# remove dipole
	#out[1][H.sphtfunc.Alm.getidx(LMAX,1,0)] = 0.0
	#out[1][H.sphtfunc.Alm.getidx(LMAX,1,1)] = 0.0
			
	#high_res_map = H.alm2map(out[1],nside_fine,lmax=LMAX,verbose=False)
	#mapHP2 = H.ud_grade(high_res_map,nside)
		
	#H.mollview(mapHP)
	#show()
	#H.mollview(mapHP2)har
	#show()
	#exit(3)
	

	dec, RA = Lambert_pix2ang(ndec,np.arange(0,npix,1))
	
	lmax = 0
	
	harmonics = bestfit.I0
	
	for l in range(1,lmax+1) :
		for m in range(1,l+1) :
		
			print(l)
			Ylm = sph_harm(m, l, RA, np.pi/2 - dec) 
			
			f = np.imag(Ylm)*np.sqrt(2)
			g = np.real(Ylm)*np.sqrt(2)

			#print(sum(f*f)*4*np.pi/npix,sum(g*g)*4*np.pi/npix,sum(f*g)*4*np.pi/npix)

			a = sum(bestfit.I*f)*4*np.pi/npix
			b = sum(bestfit.I*g)*4*np.pi/npix
	
			harmonics += a*f + b*g
			
	Lambertview(ndec,harmonics)
	show()
	#pixels = Lambert_ang2pix(ndec,dec,deltaRA)
	#exit(3)
	
	# sph_harm(m, n, theta, phi)
	
	fit0 = CRAToPy.INA_Lambert(ndec,LATITUDE/180.*np.pi,THETAMAX/180.*np.pi) 	
	fit0.I = harmonics
	fit0.N = bestfit.N
	fit0.A = bestfit.A
	
	smoothdeltaI,smoothsignificance,trialmap = bestfit.tophat(CRmap,TOPHAT,fit0=fit0)
	
	deltaI_healpy = Lambert_to_healpy(ndec,64,smoothdeltaI)
	pretrial_healpy = Lambert_to_healpy(ndec,64,np.sqrt(smoothsignificance)*np.sign(smoothdeltaI))
	posttrial_healpy =  Lambert_to_healpy(ndec,64,np.sqrt(2)*erfinv(erf(np.sqrt(smoothsignificance)/np.sqrt(2))**trialmap)*np.sign(smoothdeltaI))
	
	maxidx = np.argmax(np.abs(pretrial_healpy))
	thetamax,phimax = H.pix2ang(64,maxidx)
	
	delta3= np.pi/2.0-LATITUDE/180.*np.pi-THETAMAX/180.*np.pi
	thetaFOV3 = np.zeros(721, dtype=np.double)		
	phiFOV3 = np.zeros(721, dtype=np.double)
	
	for i in range(0,720) :
		thetaFOV3[i] = delta3
		phiFOV3[i] = i/720.0*2.0*np.pi
	thetaFOV3[720] = delta3
	phiFOV3[720] = 0
	
	delta2= np.pi/2.0-LATITUDE/180.*np.pi+THETAMAX/180.*np.pi
	thetaFOV2 = np.zeros(721, dtype=np.double)		
	phiFOV2 = np.zeros(721, dtype=np.double)
		
	for i in range(0,720) :
		thetaFOV2[i] = delta2
		phiFOV2[i] = i/720.0*2.0*np.pi
	thetaFOV2[720] = delta2
	phiFOV2[720] = 0
	
	delta= 90
	thetaFOV = np.zeros(361, dtype=np.double)		
	phiFOV = np.zeros(361, dtype=np.double)
		
	for i in range(0,360) :
		thetaFOV[i] = delta*np.pi/180.0
		phiFOV[i] = i/360.0*2.0*np.pi
	thetaFOV[360] = delta*np.pi/180.0
	phiFOV[360] = 0
	
	pixlist = np.arange(0,H.nside2npix(64),1)
	
	theta,phi = H.pix2ang(64,pixlist)
	cond1 = theta >= np.pi/2.-LATITUDE/180.*np.pi+THETAMAX/180.*np.pi
	cond2 = theta <= np.pi/2.-LATITUDE/180.*np.pi-THETAMAX/180.*np.pi	
	
	mymax=max(np.amax(deltaI_healpy),-np.amin(deltaI_healpy))	
	digits = int(np.ceil(-np.log(mymax)/np.log(10)))+1
	mymax = np.around(mymax, decimals=digits)
	
	deltaI_healpy[pixlist[np.logical_or(cond1,cond2)]] = H.UNSEEN
	
	H.mollview(deltaI_healpy ,title= r'Auger 2017 $E>8$~EeV  : anisotropy ($45^\circ$ smoothing)',coord='C',cbar=True,rot=180,cmap=my_cmap,min=-mymax,max=mymax)
	
	H.projplot(thetaFOV,phiFOV,coord=['G','C'],linewidth=1,color="black",linestyle="dashed")
	H.projscatter(np.pi/2.,0.0,coord='G',color="black",marker='*',s=100,linewidth=0.0)
	H.projscatter(thetamax,phimax,coord='C',color="black",marker='x',s=100,linewidth=1.0)
	H.graticule()
	
	savefig("CR_Augerc.pdf",bbox_inches = 'tight')
	
	mymax=max(np.amax(pretrial_healpy),-np.amin(pretrial_healpy))	
	digits = int(np.ceil(-np.log(mymax)/np.log(10)))+2
	mymax2 = np.around(mymax, decimals=digits)
	
	pretrial_healpy[pixlist[np.logical_or(cond1,cond2)]] = H.UNSEEN
	
	H.mollview(pretrial_healpy ,title= r'Auger 2017 $E>8$~EeV : pre-trial significance ($45^\circ$ smoothing, $\sigma_{\rm max} = ' + format(mymax, '.2f') + '$)',coord='C',cbar=True,rot=180,cmap=my_cmap,min=-mymax2,max=mymax2)
	
	H.projplot(thetaFOV,phiFOV,coord=['G','C'],linewidth=1,color="black",linestyle="dashed")
	H.projscatter(np.pi/2.,0.0,coord='G',color="black",marker='*',s=100,linewidth=0.0)
	H.projscatter(thetamax,phimax,coord='C',color="black",marker='x',s=100,linewidth=1.0)
	H.graticule()

	savefig("pretrial_Augerc.pdf",bbox_inches = 'tight')
	
	mymax=max(np.amax(posttrial_healpy),-np.amin(posttrial_healpy))	
	digits = int(np.ceil(-np.log(mymax)/np.log(10)))+2
	mymax2 = np.around(mymax, decimals=digits)
	
	posttrial_healpy[pixlist[np.logical_or(cond1,cond2)]] = H.UNSEEN

	H.mollview(posttrial_healpy ,title= r'Auger 2017 $E>8$~EeV : post-trial significance ($45^\circ$ smoothing, $\sigma_{\rm max} = ' + format(mymax2, '.2f') + '$)',coord='C',cbar=True,rot=180,cmap=my_cmap,min=-mymax2,max=mymax2)
	
	H.projplot(thetaFOV,phiFOV,coord=['G','C'],linewidth=1,color="black",linestyle="dashed")	
	H.projscatter(np.pi/2.,0.0,coord='G',color="black",marker='*',s=100,linewidth=0.0)
	H.projscatter(thetamax,phimax,coord='C',color="black",marker='x',s=100,linewidth=1.0)
	H.graticule()
	
	savefig("posttrial_Augerc.pdf",bbox_inches = 'tight')

	
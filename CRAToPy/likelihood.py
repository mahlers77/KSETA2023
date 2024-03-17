import healpy as H
import numpy as np

from CRAToPy.coordinates import Pixel
from CRAToPy.coordinates import Pixel_Lambert
from CRAToPy.coordinates import Lambert_pix2ang
from CRAToPy.coordinates import EQ2LC_vector
from CRAToPy.datatools import Lambertview

from pylab import *

__all__ = [
	"extract_allsky",
	"extract_allsky_Lambert",
	"extract_dipole",
	"extract_dipole_Lambert",
	"dipole_sigma",
	"dipole_sigma_Lambert",
]

class INA_Lambert :
	
	def __init__(self,ndec,latitude,maxtheta,maskLC=None,maskEQ=None) :
	
		self.ndec = ndec
		self.latitude = latitude
		self.maxtheta = maxtheta
		
		self.npix = 2*ndec*ndec
		self.pixlist = np.arange(0,self.npix,1)
		self.ntimes = 2*ndec
		
		if maskLC is None :
			self.maskLC = self.get_maskLC()
		else :
			self.maskLC = maskLC
			
		if maskEQ is None :
			self.maskEQ = self.get_maskEQ()
		else :
			self.maskEQ = maskEQ
		
		self.pixlistLC = self.pixlist[self.maskLC > 0.0]
		self.pixlistEQ = self.pixlist[self.maskEQ > 0.0]
		
		self.EQ = Pixel_Lambert(ndec,self.pixlistEQ)
		self.LC = Pixel_Lambert(ndec,self.pixlistLC)

		self.I = np.ones(self.npix, dtype=np.double)
		self.N = np.zeros(self.ntimes, dtype=np.double)
		self.A = np.zeros(self.npix, dtype=np.double)
	
		self.Xmap, self.Ymap = self.get_dipolemaps()
		
	# initalize maps for dipole fit
	def get_dipolemaps(self) :
	
		dec, RA = Lambert_pix2ang(self.ndec,self.pixlist) 
	
		Xmap = np.cos(RA)*np.cos(dec)
		Ymap = np.sin(RA)*np.cos(dec)
		
		return Xmap, Ymap
		
	# bestfit solution for isotropic solution
	def firstguess(self,CRmap) :

		I0	= np.ones(self.npix, dtype=np.double)
		A0 = np.zeros(self.npix, dtype=np.double)
		N0 = np.zeros(self.ntimes, dtype=np.double)
		
		for timeidx in range(0,self.ntimes) :
			A0[self.pixlistLC] += CRmap[timeidx][self.pixlistLC]
			N0[timeidx] = sum(CRmap[timeidx][self.pixlistLC])
			
		#self.pixlistLC = self.pixlist[A0 > 0.5]	
		
		A0[self.pixlistLC] = A0[self.pixlistLC]/sum(A0[self.pixlistLC])
		
		self.I = I0
		self.A = A0
		self.N = N0
		
		self.I0 = I0
		self.A0 = A0
		self.N0 = N0
		
	# remove all m=0 spherical harmonics in relative intensity	
	def remove_m0(self,deltaI) :

		for i in range(0,self.ndec) :
			
			templist = np.arange(i*2*self.ndec,(i+1)*2*self.ndec,1)
			
			temp = sum(deltaI[templist])
			deltaI[templist] += -temp/2/self.ndec
		
		return deltaI
		
	# determine next iteration of dipole anisotropy for fixed N and A
	def iterate_dipole(self,CRmap) :

		Xmap = self.Xmap
		Ymap = self.Ymap
		
		ISOmap = np.ones(self.npix, dtype=np.double)
	
		Axx = 0.0
		Axy = 0.0
		Ayy = 0.0
		Bx = 0.0
		By = 0.0
		
		pixlistFOV = self.LC.pixel
		
		for timeidx in range(0,self.ntimes) :
			
			pixlistEQ  = self.LC.LC2EQ(timeidx)
			
			#pixlistEQ = self.EQ.pixel
			#pixlistFOV = self.EQ.EQ2LC(timeidx)
			
			Bx += sum((CRmap[timeidx][pixlistFOV] - self.N[timeidx]*self.A[pixlistFOV])*Xmap[pixlistEQ])
			By += sum((CRmap[timeidx][pixlistFOV] - self.N[timeidx]*self.A[pixlistFOV])*Ymap[pixlistEQ])
				
			Axx += sum(CRmap[timeidx][pixlistFOV]*Xmap[pixlistEQ]**2)
			Axy += sum(CRmap[timeidx][pixlistFOV]*Xmap[pixlistEQ]*Ymap[pixlistEQ])
			Ayy += sum(CRmap[timeidx][pixlistFOV]*Ymap[pixlistEQ]**2)

		delta0h = (Axy*By-Ayy*Bx)/(Axy**2-Axx*Ayy)
		delta6h = (Axy*Bx-Axx*By)/(Axy**2-Axx*Ayy)
		
		deltaI = delta0h*Xmap + delta6h*Ymap	
	
		self.I = ISOmap + deltaI	
		
		return delta0h, delta6h 
		
	# determine next iteration of allsky anisotropy for fixed N and A
	def iterate_allsky(self,CRmap) :
		
		deltaI = np.zeros(self.npix, dtype=np.double)
		ISOmap = np.ones(self.npix, dtype=np.double)
		
		pixlistLC = self.LC.pixel
		
		temp1 = np.zeros(self.npix, dtype=np.double)
		temp2 = np.zeros(self.npix, dtype=np.double)
		
		for timeidx in range(0,self.ntimes) :
	
			pixlistLC2EQ = self.LC.LC2EQ(timeidx)
		
			temp1[pixlistLC2EQ] += CRmap[timeidx][pixlistLC]
			temp2[pixlistLC2EQ] += self.N[timeidx]*self.A[pixlistLC]	
		
		pixlistPOS = self.pixlist[temp2 > 0.0]
		deltaI[pixlistPOS] = temp1[pixlistPOS]/temp2[pixlistPOS] - ISOmap[pixlistPOS]
		
		pixlistLARGE = self.pixlist[deltaI < -ISOmap]
		deltaI[pixlistLARGE] = -ISOmap[pixlistLARGE]
	
		deltaI = self.remove_m0(deltaI)
		
		self.I = ISOmap + deltaI		
	
	# determine next iteration of background rate N and relative acceptance A for fixed I
	def iterate_N_and_A(self,CRmap) :
	
		newN = np.zeros(self.ntimes, dtype=np.double)
		newA = np.zeros(self.npix, dtype=np.double)
		
		pixlistLC = self.LC.pixel
		
		for timeidx in range(0,self.ntimes) :
			
			pixlistLC2EQ =	self.LC.LC2EQ(timeidx)
			
			temp1 = sum(CRmap[timeidx][pixlistLC])
			temp2 = sum(self.A[pixlistLC]*self.I[pixlistLC2EQ])
			
			if temp2 > 0.0 :
				newN[timeidx] = temp1/temp2
			else :
				newN[timeidx] = 0.0
		
		self.N = newN
		
		#pixlistLC = self.LC.pixel
		
		temp1 = np.zeros(self.npix, dtype=np.double)
		temp2 = np.zeros(self.npix, dtype=np.double)
		
		for timeidx in range(0,self.ntimes) :
			
			pixlistLC2EQ = self.LC.LC2EQ(timeidx)
		
			temp1[pixlistLC] += CRmap[timeidx][pixlistLC]
			temp2[pixlistLC] += self.N[timeidx]*self.I[pixlistLC2EQ]
			
		pixlistPOS = self.pixlist[temp2 > 0.0]
		newA[pixlistPOS] = temp1[pixlistPOS]/temp2[pixlistPOS] 
		
		# cleaning up (normalizing A)
	
		temp = sum(newA[pixlistLC])	
		self.N *= temp				
		newA[pixlistLC] = newA[pixlistLC]/temp	
		
		self.A = newA
	
	def chi2(self,CRmap) :
		
		npix = self.npix
	
		significance = np.zeros(self.npix, dtype=np.double)
	
		temp1 = np.zeros(self.npix, dtype=np.double)
		temp2 = np.zeros(self.npix, dtype=np.double)
	
		for timeidx in range(0,self.ntimes) :
			
			pixlistLC = self.LC.pixel
			pixlistFOV = self.LC.LC2EQ(timeidx)
			
			#temp1[pixlistFOV] = self.A[pixlistLC]*self.N[timeidx]*self.I0[pixlistFOV]
			temp1[pixlistFOV] = self.A0[pixlistLC]*self.N0[timeidx]*self.I0[pixlistFOV]
			temp2[pixlistFOV] = self.A[pixlistLC] *self.N[timeidx] *self.I[pixlistFOV] 
			
			cond1 = temp1[pixlistFOV] > 0.0
			cond2 = temp2[pixlistFOV] > 0.0
			cond3 = CRmap[timeidx][pixlistLC] > 0.0
		
			pixlistPOS1 = pixlistFOV[cond1]	
			pixlistPOS2 = pixlistFOV[cond2]	
		
			pixlistPOS = pixlistFOV[np.logical_and(np.logical_and(cond1,cond2),cond3)]	
			pixlistLCPOS = pixlistLC[np.logical_and(np.logical_and(cond1,cond2),cond3)]	
	
			significance[pixlistPOS1] +=  2.0*temp1[pixlistPOS1]
			significance[pixlistPOS2] += -2.0*temp2[pixlistPOS2]

			significance[pixlistPOS] +=  2.0*CRmap[timeidx][pixlistLCPOS]*np.log(temp2[pixlistPOS]/temp1[pixlistPOS])
		
		return sum(significance)
		
	# define the field of fiew in the local coordinate system by maximum zenith cut
	def get_maskLC(self) :
	
		maskLC = np.ones(self.npix, dtype=np.double)
		
		dec, RA = Lambert_pix2ang(self.ndec,self.pixlist) 
	
		vx = np.cos(RA)*np.cos(dec)
		vy = np.sin(RA)*np.cos(dec)
		vz = np.sin(dec)
	
		vLCx,vLCy,vLCz = EQ2LC_vector(vx,vy,vz,0.0,self.latitude)
	
		vLC = np.transpose([vLCx,vLCy,vLCz])
		theta, phi = H.vec2ang(vLC)
	
		maskLC[theta > self.maxtheta] = 0.0
		
		return maskLC
	
	## define the field of fiew in the equatorial coordinate system by maximum zenith cut
	def get_maskEQ(self) :
	
		maskEQ = np.ones(self.npix, dtype=np.double)
		
		dec, RA = Lambert_pix2ang(self.ndec,self.pixlist) 
		
		cond1 = dec < self.latitude - self.maxtheta
		cond2 = dec > self.latitude + self.maxtheta
	
		maskEQ[np.logical_or(cond1,cond2)] = 0.0
		
		return maskEQ
	

	def tophat(self,CRmap,tophat,fit0=None) :
	
		npix = self.npix
	
		pixlistEQ = self.EQ.pixel
		
		mu = np.zeros(self.npix, dtype=np.double)
		mu0 = np.zeros(self.npix, dtype=np.double)
		CRbin = np.zeros(self.npix, dtype=np.double)
	
		for timeidx in range(0,self.ntimes) :
			
			pixlistEQ2LC = self.EQ.EQ2LC(timeidx)
		
			if fit0 is None :
				#mu0[pixlistEQ] += self.I0[pixlistEQ]*self.A0[pixlistEQ2LC]*self.N0[timeidx] # achtung
				mu0[pixlistEQ] += self.I0[pixlistEQ]*self.A[pixlistEQ2LC]*self.N[timeidx] 
			else :
				mu0[pixlistEQ] += fit0.I[pixlistEQ]*fit0.A[pixlistEQ2LC]*fit0.N[timeidx]
				
			mu[pixlistEQ]  += self.I[pixlistEQ]*self.A[pixlistEQ2LC]*self.N[timeidx] 
			CRbin[pixlistEQ] += CRmap[timeidx][pixlistEQ2LC]
	
		smoothsignificance = np.zeros(self.npix, dtype=np.double)
	
		smoothdeltaI = np.zeros(self.npix, dtype=np.double)
		trialmap = np.ones(self.npix, dtype=np.double)
	
		dec, RA = Lambert_pix2ang(self.ndec,self.pixlist) 
		#dec = np.arcsin(2.*((self.pixlist // (2*self.ndec)) + 0.5)/self.ndec-1.0)
		#RA  = ((self.pixlist % (2*self.ndec)) + 0.5)*2.*np.pi/(2*self.ndec)
		
		vx = np.cos(RA)*np.cos(dec)
		vy = np.sin(RA)*np.cos(dec)
		vz = np.sin(dec)
		
		ctophat = np.cos(tophat)
		
		for j in pixlistEQ :
		
			#print(j)
		
			if tophat == 0.0 :
				set2 = [j]
				
			else :
				set = self.pixlist[vx[j]*vx + vy[j]*vy + vz[j]*vz >= ctophat]		
				set2 = set[self.maskEQ[set] > 0.0]
			
			trialmap[j] = max(1.0,len(pixlistEQ)/len(set2))
			
			#trialmap[j] = len(pixlistEQ)/len(set2)
			
			musum = sum(mu[set2])
			mu0sum = sum(mu0[set2])
			CRbinsum= sum(CRbin[set2])
				
			if musum > 0.0 and mu0sum > 0.0 : 
				smoothdeltaI[j] = musum/mu0sum-1.0 
			
				smoothsignificance[j] = -2.0*musum + 2.0*mu0sum	
				if CRbinsum > 0.0 :
					smoothsignificance[j] += 2.0*CRbinsum*np.log(musum/mu0sum)
		
			if smoothsignificance[j] < 0.0 :
				smoothsignificance[j] = 0.0
		
		return smoothdeltaI,smoothsignificance,trialmap
		
class INA :
	
	def __init__(self,nside,ntimes,latitude,maxtheta,maskLC=None,maskEQ=None) :
	
		self.nside = nside
		self.ntimes = ntimes
		self.latitude = latitude
		self.maxtheta = maxtheta
		
		self.npix = H.nside2npix(nside)
		self.pixlist = np.arange(0,self.npix,1)
		
		if maskLC is None :
			self.maskLC = self.get_maskLC()
		else :
			self.maskLC = maskLC
			
		if maskEQ is None :
			self.maskEQ = self.get_maskEQ()
		else :
			self.maskEQ = maskEQ
		
		self.pixlistLC = self.pixlist[self.maskLC > 0.0]
		self.pixlistEQ = self.pixlist[self.maskEQ > 0.0]
		
		self.EQ = Pixel(nside,self.pixlistEQ,latitude)
		self.LC = Pixel(nside,self.pixlistLC,latitude)

		self.I = np.ones(self.npix, dtype=np.double)
		self.N = np.zeros(ntimes, dtype=np.double)
		self.A = np.zeros(self.npix, dtype=np.double)
	
		self.Xmap, self.Ymap = self.get_dipolemaps()
		
	# initalize maps for dipole fit
	def get_dipolemaps(self) :
	
		alm = np.array([0.0+0.0j ,0.0+0.0j, -1.0+0.0j])
		Xmap = H.alm2map(alm,self.nside,lmax=1,verbose=False)/np.sqrt(3./2./np.pi)
		
		alm = np.array([0.0+0.0j, 0.0+0.0j, 0.0+1.0j])
		Ymap = H.alm2map(alm,self.nside,lmax=1,verbose=False)/np.sqrt(3./2./np.pi)
		
		return Xmap, Ymap
		
	# bestfit solution for isotropic solution
	def firstguess(self,CRmap) :

		I0	= np.ones(self.npix, dtype=np.double)
		A0 = np.zeros(self.npix, dtype=np.double)
		N0 = np.zeros(self.ntimes, dtype=np.double)
		
		for timeidx in range(0,self.ntimes) :
			A0[self.pixlistLC] += CRmap[timeidx][self.pixlistLC]
			N0[timeidx] = sum(CRmap[timeidx][self.pixlistLC])
			
		A0 = A0/sum(A0)
		
		self.I = I0
		self.A = A0
		self.N = N0
		
		self.I0 = I0
		self.A0 = A0
		self.N0 = N0
		
	# remove all m=0 spherical harmonics in relative intensity	
	def remove_m0(self,deltaI,nsideFOV=1024) :

		high_res_map = H.ud_grade(deltaI,nsideFOV)
		LMAX = 4*nsideFOV
		
		out  = H.anafast(high_res_map,alm=True,lmax=LMAX)	
		for i in range(0,LMAX+1) :
	
			index = H.sphtfunc.Alm.getidx(LMAX,i,0)
			out[1][index] = 0.0
			
		high_res_map = H.alm2map(out[1],nsideFOV,lmax=LMAX,verbose=False)
		deltaI = H.ud_grade(high_res_map,self.nside)
		
		return deltaI
		
	# determine next iteration of dipole anisotropy for fixed N and A
	def iterate_dipole(self,CRmap) :

		Xmap = self.Xmap
		Ymap = self.Ymap
		
		ISOmap = np.ones(self.npix, dtype=np.double)
	
		Axx = 0.0
		Axy = 0.0
		Ayy = 0.0
		Bx = 0.0
		By = 0.0
		
		for timeidx in range(0,self.ntimes) :
		
			hour = (timeidx+0.5)/self.ntimes*np.pi*2
			
			pixlistFOV = self.LC.pixel
			pixlistEQ  = self.LC.LC2EQ(hour)
			
			Bx += sum((CRmap[timeidx][pixlistFOV] - self.N[timeidx]*self.A[pixlistFOV])*Xmap[pixlistEQ])
			By += sum((CRmap[timeidx][pixlistFOV] - self.N[timeidx]*self.A[pixlistFOV])*Ymap[pixlistEQ])
				
			Axx += sum(CRmap[timeidx][pixlistFOV]*Xmap[pixlistEQ]**2)
			Axy += sum(CRmap[timeidx][pixlistFOV]*Xmap[pixlistEQ]*Ymap[pixlistEQ])
			Ayy += sum(CRmap[timeidx][pixlistFOV]*Ymap[pixlistEQ]**2)

	
		delta0h = (Axy*By-Ayy*Bx)/(Axy**2-Axx*Ayy)
		delta6h = (Axy*Bx-Axx*By)/(Axy**2-Axx*Ayy)
		
		deltaI = delta0h*Xmap + delta6h*Ymap	
	
		self.I = ISOmap + deltaI	
		
		return delta0h, delta6h 
		
	# determine next iteration of allsky anisotropy for fixed N and A
	def iterate_allsky(self,CRmap,nsideFOV=1024) :
		
		deltaI = np.zeros(self.npix, dtype=np.double)
		ISOmap = np.ones(self.npix, dtype=np.double)
		
		pixlistEQ = self.EQ.pixel
		
		temp1 = np.zeros(self.npix, dtype=np.double)
		temp2 = np.zeros(self.npix, dtype=np.double)
		
		for timeidx in range(0,self.ntimes) :
		
			hour = (timeidx+0.5)/self.ntimes*np.pi*2 
	
			pixlistEQ2LC = self.EQ.EQ2LC(hour)
		
			temp1[pixlistEQ] += CRmap[timeidx][pixlistEQ2LC]
			temp2[pixlistEQ] += self.N[timeidx]*self.A[pixlistEQ2LC]	
		
		pixlistPOS = self.pixlist[np.logical_and(temp1 > 0.0,temp2 > 0.0)]
		deltaI[pixlistPOS] = temp1[pixlistPOS]/temp2[pixlistPOS] - ISOmap[pixlistPOS]
		
		pixlistLARGE = self.pixlist[deltaI < -ISOmap]
		deltaI[pixlistLARGE] = -ISOmap[pixlistLARGE]
	
		deltaI = self.remove_m0(deltaI,nsideFOV)
			
		self.I = ISOmap + deltaI		
	
	# determine next iteration of background rate N and relative acceptance A for fixed I
	def iterate_N_and_A(self,CRmap) :
	
		newN = np.zeros(self.ntimes, dtype=np.double)
		newA = np.zeros(self.npix, dtype=np.double)
		
		pixlistLC = self.LC.pixel
		#pixlistLC2EQ =	self.EQ.pixel
		
		for timeidx in range(0,self.ntimes) :

			hour = (timeidx+0.5)/self.ntimes*np.pi*2 
			
			pixlistLC2EQ =	self.LC.LC2EQ(hour)
			#pixlistLC =	self.EQ.EQ2LC(hour)
			
			temp1 = sum(CRmap[timeidx][pixlistLC])
			temp2 = sum(self.A[pixlistLC]*self.I[pixlistLC2EQ])
			
			if temp1 > 0.0 and temp2 > 0.0:
				newN[timeidx] = temp1/temp2
			else : 
				newN[timeidx] = self.N[timeidx]
				
		self.N = newN
		
		pixlistLC = self.LC.pixel
		
		temp1 = np.zeros(len(pixlistLC), dtype=np.double)
		temp2 = np.zeros(len(pixlistLC), dtype=np.double)
		
		for timeidx in range(0,self.ntimes) :
		
			hour = (timeidx+0.5)/self.ntimes*np.pi*2
			
			pixlistLC2EQ = self.LC.LC2EQ(hour)
		
			temp1[pixlistLC] += CRmap[timeidx][pixlistLC]
			temp2[pixlistLC] += self.N[timeidx]*self.I[pixlistLC2EQ]
			
		pixlistPOS = pixlistLC[np.logical_and(temp1 > 0.0,temp2 > 0.0)]
		
		newA[pixlistPOS] = temp1[pixlistPOS]/temp2[pixlistPOS] 
		
		# cleaning up (normalizing A)
	
		temp = sum(newA)	
		self.N *= temp				
		newA = newA/temp	
		
		self.A = newA
	
	def chi2(self,CRmap) :
		
		npix = self.npix
	
		significance = np.zeros(self.npix, dtype=np.double)
	
		temp1 = np.zeros(self.npix, dtype=np.double)
		temp2 = np.zeros(self.npix, dtype=np.double)
	
		for timeidx in range(0,self.ntimes) :
		
			hour = (timeidx+0.5)/self.ntimes*np.pi*2 
			
			#pixlistFOV = self.EQ.pixel
			#pixlistLC = self.EQ.EQ2LC(hour)
		
			pixlistLC = self.LC.pixel
			pixlistFOV = self.LC.LC2EQ(hour)
			
			#temp1[pixlistFOV] = self.A[pixlistLC]*self.N[timeidx]*self.I0[pixlistFOV]
			temp1[pixlistFOV] = self.A0[pixlistLC]*self.N0[timeidx]*self.I0[pixlistFOV]
			temp2[pixlistFOV] = self.A[pixlistLC]*self.N[timeidx]*self.I[pixlistFOV] 
		
			cond1 = temp1[pixlistFOV] > 0.0
			cond2 = temp2[pixlistFOV] > 0.0
			#cond3 = CRmap[timeidx][pixlistLC] > 0.0
		
			#pixlistPOS = pixlistFOV[np.logical_and(np.logical_and(cond1,cond2),cond3)]	
			#pixlistLCPOS = pixlistLC[np.logical_and(np.logical_and(cond1,cond2),cond3)]	
			
			pixlistPOS = pixlistFOV[np.logical_and(cond1,cond2)]	
			pixlistLCPOS = pixlistLC[np.logical_and(cond1,cond2)]	
			
			#significance[pixlistFOV] +=  2.0*temp1[pixlistFOV]
			#significance[pixlistFOV] += -2.0*temp2[pixlistFOV]
			
			significance[pixlistPOS] +=  2.0*temp1[pixlistPOS]
			significance[pixlistPOS] += -2.0*temp2[pixlistPOS]
			significance[pixlistPOS] +=  2.0*CRmap[timeidx][pixlistLCPOS]*np.log(temp2[pixlistPOS]/temp1[pixlistPOS])
		
		return sum(significance)
		
	# define the field of fiew in the local coordinate system by maximum zenith cut
	def get_maskLC(self) :
	
		maskLC = np.ones(self.npix, dtype=np.double)
		
		theta, phi = H.pix2ang(self.nside,self.pixlist)
		
		maskLC[theta > self.maxtheta] = 0.0
		
		return maskLC
	
	## define the field of fiew in the equatorial coordinate system by maximum zenith cut
	def get_maskEQ(self) :
	
		maskEQ = np.ones(self.npix, dtype=np.double)
		
		theta, phi = H.pix2ang(self.nside,self.pixlist)
		
		cond1 = theta < np.pi/2. - self.latitude - self.maxtheta
		cond2 = theta > np.pi/2. - self.latitude + self.maxtheta
	
		maskEQ[np.logical_or(cond1,cond2)] = 0.0
		
		return maskEQ
	
	def tophat(self,CRmap,tophat,fit0=None) :
	
		npix = self.npix
		nside = self.nside
	
		pixlistEQ = self.EQ.pixel
		
		mu = np.zeros(self.npix, dtype=np.double)
		mu0 = np.zeros(self.npix, dtype=np.double)
		CRbin = np.zeros(self.npix, dtype=np.double)
	
		for timeidx in range(0,self.ntimes) :
		
			hour = (timeidx+0.5)/self.ntimes*np.pi*2 
			
			pixlistEQ2LC = self.EQ.EQ2LC(hour)
		
			if fit0 is None :
				mu0[pixlistEQ] += self.I0[pixlistEQ]*self.A[pixlistEQ2LC]*self.N[timeidx]
			else :
				mu0[pixlistEQ] += fit0.I[pixlistEQ]*fit0.A[pixlistEQ2LC]*fit0.N[timeidx]
				
			mu[pixlistEQ]  += self.I[pixlistEQ]*self.A[pixlistEQ2LC]*self.N[timeidx] 
			CRbin[pixlistEQ] += CRmap[timeidx][pixlistEQ2LC]
	
		smoothsignificance = np.zeros(self.npix, dtype=np.double)
	
		smoothdeltaI = np.zeros(self.npix, dtype=np.double)
		trialmap = np.zeros(self.npix, dtype=np.double)
	
		for j in pixlistEQ :
		
			vec = H.pix2vec(self.nside,j)
			set = H.query_disc(self.nside,vec,tophat)
		
			set2 = set[self.maskEQ[set] > 0.0]
			
			#trialmap[j] = max(1.0,len(pixlistEQ)/len(set2))
			
			trialmap[j] = len(pixlistEQ)/len(set2)
			
			musum = sum(mu[set2])
			mu0sum = sum(mu0[set2])
			CRbinsum= sum(CRbin[set2])
				
			#if CRbinsum > 0.0 and musum > 0.0 and mu0sum > 0.0 : 
			if musum > 0.0 and mu0sum > 0.0 : 
				smoothdeltaI[j] = musum/mu0sum-1.0 
			
				smoothsignificance[j] = -2.0*musum + 2.0*mu0sum	
				smoothsignificance[j] += 2.0*CRbinsum*np.log(musum/mu0sum)
		
			if smoothsignificance[j] < 0.0 :
				smoothsignificance[j] = 0.0
		
		return smoothdeltaI,smoothsignificance,trialmap

def extract_allsky_Lambert(CRmap,fit,niteration,chi2flag=True) :

	chi2list = []
	
	for iteration in range(0,niteration) :

		fit.iterate_allsky(CRmap)
		
		temp = fit.chi2(CRmap)
		print(2*iteration,temp)
	
		fit.iterate_N_and_A(CRmap)
		
		if chi2flag == True :
			#significancemap = significance(fit,CRmap) 
			#ndof = len(fit.pixlistEQ)
			#ndof = len(fit.pixlistEQ)/fit.npix*4*np.pi/(2*np.pi*(1.0 - np.cos(FWHM)))
			
			temp = fit.chi2(CRmap) 
			print(2*iteration+1,temp)
			
			chi2list.append([iteration,temp])
	
	if chi2flag == True :
		return np.array(chi2list)
	else :
		return
		
def extract_allsky(CRmap,fit,niteration,nsideFOV = 1024,chi2flag=True) :

	chi2list = []
	
	for iteration in range(0,niteration) :

		fit.iterate_allsky(CRmap,nsideFOV=nsideFOV)
		
		#temp = fit.chi2(CRmap)
		#print(iteration,temp)
			
		fit.iterate_N_and_A(CRmap)
		
		if chi2flag == True :
			#significancemap = significance(fit,CRmap) 
			#ndof = len(fit.pixlistEQ)
			#ndof = len(fit.pixlistEQ)/fit.npix*4*np.pi/(2*np.pi*(1.0 - np.cos(FWHM)))
			
			temp = fit.chi2(CRmap) 
			print(iteration,temp)
			
			chi2list.append([iteration,temp])
	
	if chi2flag == True :
		return np.array(chi2list)
	else :
		return
	
def extract_dipole(CRmap,fit,niteration,chi2flag=True) :
	
	chi2list = []
	
	for iteration in range(0,niteration) :
		
		delta0h, delta6h = fit.iterate_dipole(CRmap)
		
		#temp = fit.chi2(CRmap)
		#print(iteration,temp)
		
		fit.iterate_N_and_A(CRmap)
		
		if chi2flag == True :
			#significancemap = significance(fit,CRmap) 
	
			temp = fit.chi2(CRmap)
			print(iteration,temp)
			chi2list.append([iteration,temp])
			
	Amp = np.sqrt(delta0h**2 + delta6h**2)	
	
	phase = (np.arctan2(delta6h,delta0h)/np.pi*180.0 + 360.0) % 360.0
	
	ddelta0h,ddelta6h = dipole_sigma(CRmap,fit,delta0h,delta6h) 
		
	dAmp= np.sqrt(delta0h**2/Amp**2*ddelta0h**2 + delta6h**2/Amp**2*ddelta6h**2)

	dphase = np.sqrt(ddelta0h**2*delta6h**2 + ddelta6h**2*delta0h**2)/Amp**2/np.pi*180.
	
	#Amp = np.sqrt(delta0h**2 + delta6h**2)	
	
	#ddelta0h,ddelta6h = dipole_sigma(CRmap,fit,delta0h,delta6h) 
		
	#dAmp= np.sqrt(delta0h**2/Amp**2*ddelta0h**2 + delta6h**2/Amp**2*ddelta6h**2)

	#phase = (np.arctan2(delta6h,delta0h)/np.pi*180.0 + 360.0) % 360.0
	
	if chi2flag == True :
		return delta0h,ddelta0h,delta6h,ddelta6h,Amp,dAmp,phase,dphase,np.array(chi2list)
	else :
		return delta0h,ddelta0h,delta6h,ddelta6h,Amp,dAmp,phase,dphase

def extract_dipole_Lambert(CRmap,fit,niteration,chi2flag=True,verbose=False) :
	
	chi2list = []
	
	for iteration in range(0,niteration) :
		
		delta0h, delta6h = fit.iterate_dipole(CRmap)
		
		temp = fit.chi2(CRmap)
		if verbose :
			print(2*iteration,temp)
		
		fit.iterate_N_and_A(CRmap)
		
		if chi2flag == True :
			#significancemap = significance(fit,CRmap) 
	
			temp = fit.chi2(CRmap)
			if verbose :
				print(2*iteration+1,temp)
			chi2list.append([iteration,temp])
			
	Amp = np.sqrt(delta0h**2 + delta6h**2)	
	
	phase = (np.arctan2(delta6h,delta0h)/np.pi*180.0 + 360.0) % 360.0
	
	ddelta0h,ddelta6h = dipole_sigma_Lambert(CRmap,fit,delta0h,delta6h) 
		
	dAmp= np.sqrt(delta0h**2/Amp**2*ddelta0h**2 + delta6h**2/Amp**2*ddelta6h**2)

	dphase = np.sqrt(ddelta0h**2*delta6h**2 + ddelta6h**2*delta0h**2)/Amp**2/np.pi*180.
	
	#return delta0h,delta6h,Amp,phase
	
	if chi2flag == True :
		return delta0h,ddelta0h,delta6h,ddelta6h,Amp,dAmp,phase,dphase,np.array(chi2list)
	else :
		return delta0h,ddelta0h,delta6h,ddelta6h,Amp,dAmp,phase,dphase
		
def dipole_sigma(CRmap,fit,delta0h,delta6h) :

	nside = fit.nside
	npix = fit.npix
	pixlist = fit.pixlist

	theta,phi = H.pix2ang(nside,pixlist)
		
	pixlistFOV = fit.LC.pixel
	
	LCvx,LCvy,LCvz = H.pix2vec(nside,pixlistFOV)
	
	ISOmap = np.ones(npix, dtype=np.double)
		
	Xmap = np.cos(phi)*np.sin(theta)
	Ymap = np.sin(phi)*np.sin(theta)
	
	temp0h0h = temp6h6h = temp0h6h=0.0
			
	tempX = np.zeros(fit.ntimes, dtype=np.double)	
	tempY = np.zeros(fit.ntimes, dtype=np.double)	
	tempN = np.zeros(fit.ntimes, dtype=np.double)	
		
	for timeidx in range(0,fit.ntimes) :
		
		hour = (timeidx+0.5)/fit. ntimes*np.pi*2 
		pixlistEQ = fit.LC.LC2EQ(hour)
		
		temp0h0h += sum(CRmap[timeidx][pixlistFOV]*(Xmap[pixlistEQ]*delta0h)**2/(fit.I[pixlistEQ])**2)
		temp6h6h += sum(CRmap[timeidx][pixlistFOV]*(Ymap[pixlistEQ]*delta6h)**2/(fit.I[pixlistEQ])**2)
		temp0h6h += sum(CRmap[timeidx][pixlistFOV]*(Xmap[pixlistEQ]*delta0h)*(Ymap[pixlistEQ]*delta6h)/(fit.I[pixlistEQ])**2)
		
		tempX[timeidx] += sum(fit.N[timeidx]*fit.A[pixlistFOV]*Xmap[pixlistEQ]*delta0h)
		tempY[timeidx] += sum(fit.N[timeidx]*fit.A[pixlistFOV]*Ymap[pixlistEQ]*delta6h)
		tempN[timeidx] += sum(CRmap[timeidx][pixlistFOV])
	
	for timeidx in range(0,fit.ntimes) :
		if tempN[timeidx] > 0.0 :
			temp0h0h += - tempX[timeidx]**2/tempN[timeidx]
			temp6h6h += - tempY[timeidx]**2/tempN[timeidx]
			temp0h6h += - tempX[timeidx]*tempY[timeidx]/tempN[timeidx]
				
	ddelta0h = np.sqrt(delta0h**2*temp6h6h/(temp0h0h*temp6h6h - temp0h6h**2))
	ddelta6h = np.sqrt(delta6h**2*temp0h0h/(temp0h0h*temp6h6h - temp0h6h**2))
	
	return ddelta0h,ddelta6h		
	
def dipole_sigma_Lambert(CRmap,fit,delta0h,delta6h) :

	ndec = fit.ndec
	npix = fit.npix
	pixlist = fit.pixlist

	dec, RA = Lambert_pix2ang(ndec,pixlist) 
	
	Xmap = np.cos(RA)*np.cos(dec)
	Ymap = np.sin(RA)*np.cos(dec)
		
	pixlistFOV = fit.LC.pixel
	
	ISOmap = np.ones(npix, dtype=np.double)
	
	temp0h0h = temp6h6h = temp0h6h=0.0
			
	tempX = np.zeros(fit.ntimes, dtype=np.double)	
	tempY = np.zeros(fit.ntimes, dtype=np.double)	
	tempN = np.zeros(fit.ntimes, dtype=np.double)	
		
	for timeidx in range(0,fit.ntimes) :
		
		#hour = (timeidx+0.5)/fit. ntimes*np.pi*2 
		pixlistEQ = fit.LC.LC2EQ(timeidx)
		
		temp0h0h += sum(CRmap[timeidx][pixlistFOV]*(Xmap[pixlistEQ]*delta0h)**2/(fit.I[pixlistEQ])**2)
		temp6h6h += sum(CRmap[timeidx][pixlistFOV]*(Ymap[pixlistEQ]*delta6h)**2/(fit.I[pixlistEQ])**2)
		temp0h6h += sum(CRmap[timeidx][pixlistFOV]*(Xmap[pixlistEQ]*delta0h)*(Ymap[pixlistEQ]*delta6h)/(fit.I[pixlistEQ])**2)
		
		tempX[timeidx] += sum(fit.N[timeidx]*fit.A[pixlistFOV]*Xmap[pixlistEQ]*delta0h)
		tempY[timeidx] += sum(fit.N[timeidx]*fit.A[pixlistFOV]*Ymap[pixlistEQ]*delta6h)
		tempN[timeidx] += sum(CRmap[timeidx][pixlistFOV])
	
	for timeidx in range(0,fit.ntimes) :
		if tempN[timeidx] > 0.0 :
			temp0h0h += - tempX[timeidx]**2/tempN[timeidx]
			temp6h6h += - tempY[timeidx]**2/tempN[timeidx]
			temp0h6h += - tempX[timeidx]*tempY[timeidx]/tempN[timeidx]
				
	ddelta0h = np.sqrt(delta0h**2*temp6h6h/(temp0h0h*temp6h6h - temp0h6h**2))
	ddelta6h = np.sqrt(delta6h**2*temp0h0h/(temp0h0h*temp6h6h - temp0h6h**2))
	
	return ddelta0h,ddelta6h	
import healpy as H
import numpy as np

__all__ = [
	"Lambert_ang2pix",
	"Lambert_pix2ang",
	"LC2EQ_vector",
	"EQ2LC_vector",
	"Lambert_to_healpy",
]

def Lambert_to_healpy(ndec,nside,inmap) :

	npix = 2*ndec*ndec
	dec, RA = Lambert_pix2ang(ndec,np.arange(0,npix,1))
	
	nsideHIGH = 1024
	npix_healpy = H.nside2npix(nsideHIGH)
	pixlist_healpy = np.arange(0,npix_healpy,1)
	theta,phi = H.pix2ang(nsideHIGH,pixlist_healpy)
	
	pixlist = Lambert_ang2pix(ndec,np.pi/2-theta,phi)
	#tempmap = inmap[pixlist]
	tempmap = H.ud_grade(inmap[pixlist],nside)
	return tempmap # *np.cos(np.pi/2-theta)/np.cos(dec[pixlist])

def Lambert_ang2pix_old(ndec,dec,RA) :

	x = np.floor(RA/np.pi/2.*2*ndec)
	y = np.floor((np.sin(dec) + 1.0)/2.0*ndec)
	
	x[x == 2*ndec] = 2*ndec-1
	y[y == ndec] = ndec-1

	a = np.array(x + y*2*ndec)
	return a.astype('int')

def Lambert_ang2pix(ndec,dec,RA) :

	x = (np.digitize(RA,np.linspace(0.0, 2.*np.pi, num=2*ndec+1))-1) % (2*ndec)
	y = (np.digitize(np.sin(dec),np.linspace(-1.0,1.0, num=ndec+1))-1) % ndec
	
	return x + y*2*ndec
	
def Lambert_pix2ang(ndec,pixel) :

	dec = np.arcsin(2.*((pixel // (2*ndec)) + 0.5)/ndec-1.0)
	RA  = ((pixel % (2*ndec)) + 0.5)*2.*np.pi/(2*ndec)

	return dec, RA
	
def LC2EQ_vector(vLCx,vLCy,vLCz,hour,latitude) :

	"""
	Returns the coordinates vEQx, vEQy and vEQz of a unit vector in the equatorial 
	coordinate system given its components vx, vy and vz in the local coordinate 
	system, hour angle and detector latitiute
	
	Parameters
	----------
	vx,vy,vz : floats, scalar or array-like
		components of unit vector in local coodinate system
	hour : floats, scalar or array-like
		hour angle in radians of observation in local coordinates, i.e. the RA angle 
		of the meridian at the time of observation
	latitude : floats, scalar or array-like
		latitude of the observatory in radians
		
	Returns	
	-------
	vEQx,vEQy,vEQz : floats, scalar or array-like
		components of unit vector in equatorial coodinate system
    
	Notes
	-----
	A unit vector in the local coordinate system is defined as
	
	v = (cos(phi)*sin(theta),-sin(phi)*sin(theta),cos(theta)), 
	
	where theta is the zenith angle and phi is the azimuth angle measured from North
	and increasing towards East. The unit vector in the equatorial coordinate system 
	is defined as:
	
	vEQ = (cos(RA)*cos(dec),sin(RA)*cos(dec),sin(dec))
	"""
	
	vecLC = np.array([vLCx,vLCy,vLCz])
	
	ch = np.cos(hour)
	sh = np.sin(hour)
	
	cl = np.cos(latitude)
	sl = np.sin(latitude)
	
	# rotation matrix
	R = np.array([[-ch*sl,-sh*sl,cl],[sh,-ch,0],[ch*cl,cl*sh,sl]])
	
	# inverse rotation matrix
	RT = np.transpose(R)
	
	#rotation from local frame to Equatorial (ra,dec)
	vecEQ = np.dot(RT,vecLC)
	
	return vecEQ[0],vecEQ[1],vecEQ[2]
		
def EQ2LC_vector(vEQx,vEQy,vEQz,hour,latitude) :

	"""
	Returns the coordinates vLCx, vLCy and vLCz of a unit vector in the local 
	coordinate system given its components vEQx, vEQy and vEQz in the equatorial 
	coordinate system, hour angle and detector latitiute
	
	Parameters
	----------
	vEQx,vEQy,vEQz : floats, scalar or array-like
		components of unit vector in equatorial coodinate system
	hour : floats, scalar or array-like
		hour angle in radians of observation in local coordinates, i.e. the RA angle 
		of the meridian at the time of observation
	latitude : floats, scalar or array-like
		latitude of the observatory in radians
		
	Returns	
	-------
	vEQx,vEQy,vEQz : floats, scalar or array-like
		components of unit vector in equatorial coodinate system
    
	Notes
	-----
	A unit vector in the local coordinate system is defined as
	
	vLC = (cos(phi)*sin(theta),-sin(phi)*sin(theta),cos(theta)), 
	
	where theta is the zenith angle and phi is the azimuth angle measured from North
	and increasing towards East. The unit vector in the equatorial coordinate system 
	is defined as:
	
	vEQ = (cos(RA)*cos(dec),sin(RA)*cos(dec),sin(dec))
	"""
	
	vecEQ = np.array([vEQx,vEQy,vEQz])
	
	ch = np.cos(hour)
	sh = np.sin(hour)
	
	cl = np.cos(latitude)
	sl = np.sin(latitude)
	
	# rotation matrix
	R = np.array([[-ch*sl,-sh*sl,cl],[sh,-ch,0],[ch*cl,cl*sh,sl]])
	
	#rotation from local frame to Equatorial (ra,dec)
	vecLC = np.dot(R,vecEQ)
	
	return vecLC[0],vecLC[1],vecLC[2]
	
class Pixel :

	"""
	Pixel class facilitating easy transformations between local and equatorial.
	Uses healpix tiling of the sphere implemented via healpy.	
	"""
	
	def __init__(self,nside,pixel,latitude):

		self.nside = nside
		self.pixel = pixel
		self.latitude = latitude
		
		self.vector = H.pix2vec(nside,pixel)
		
	def LC2EQ(self,hour) :
	
		vectorEQ = LC2EQ_vector(self.vector[0],self.vector[1],self.vector[2],hour,self.latitude)
	
		return H.vec2pix(self.nside,vectorEQ[0],vectorEQ[1],vectorEQ[2])
		
	def EQ2LC(self,hour) :
	
		vectorLC = EQ2LC_vector(self.vector[0],self.vector[1],self.vector[2],hour,self.latitude)
	
		return H.vec2pix(self.nside,vectorLC[0],vectorLC[1],vectorLC[2])
		
class Pixel_Lambert :

	"""
	Pixel class facilitating easy transformations between local and equatorial.
	Uses healpix tiling of the sphere implemented via healpy.	
	"""
	
	def __init__(self,ndec,pixel):

		self.ndec = ndec
		self.pixel = pixel
		
	def LC2EQ(self,hour_idx) :
	
		LC_dec_idx = self.pixel // (2*self.ndec) 
		LC_RA_idx =  self.pixel % (2*self.ndec) 
		EQ_RA_idx = (LC_RA_idx + hour_idx) % (2*self.ndec) 
		
		EQpixel = EQ_RA_idx + (2*self.ndec)*LC_dec_idx
		
		return EQpixel
		
	def EQ2LC(self,hour_idx) :
	
		EQ_dec_idx =  self.pixel // (2*self.ndec) 
		EQ_RA_idx =  self.pixel % (2*self.ndec) 
		LC_RA_idx = (EQ_RA_idx - hour_idx) % (2*self.ndec) 
		
		LCpixel = LC_RA_idx + (2*self.ndec)*EQ_dec_idx
		
		return LCpixel
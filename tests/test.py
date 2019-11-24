from nbodykit.lab import *
from nbodykit import setup_logging
from matplotlib import pyplot
from pylognormal import LogNormalCatalog

setup_logging()

def load_reference_params():
	
	from run_module import read_params
	params = read_params('example.ini')
	
	cosmo_kwargs = dict(Omega0_cdm=params['oc0h2']/params['h0']**2,Omega0_b=params['ob0h2']/params['h0']**2,h=params['h0'],n_s=params['ns'],alpha_s=params['run'],N_ur=2.0328,N_ncdm=1,m_ncdm=params['mnu'])
	cosmo_kwargs['ln10^{10}A_s'] = params['lnAs']
	if params['mnu'] == 0.:
		cosmo_kwargs['N_ur'] = 3.046
		cosmo_kwargs['N_ncdm'] = 0
		cosmo_kwargs['m_ncdm'] = []
	
	input_kwargs = dict(redshift=params['z'],bias=params['bias'],transfer='EisensteinHu')
	
	BoxSize = numpy.array([params['L{}'.format(c)] for c in ['x','y','z']],dtype='float')
	Nmesh = params['Pnmax']
	#Nmesh = numpy.array([256,256,256])
	
	lognormal_kwargs = dict(redshift=params['z'],nbar=params['Ngalaxies']/BoxSize.prod(),BoxSize=BoxSize,Nmesh=Nmesh,seed=params['seed'])
	#lognormal_kwargs['seed'] = 320
	
	power_kwargs = dict(los=numpy.array([params['los{}'.format(c)] for c in ['x','y','z']],dtype='int'),poles=[0,2,4],dk=params['kbin'])
	
	return cosmo_kwargs,input_kwargs,lognormal_kwargs,power_kwargs

def load_reference_gaussian_spectrum():
	
	dtype = [('k','f8'),('power','f8')]
	Pg_delta = numpy.loadtxt('inputs/example_pkG.dat',dtype=dtype)
	Pg_disp = numpy.loadtxt('inputs/example_mpkG.dat',dtype=dtype)
	
	return Pg_delta,Pg_disp

def load_reference_power_spectrum():

	dtype = [('k','f8')] + [('power_{:d}'.format(ell),'f8') for ell in [0,2,4]] + [('modes','i8')]
	poles = numpy.loadtxt('pk/example_pk_rlz0.dat',dtype=dtype)
	
	return poles

def load_reference_catalogue():

	import dask
	cosmo_kwargs,input_kwargs,lognormal_kwargs,power_kwargs = load_reference_params()

	coords = ['x','y','z']
	vels = ['vx','vy','vz']
	cat = CSVCatalog('example_lognormal_rlz0.cat',names=coords+vels,attrs=lognormal_kwargs)
	cat['Position'] = dask.array.stack([cat[c] for c in coords]).T
	cat['Velocity'] = dask.array.stack([cat[c] for c in vels]).T
	
	cosmo = cosmology.Cosmology(**cosmo_kwargs)
	z = lognormal_kwargs['redshift']
	aH = 100 * cosmo.efunc(z) / (1+z)
	cat['VelocityOffset'] = cat['Velocity']/aH
	
	return cat

def calc_reference_power_spectrum():
	
	cosmo_kwargs,input_kwargs,lognormal_kwargs,power_kwargs = load_reference_params()
	cat = load_reference_catalogue()

	los = power_kwargs['los']
	cat['RSDPosition'] = cat['Position'] + cat['VelocityOffset'] * los
	cat['RSDPosition'] %= lognormal_kwargs['BoxSize']
	rsd_mesh = cat.to_mesh(compensated=True, window='cic', position='RSDPosition')

	# compute the 2D power
	r = FFTPower(rsd_mesh,mode='2d',**power_kwargs)
	r.poles['power_0'] -= r.attrs['shotnoise']
	
	return r.poles

def kaiser_pkmu(Plin, mu, f, b1):
	'''
	Returns the Kaiser linear P(k,mu) in redshift space

	.. math::

		P(k,mu) = (1 + f/b_1 mu^2)^2 b_1^2 P_\mathrm{lin}(k)
	'''
	beta = f / b1
	return (1 + beta*mu**2)**2 * b1**2 * Plin

def kaiser_pkl(Plin, ell, f, b1):
	if ell == 0: return (b1**2+2./3.*b1*f+1./5.*f**2)*Plin
	if ell == 2: return (4./3.*b1*f+4./7.*f**2)*Plin
	if ell == 4: return 8./35.*f**2*Plin

def test_lognormal():
	
	cosmo_kwargs,input_kwargs,lognormal_kwargs,power_kwargs = load_reference_params()

	cosmo = cosmology.Cosmology(**cosmo_kwargs)
	f = cosmo.scale_independent_growth_rate(input_kwargs['redshift'])
	b1 = input_kwargs.pop('bias')
	Plin = cosmology.LinearPower(cosmo,**input_kwargs)
	Pdelta = lambda k: b1**2*Plin(k)
	'''
	Pg_delta,Pg_disp = load_reference_gaussian_spectrum()
	Pdelta = lambda k: numpy.interp(k,Pg_delta['k'],Pg_delta['power'])
	Pdisp = lambda k: numpy.interp(k,Pg_disp['k'],Pg_disp['power'])
	cat = LogNormalCatalog(Pdelta=Pdelta,Pdisp=Pdisp,cosmo=cosmo,**lognormal_kwargs)
	'''
	cat = LogNormalCatalog(Pdelta=Pdelta,Pdisp=Plin,cosmo=cosmo,**lognormal_kwargs)
	'''
	if cat.comm.rank == 0:
		print numpy.mean(cat['Velocity'][:,0]).compute(), numpy.mean(cat['Velocity'][:,1]).compute(), numpy.mean(cat['Velocity'][:,2]).compute()
		print numpy.var(cat['Velocity'][:,0]).compute(), numpy.var(cat['Velocity'][:,1]).compute(), numpy.var(cat['Velocity'][:,2]).compute()
		print numpy.min(cat['Velocity'][:,0]).compute(), numpy.min(cat['Velocity'][:,1]).compute(), numpy.min(cat['Velocity'][:,2]).compute()
		print numpy.max(cat['Velocity'][:,0]).compute(), numpy.max(cat['Velocity'][:,1]).compute(), numpy.max(cat['Velocity'][:,2]).compute()
	'''
	los = power_kwargs['los']
	cat['RSDPosition'] = cat['Position'] + cat['VelocityOffset'] * los
	cat['RSDPosition'] %= lognormal_kwargs['BoxSize']
	rsd_mesh = cat.to_mesh(compensated=True,window='cic',position='RSDPosition')

	# compute the 2D power
	r = FFTPower(rsd_mesh,mode='2d',**power_kwargs)

	if cat.comm.rank == 0:
	
		xlim = [0.2,1.]
		
		#ref = load_reference_power_spectrum()
		ref = calc_reference_power_spectrum()
		colors = ['b','r','g']
		
		for ell,color in zip(r.attrs['poles'],colors):
			
			power = r.poles['power_{:d}'.format(ell)].real
			if ell==0: power -= r.poles.attrs['shotnoise']
			mask = (r.poles['k'] >= xlim[0]) & (r.poles['k'] <= xlim[-1])
	 		pyplot.plot(r.poles['k'][mask],r.poles['k'][mask]*power[mask],color=color,linestyle='-',label='$\\ell = {:d}$'.format(ell))
	 		
	 		power = ref['power_{:d}'.format(ell)].real
	 		mask = (ref['k'] >= xlim[0]) & (ref['k'] <= xlim[-1])
			pyplot.plot(ref['k'][mask],ref['k'][mask]*power[mask],color=color,linestyle='--')
			
			kaiser = kaiser_pkl(Plin(ref['k'][mask]),ell,f,b1)
			pyplot.plot(ref['k'][mask],ref['k'][mask]*kaiser,color=color,linestyle=':')
			
		# add a legend and axis labels
		pyplot.xscale('linear')
		pyplot.yscale('linear')
		pyplot.legend(loc=0, ncol=1)
		pyplot.xlabel('$k$ [$h \ \\mathrm{Mpc}^{-1}$]')
		pyplot.ylabel('$kP^{(\\ell)}(k)$ [$(\\mathrm{{Mpc}} \ h^{{-1}})^{{2}}$]')
		pyplot.xlim(xlim)

		pyplot.show()
	
	if cat.comm.rank == 0:
	
		xlim = [0.2,1.]
	
		#ref = load_reference_power_spectrum()
		#ref = calc_reference_power_spectrum()
		colors = ['b','r','g']
		
		for ell,color in zip(r.attrs['poles'],colors):
			mask = (r.poles['k'] >= xlim[0]) & (r.poles['k'] <= xlim[-1])
			diff = r.poles['power_{:d}'.format(ell)].real[mask]-numpy.interp(r.poles['k'][mask],ref['k'],ref['power_{:d}'.format(ell)].real)
	 		pyplot.plot(r.poles['k'][mask],diff,color=color,linestyle='-',label='$\\ell = {:d}$'.format(ell))
			
		# add a legend and axis labels
		pyplot.grid(True)
		pyplot.xscale('linear')
		pyplot.yscale('linear')
		pyplot.legend(loc=0, ncol=1)
		pyplot.xlabel('$k$ [$h \ \\mathrm{Mpc}^{-1}$]')
		pyplot.ylabel('$\Delta P^{(\\ell)}(k)$ [$(\\mathrm{{Mpc}} \ h^{{-1}})^{{3}}$]')
		pyplot.xlim(xlim)

		pyplot.show()
	
#calc_reference_power_spectrum()
test_lognormal()

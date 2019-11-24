from nbodykit.base.catalog import CatalogSource, column
from nbodykit import cosmology
from nbodykit.utils import attrs_to_dict
from nbodykit import CurrentMPIComm
import mockmaker

import logging
import numpy

class LogNormalCatalog(CatalogSource):
	"""
	A CatalogSource containing biased particles that have
	been Poisson-sampled from a log-normal density field.

	Parameters
	----------
	Pdelta : callable
		callable specifying the galaxy power spectrum at the desired redshift.
	nbar : float
		the number density of the particles in the box, assumed constant across
		the box; this is used when Poisson sampling the density field
	BoxSize : float, 3-vector of floats
		the size of the box to generate the grid on
	Nmesh : int
		the mesh size to use when generating the density and displacement
		fields, which are Poisson-sampled to particles
	Pdisp : callable, optional
		callable specifying the velocity divergence power spectrum at the desired redshift
	k : array, optional
		if Pdelta (and Pdisp) are arrays, the input k
	seed : int, optional
		the global random seed; if set to ``None``, the seed will be set
		randomly
	cosmo : :class:`nbodykit.cosmology.core.Cosmology`, optional
		this must be supplied if ``Pdelta`` does not carry ``cosmo`` attribute
	redshift : float, optional
		this must be supplied if ``Pdelta`` does not carry a ``redshift`` attribute
	comm : MPI Communicator, optional
		the MPI communicator instance; default (``None``) sets to the
		current communicator

	References
	----------
	`Cole and Jones, 1991 <http://adsabs.harvard.edu/abs/1991MNRAS.248....1C>`_
	`Agrawal et al. 2017 <https://arxiv.org/abs/1706.09195>`_
	"""
	def __repr__(self):
		return "LogNormalCatalog(seed=%(seed)d, redshift=%(redshift)g)" %self.attrs

	logger = logging.getLogger("LogNormalCatalog")

	@CurrentMPIComm.enable
	def __init__(self, Pdelta, nbar=1e-4, BoxSize=1000., Nmesh=256, k=None, Pdisp=None, seed=None,
					cosmo=None, redshift=None,
					unitary_amplitude=False, inverted_phase=False, comm=None):

		self.comm = comm
		
		self.Pg_delta = mockmaker.target_to_gaussian_power_spectrum(Pdelta,k=k,s=None,extrap=False)
		if Pdisp is not None:
			self.Pg_disp = mockmaker.target_to_gaussian_power_spectrum(Pdisp,k=k,s=None,extrap=False)
		else:
			self.Pg_disp = None
		"""
		self.Pg_delta = Pdelta
		self.Pg_disp = Pdisp
		"""
		# try to infer cosmo or redshift from Pdelta
		if self.Pg_disp is not None:
			if cosmo is None:
				cosmo = getattr(Pdelta, 'cosmo', None)
			if redshift is None:
				redshift = getattr(Pdelta, 'redshift', None)
			if cosmo is None:
				raise ValueError("'cosmo' must be passed if 'Pdelta' does not have 'cosmo' attribute")
			if redshift is None:
				raise ValueError("'redshift' must be passed if 'Pdelta' does not have 'redshift' attribute")
			self.cosmo = cosmo

		# try to add attrs from the Pdelta
		if hasattr(Pdelta, 'attrs'):
			self.attrs.update(Pdelta.attrs)
		else:
			self.attrs['cosmo'] = dict(cosmo) if cosmo is not None else None

		# save the meta-data
		self.attrs['nbar'] = nbar
		self.attrs['redshift'] = redshift
		self.attrs['unitary_amplitude'] = unitary_amplitude
		self.attrs['inverted_phase'] = inverted_phase

		# set the seed randomly if it is None
		if seed is None:
			if self.comm.rank == 0:
				seed = numpy.random.randint(0, 4294967295)
			seed = self.comm.bcast(seed)
		self.attrs['seed'] = seed

		# make the actual source
		self._source, pm = self._makesource(BoxSize=BoxSize, Nmesh=Nmesh)
		self.pm = pm

		self.attrs['Nmesh'] = pm.Nmesh.copy()
		self.attrs['BoxSize'] = pm.BoxSize.copy()

		# set the size
		self._size = len(self._source)

		# init the base class
		CatalogSource.__init__(self, comm=comm)

		# crash with no particles!
		if self.csize == 0:
			raise ValueError("no particles in LogNormal source; try increasing ``nbar`` parameter")

	@column
	def Position(self):
		"""
		Position assumed to be in Mpc/h
		"""
		return self.make_column(self._source['Position'])

	@column
	def Velocity(self):
		"""
		Velocity in km/s
		"""
		return self.make_column(self._source['Velocity'])

	@column
	def VelocityOffset(self):
		"""
		The corresponding RSD offset, in Mpc/h
		"""
		return self.make_column(self._source['VelocityOffset'])

	def _makesource(self, BoxSize, Nmesh):

		from pmesh.pm import ParticleMesh

		# the particle mesh for gridding purposes
		_Nmesh = numpy.empty(3, dtype='i8')
		_Nmesh[:] = Nmesh
		pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh, dtype='f4', comm=self.comm)

		# compute the linear overdensity and displacement fields
		delta = mockmaker.gaussian_complex_field(pm, self.attrs['seed'],
					unitary_amplitude=self.attrs['unitary_amplitude'],
					inverted_phase=self.attrs['inverted_phase'],
					logger=self.logger)
		
		disp = delta.copy()
	
		if self.comm.rank == 0:
			self.logger.info("Gaussian field computed in Fourier space")

		mockmaker.apply_power_spectrum(delta,pm.BoxSize,self.Pg_delta)
		delta = delta.c2r()
		mockmaker.apply_lognormal_transform(delta,bias=1.)
		
		std = (delta ** 2).cmean() ** 0.5
		if self.comm.rank == 0:
			self.logger.info("Overdensity computed in configuration space: std = {:.4g}".format(std))
		
		if self.Pg_disp is not None:
			mockmaker.apply_power_spectrum(disp,pm.BoxSize,self.Pg_disp)
			disp = disp.c2r()
			mockmaker.apply_lognormal_transform(disp,bias=1.)
			disp = disp.r2c()
			disp = mockmaker.divergence_to_displacement(disp)
			disp = [d.c2r() for d in disp]
			std = [(d ** 2).cmean() ** 0.5 for d in disp]
			if self.comm.rank == 0:
				self.logger.info("Displacement computed in configuration space: std = [{:.4g} {:.4g} {:.4g}]".format(*std))
		else:
			disp = None

		# poisson sample to points
		# this returns position and velocity offsets
		kws = {'seed':self.attrs['seed'], 'logger' : self.logger}
		pos, disp = mockmaker.poisson_sample_to_points(delta, pm, self.attrs['nbar'], displacement=disp, **kws)

		if self.comm.rank == 0:
			self.logger.info("Poisson sampling is generated")

		if disp is not None:
		
			# growth rate to do RSD in the Zel'dovich approx
			f = self.cosmo.scale_independent_growth_rate(self.attrs['redshift'])

			if self.comm.rank == 0:
				self.logger.info("Growth Rate is %g" % f)

			# velocity from displacement (assuming Mpc/h)
			# this is f * H(z) * a / h = f 100 E(z) a --> converts from Mpc/h to km/s
			z = self.attrs['redshift']
			velocity_norm = f * 100 * self.cosmo.efunc(z) / (1+z)
			vel = velocity_norm * disp

			# return data
			dtype = numpy.dtype([
					('Position', ('f4', 3)),
					('Velocity', ('f4', 3)),
					('VelocityOffset', ('f4', 3))
			])
			source = numpy.empty(len(pos), dtype)
			source['Position'][:] = pos[:] # in Mpc/h
			source['Velocity'][:] = vel[:] # in km/s
			source['VelocityOffset'][:] = f*disp[:] # in Mpc/h
		
		else:
			# return data
			dtype = numpy.dtype([
					('Position', ('f4', 3)),
			])
			source = numpy.empty(len(pos), dtype)
			source['Position'][:] = pos[:] # in Mpc/h

		return source, pm


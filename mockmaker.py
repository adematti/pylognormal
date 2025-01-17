import numpy
import numbers
from mpi4py import MPI

from pmesh.pm import RealField, ComplexField
from nbodykit.meshtools import SlabIterator
from nbodykit.utils import GatherArray, ScatterArray
from nbodykit.mpirng import MPIRandomState
import mpsort

def gaussian_complex_field(pm, seed,
			unitary_amplitude=False, inverted_phase=False, logger=None):
	r"""
	Make a Gaussian realization of a overdensity field, :math:`\delta(x)`.

	The complex field has unity variance. This
	is equivalent to generating real-space normal variates
	with mean and unity variance, calling r2c() and dividing by :math:`N^3`
	since the variance of the complex FFT (with no additional normalization)
	is :math:`N^3 \times \sigma^2_\mathrm{real}`.

	Parameters
	----------
	pm : pmesh.pm.ParticleMesh
		the mesh object
	seed : int
		the random seed used to generate the random field
	unitary_amplitude : bool, optional
		if ``True``, the seed gaussian has unitary_amplitude.
	inverted_phase: bool, optional
		if ``True``, the phase of the seed gaussian is inverted

	Returns
	-------
	delta_k : ComplexField
		the real-space Gaussian overdensity field
	"""
	if not isinstance(seed, numbers.Integral):
		raise ValueError("the seed used to generate the linear field must be an integer")

	if logger and pm.comm.rank == 0:
		logger.info("Generating whitenoise")

	# use pmesh to generate random complex white noise field (done in parallel)
	# variance of complex field is unity
	# multiply by P(k)**0.5 to get desired variance
	delta_k = pm.generate_whitenoise(seed, type='untransposedcomplex', unitary=unitary_amplitude)

	if logger and pm.comm.rank == 0:
		logger.info("Whitenoise is generated")

	if inverted_phase: delta_k[...] *= -1

	return delta_k


def apply_power_spectrum(delta_k,BoxSize,gaussian_power):

	r"""
	Scale the Fourier field by :math:`(P(k) / V)^{1/2}`

	The power spectrum is defined as V * variance.
	So a normalization factor of 1 / V shows up such that the power spectrum is P(k).
	"""

	# volume factor needed for normalization
	norm = 1.0 / BoxSize.prod()

	# iterate in slabs over fields
	slabs = [delta_k.slabs.x, delta_k.slabs]

	# loop over the mesh, slab by slab
	for islabs in zip(*slabs):
		kslab, delta_slab = islabs[:2] # the k arrays and delta slab

		# the square of the norm of k on the mesh
		k2 = sum(kk**2 for kk in kslab)
		zero_idx = k2 == 0.

		k2[zero_idx] = 1. # avoid dividing by zero

		# multiply complex field by sqrt of power
		delta_slab[...].flat *= (gaussian_power((k2**0.5).flatten())*norm)**0.5

		# set k == 0 to zero (zero config-space mean)
		delta_slab[zero_idx] = 0.
	
def divergence_to_displacement(div_k):

	disp_k = [div_k.copy() for i in range(div_k.ndim)]
	for i in range(div_k.ndim): disp_k[i][:] = 1j
	
	# iterate in slabs over fields
	slabs = [div_k.slabs.x,div_k.slabs] + [d.slabs for d in disp_k]

	# loop over the mesh, slab by slab
	for islabs in zip(*slabs):

		kslab, div_slab = islabs[:2]
		# the square of the norm of k on the mesh
		k2 = sum(kk**2 for kk in kslab)
		zero_idx = k2 == 0.

		k2[zero_idx] = 1. # avoid dividing by zero

		# ignore division where k==0 and set to 0
		with numpy.errstate(invalid='ignore', divide='ignore'):
			for i in range(div_k.ndim):
				disp_slab = islabs[2+i]
				disp_slab[...] *= kslab[i] / k2 * div_slab[...]
				disp_slab[zero_idx] = 0. # no bulk displacement
	
	return disp_k
	

def apply_lognormal_transform(density, bias=1.):
	r"""
	Apply a (biased) lognormal transformation of the density
	field by computing:

	.. math::

		F(\delta) = \frac{1}{N} e^{b*\delta}

	where :math:`\delta` is the initial overdensity field and the
	normalization :math:`N` is chosen such that
	:math:`\langle F(\delta) \rangle = 1`
	The transformation is done in place.

	Parameters
	----------
	density : array_like
		the input density field to apply the transformation to
	bias : float, optional
		optionally apply a linear bias to the density field;
		default is unbiased (1.0)
	"""
	density[:] = numpy.exp(bias * density.value)
	density[:] /= density.cmean(dtype='f8')
	density[:] -= 1.


def poisson_sample_to_points(delta, pm, nbar, displacement=None, bias=1., seed=None, logger=None):
	"""
	Poisson sample the linear delta and displacement fields to points.

	The steps in this function:

	#.  Apply a biased, lognormal transformation to the input ``delta`` field
	#.  Poisson sample the overdensity field to discrete points
	#.  Disribute the positions of particles uniformly within the mesh cells,
		and assign the displacement field at each cell to the particles

	Parameters
	----------
	delta : RealField
		the linear overdensity field to sample
	displacement : list of RealField (3,)
		the linear displacement fields which is used to move the particles
	nbar : float
		the desired number density of the output catalog of objects
	bias : float, optional
		apply a linear bias to the overdensity field (default is 1.)
	seed : int, optional
		the random seed used to Poisson sample the field to points

	Returns
	-------
	pos : array_like, (N, 3)
		the Cartesian positions of each of the generated particles
	displ : array_like, (N, 3)
		the displacement field sampled for each of the generated particles in the
		same units as the ``pos`` array
	"""
	comm = delta.pm.comm

	# seed1 used for poisson sampling
	# seed2 used for uniform shift within a cell.
	seed1, seed2 = numpy.random.RandomState(seed).randint(0, 0xfffffff, size=2)

	if logger and pm.comm.rank == 0:
		logger.info("Lognormal transformation done")

	# mean number of objects per cell
	H = delta.BoxSize / delta.Nmesh
	overallmean = H.prod() * nbar

	# number of objects in each cell (per rank, as a RealField)
	cellmean = (delta+1.) * overallmean

	# create a random state with the input seed
	rng = MPIRandomState(seed=seed1, comm=comm, size=delta.size)

	# generate poissons. Note that we use ravel/unravel to
	# maintain MPI invariane.
	Nravel = rng.poisson(lam=cellmean.ravel())
	N = delta.pm.create(type='real')
	N.unravel(Nravel)

	Ntot = N.csum()
	if logger and pm.comm.rank == 0:
		logger.info("Poisson sampling done, total number of objects is %d" % Ntot)

	pos_mesh = delta.pm.generate_uniform_particle_grid(shift=0.0)

	# no need to do decompose because pos_mesh is strictly within the
	# local volume of the RealField.
	N_per_cell = N.readout(pos_mesh, resampler='nnb')

	# fight round off errors, if any
	N_per_cell = numpy.int64(N_per_cell + 0.5)

	if displacement is not None:
		disp_mesh = numpy.empty_like(pos_mesh)
		for i in range(N.ndim):
			disp_mesh[:, i] = displacement[i].readout(pos_mesh, resampler='nnb')
		disp = disp_mesh.repeat(N_per_cell, axis=0)
		del disp_mesh

	pos = pos_mesh.repeat(N_per_cell, axis=0)
	del pos_mesh

	if logger and pm.comm.rank == 0:
		logger.info("catalog produced. Assigning in cell shift.")

	# generate linear ordering of the positions.
	# this should have been a method in pmesh, e.g. argument
	# to genereate_uniform_particle_grid(return_id=True);

	# FIXME: after pmesh update, remove this
	orderby = numpy.int64(pos[:, 0] / H[0] + 0.5)
	for i in range(1, delta.ndim):
		orderby[...] *= delta.Nmesh[i]
		orderby[...] += numpy.int64(pos[:, i] / H[i] + 0.5)

	# sort by ID to maintain MPI invariance.
	pos = mpsort.sort(pos, orderby=orderby, comm=comm)
	if displacement is not None: disp = mpsort.sort(disp, orderby=orderby, comm=comm)
	else: disp = None

	if logger and pm.comm.rank == 0:
		logger.info("sorting done")

	rng_shift = MPIRandomState(seed=seed2, comm=comm, size=len(pos))
	in_cell_shift = rng_shift.uniform(0, H[i], itemshape=(delta.ndim,))

	pos[...] += in_cell_shift
	pos[...] %= delta.BoxSize

	if logger and pm.comm.rank == 0:
		logger.info("catalog shifted.")

	return pos, disp


def target_to_gaussian_power_spectrum(Ptg,k=None,s=None,extrap=False):
	
	from nbodykit import cosmology

	if k is None:
		k = numpy.logspace(-5,5,10000,base=10)
		Ptg = Ptg(k)
	xitg = cosmology.correlation.pk_to_xi(k,Ptg,extrap=extrap)
	if s is None: s = 1./k[::-1]
	xitg = xitg(s)
	xig = numpy.log(1+xitg)
	
	Pg = cosmology.correlation.xi_to_pk(s,xig,extrap=extrap)
	
	def Pg_safe(k):
		toret = Pg(k)
		toret[k==0.] = 0.
		return toret

	return Pg_safe

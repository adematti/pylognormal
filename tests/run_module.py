import os
import sys
import re
from string import *
import numpy as np

# read parameters from .ini file
def read_params(ini_fname):
	ini_file = open(ini_fname,'r')
	ofile_prefix = strip(re.split('\.',ini_fname)[0])
	params = {'ofile_prefix':ofile_prefix,\
			  'inp_pk_fname':'', 'xi_fname':'',\
			  'pkg_fname':'','mpkg_fname':'','cpkg_fname':'',\
			  'f_fname':'',\
			  'z':0.0,'mnu':0.06,'oc0h2':0.144,\
			  'ob0h2':0.025,'ns':0.96,\
			  'lnAs':3.04,'h0':0.678,'w':-1.0,'run':0.0,\
			  'bias':1.0,'bias_mpkG':1.0,'bias_cpkG':1.0,\
			  'Nrealization':1,\
			  'Ngalaxies':10000,\
			  'Lx':500.,'Ly':500.,'Lz':500.,\
			  'rmax':10000.,'seed':1,\
			  'Pnmax':1024,'losx':0.,'losy':0.,'losz':0.,\
			  'kbin':0.01,'kmax':0.,'lmax':4,\
			  'gen_inputs':False,'run_lognormal':False,'calc_pk':False,'calc_cpk':False,\
			  'use_cpkG':0,\
                          'output_matter':1,'output_gal':1,\
			  'calc_mode_pk':0,\
			  'out_dir':'\./data',\
			  'halofname_prefix':'',\
			  'imul_fname':'',\
			  'num_para':1}
 
	ini_lines = ini_file.readlines()
	for line in ini_lines:
		if not (line.startswith("#") or len(line.strip()) == 0): # skip comments
			pname = strip(re.split(r'=|#', line)[0]) 			
			value = strip(re.split(r'=|#', line)[1]) 
			for key in params.keys():
				if key == pname:
					if type(params[key]) == bool:
						if value in ['True','T']:
							params[key] = True
						elif value in ['False','F']:
							params[key] = False
						else:
							print 'logical parameters should be True, T, False or F!'
							quit() 
					else:
						params[key] = type(params[key])(value)

	# if params['inp_pk_fname'] is blanck,  use Eisenstein & Hu for input pk
	if params['inp_pk_fname'] == '':
		params['inp_pk_fname'] = params['out_dir']+'/inputs/'+params['ofile_prefix']+'_pk.txt'
	if params['xi_fname'] == '':
		params['xi_fname'] = params['out_dir']+'/inputs/'+params['ofile_prefix']+'_Rh_xi.txt'
	if params['pkg_fname'] == '':
		params['pkg_fname'] = params['out_dir']+'/inputs/'+params['ofile_prefix']+'_pkG.dat'
	if params['mpkg_fname'] == '':
		params['mpkg_fname'] = params['out_dir']+'/inputs/'+params['ofile_prefix']+'_mpkG.dat'
	if params['cpkg_fname'] == '':
		if params['use_cpkG'] == 0:
			params['cpkg_fname'] = params['mpkg_fname'] # dummy
		else:
			params['cpkg_fname'] = params['out_dir']+'/inputs/'+params['ofile_prefix']+'_cpkG.dat'
	if params['f_fname'] == '':
		params['f_fname'] = params['out_dir']+'/inputs/'+params['ofile_prefix']+'_fnu.txt'

	params['om0h2'] = params['oc0h2']+params['ob0h2']+params['mnu']/93.1
	params['om0'] = params['om0h2']/params['h0']**2
	params['ob0'] = params['ob0h2']/params['h0']**2
	params['ode0'] = 1.0-params['om0']
	params['As'] = np.exp(params['lnAs'])*1e-10
	params['aH'] = 100.*pow(params['om0']*pow(1.+params['z'],3)+params['ode0'],0.5)/(1.+params['z'])	

	return params

def check_dir(params):
	try:
	        os.mkdir(params['out_dir'])
	except:
	        print 'Directory '+params['out_dir']+' exist!'
	try:
	        os.mkdir(params['out_dir']+'/inputs')
	except:
	        print 'Directory '+params['out_dir']+'/inputs'+' exist!'
	try:
	        os.mkdir(params['out_dir']+'/lognormal')
	except:
	        print 'Directory '+params['out_dir']+'/lognormal'+' exist!'
	try:
	        os.mkdir(params['out_dir']+'/pk')
	except:
	        print 'Directory '+params['out_dir']+'/pk'+' exist!'
	try:
	        os.mkdir(params['out_dir']+'/coupling')
	except:
	        print 'Directory '+params['out_dir']+'/coupling'+' exist!'


class executable:
	''' class for execute commands'''
	def __init__(self,name):
		self.name = name

	def run(self,exename,args,params):
		args = ' '.join(map(str, [params[key] for key in args]))
		cmd = 'time ./'+exename+' '+args
		print cmd
		os.system(cmd)


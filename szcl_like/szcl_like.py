import pyccl as ccl
from .theory import HaloProfileArnaud, SZTracer
import numpy as np
from scipy.interpolate import interp1d


class SZClLike(object):
    def __init__(self, config):
        self.config = config
        self.nl_per_decade = 5
        self.mdef = ccl.halos.MassDef(500, 'critical')
        self.prof = HaloProfileArnaud(0.2)
        self._read_data()
        self.cosmo = ccl.Cosmology(Omega_c=0.25,
                                   Omega_b=0.05,
                                   h=0.67,
                                   n_s=0.96,
                                   sigma8=0.8)
        self._get_sz_model()
        self.ks = np.geomspace(1E-4, 100, 256)
        self.lks = np.log(self.ks)
        self.a_s = np.linspace(0.1, 1, 10)
        self.add_2h = False

    def _read_data(self):
        import sacc
        # Read data vector and covariance
        s = sacc.Sacc.load_fits(self.config['cl_file'])
        if self.config['map_name'] not in list(s.tracers.keys()):
            raise KeyError("Map not found")

        inds = s.indices('cl_00',
                         (self.config['map_name'],
                          self.config['map_name']),
                         ell__gt=self.config['l_min'],
                         ell__lt=self.config['l_max'])
        s.keep_indices(inds)
        ls, cl, win = s.get_ell_cl('cl_00',
                                   self.config['map_name'],
                                   self.config['map_name'],
                                   return_windows=True)
        self.leff = ls
        self.data = cl
        self.cov = s.covariance.covmat
        self.invcov = np.linalg.inv(self.cov)

        # Read bandpower window functions
        self.ls_all = win.values
        self.l_ls_all = np.log(self.ls_all)
        self.windows = win.weight.T

        # Read beam and resample
        t = s.get_tracer(self.config['map_name'])
        beam_f = interp1d(t.ell, t.beam_ell,
                          bounds_error=False,
                          fill_value=0)
        self.beam2 = beam_f(self.ls_all)**2

        # Compute ell nodes
        l10_lmax = np.log10(self.ls_all[-1])
        n_sample = int(l10_lmax * self.nl_per_decade) + 1
        self.ls_sample = np.unique(np.logspace(0,
                                               l10_lmax,
                                               n_sample).astype(int)).astype(float)
        self.l_ls_sample = np.log(self.ls_sample)
        
    def _get_sz_model(self):
        self.hmf = ccl.halos.MassFuncTinker08(self.cosmo,
                                              mass_def=self.mdef)
        self.hmb = ccl.halos.HaloBiasTinker10(self.cosmo,
                                              mass_def=self.mdef,
                                              mass_def_strict=False)
        self.hmc = ccl.halos.HMCalculator(self.cosmo,
                                          self.hmf,
                                          self.hmb,
                                          self.mdef)
        self.szk = SZTracer(self.cosmo)

    def _check_cosmo_changed(self, **pars):
        if ((pars['Omega_c']!=self.cosmo['Omega_c']) or
            (pars['Omega_b']!=self.cosmo['Omega_b']) or
            (pars['h']!=self.cosmo['h']) or
            (pars['n_s']!=self.cosmo['n_s']) or
            (pars['sigma8']!=self.cosmo['sigma8'])):
            self.cosmo = ccl.Cosmology(Omega_c=pars['Omega_c'],
                                       Omega_b=pars['Omega_b'],
                                       h=pars['h'],
                                       n_s=pars['n_s'],
                                       sigma8=pars['sigma8'])
            self._get_sz_model()

    def _get_theory(self, **pars):
        self._check_cosmo_changed(**pars)
        self.prof._update_bhydro(pars['b_hydro'])
        pk2d = ccl.halos.halomod_Pk2D(self.cosmo,
                                      self.hmc,
                                      self.prof,
                                      lk_arr=self.lks,
                                      a_arr=self.a_s,
                                      get_2h=self.add_2h)
        cls = ccl.angular_cl(self.cosmo, self.szk, self.szk,
                             self.ls_sample,
                             p_of_k_a=pk2d)
        clf = interp1d(self.l_ls_sample, np.log(cls),
                       bounds_error=False, fill_value=-200)
        cls = np.exp(clf(self.l_ls_all)) * self.beam2
        cls = np.dot(self.windows, cls)
        return cls
        
    def logp(self, **pars):
        t = self._get_theory(**pars)
        r = t - self.data
        chi2 = np.dot(r, self.invcov.dot(r))
        return -0.5 * chi2

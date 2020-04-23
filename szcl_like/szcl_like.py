import pyccl as ccl
from .theory import HaloProfileArnaud, SZTracer
import numpy as np
from scipy.interpolate import interp1d
from cobaya.likelihood import Likelihood


class SZModel:
    pass


class SZClLike(Likelihood):
    cl_file: str = "data/cl_yy.fits"
    map_name: str = "SO_y"
    l_min: int = 100
    l_max: int = 3000

    params = {'b_hydro': 0.2}

    def initialize(self):
        self.nl_per_decade = 5
        self.mdef = ccl.halos.MassDef(500, 'critical')
        self.prof = HaloProfileArnaud(0.2)
        self._read_data()
        self.ks = np.geomspace(1E-4, 100, 256)
        self.lks = np.log(self.ks)
        self.a_s = np.linspace(0.1, 1, 10)
        self.add_2h = False

    def get_requirements(self):
        return {'CCL': {'sz_model': self._get_sz_model}}

    def _read_data(self):
        import sacc
        # Read data vector and covariance
        s = sacc.Sacc.load_fits(self.cl_file)
        if self.map_name not in list(s.tracers.keys()):
            raise KeyError("Map not found")

        inds = s.indices('cl_00',
                         (self.map_name,
                          self.map_name),
                         ell__gt=self.l_min,
                         ell__lt=self.l_max)
        s.keep_indices(inds)
        ls, cl, win = s.get_ell_cl('cl_00',
                                   self.map_name,
                                   self.map_name,
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
        t = s.get_tracer(self.map_name)
        beam_f = interp1d(t.ell, t.beam_ell,
                          bounds_error=False,
                          fill_value=0)
        self.beam2 = beam_f(self.ls_all) ** 2

        # Compute ell nodes
        l10_lmax = np.log10(self.ls_all[-1])
        n_sample = int(l10_lmax * self.nl_per_decade) + 1
        self.ls_sample = np.unique(np.logspace(0,
                                               l10_lmax,
                                               n_sample).astype(int)).astype(float)
        self.l_ls_sample = np.log(self.ls_sample)

    def _get_sz_model(self, cosmo):
        model = SZModel()
        model.hmf = ccl.halos.MassFuncTinker08(cosmo,
                                               mass_def=self.mdef)
        model.hmb = ccl.halos.HaloBiasTinker10(cosmo,
                                               mass_def=self.mdef,
                                               mass_def_strict=False)
        model.hmc = ccl.halos.HMCalculator(cosmo,
                                           model.hmf,
                                           model.hmb,
                                           self.mdef)
        model.szk = SZTracer(cosmo)
        return model

    def _get_theory(self, **pars):
        results = self.provider.get_CCL()
        cosmo = results['cosmo']
        sz_model = results['sz_model']

        self.prof._update_bhydro(pars['b_hydro'])
        pk2d = ccl.halos.halomod_Pk2D(cosmo,
                                      sz_model.hmc,
                                      self.prof,
                                      lk_arr=self.lks,
                                      a_arr=self.a_s,
                                      get_2h=self.add_2h)
        cls = ccl.angular_cl(cosmo, sz_model.szk, sz_model.szk,
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

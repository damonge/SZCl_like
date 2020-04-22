# Simple CCL wrapper with function to return CCL cosmo object
# likelihoods should then calculate what they need themselves from the cosmo object
# Lots of things still cannot be done consistently in CCL

# For Cobaya docs see
# https://cobaya.readthedocs.io/en/devel/theory.html
# https://cobaya.readthedocs.io/en/devel/theories_and_dependencies.html

import numpy as np
from cobaya.theory import Theory


class CCL(Theory):

    def initialize(self):
        self._zmax = 0
        self._kmax = 0
        self._z_sampling = []
        self._var_pairs = set()

    def get_requirements(self):
        # These are currently required to construct a CCL cosmology object.
        # Ultimately CCL should depend only on observable not parameters
        # 'As' could be substituted by sigma8.
        return {'omch2', 'ombh2', 'ns', 'As'}

    def needs(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        # CCL currently has no way to infer the required inputs from the required outputs
        # So a lot of this is fixed

        # fix for now
        self._zmax = max(5, self._zmax)
        # sampling_step = self._zmax * 0.1
        # self._z_sampling = np.arange(0, self._zmax, sampling_step)
        # Fixed at 100 steps to z max for now
        self._z_sampling = np.linspace(0, self._zmax, 100)

        # Dictionary of the things CCL needs from CAMB/CLASS
        needs = {}

        Pk = requirements.get("Pk")
        if Pk:
            # CCL currently only supports ('delta_tot', 'delta_tot'), but call allow
            # general as placeholder
            self._kmax = max(Pk.get('kmax', 10), self._kmax)
            self._var_pairs.update(
                set(tuple(x, y) for x, y in
                    Pk.get('vars_pairs', [('delta_tot', 'delta_tot')])))

            # for the moment always get Pk_grid, when supported should
            needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': (True, False),
                'z': self._z_sampling,
                'k_max': self._kmax
            }

        needs['Hubble'] = {'z': self._z_sampling}
        needs['comoving_radial_distance'] = {'z': self._z_sampling}

        assert len(self._var_pairs) < 2, "CCL doesn't support other Pk yet"

        return needs

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        # calculate the general CCL cosmo object which likelihoods can then use to get
        # what they need (likelihoods should cache results appropriately)
        # get our requirements from self.provider

        distance = self.provider.get_comoving_radial_distance(self._z_sampling)
        hubble_z = self.provider.get_Hubble(self._z_sampling)
        H0 = hubble_z[0]
        E_of_z = hubble_z / H0
        # Array z is sorted in ascending order. CCL requires an ascending scale factor
        # as input
        # Flip the arrays to make them a function of the increasing scale factor.
        # If redshift sampling is changed, check that it is monotonically increasing
        distance = np.flip(distance)
        E_of_z = np.flip(E_of_z)

        # Create a CCL cosmology object
        import pyccl as ccl
        h = H0 / 100.
        Omega_c = self.provider.get_param('omch2') / h ** 2
        Omega_b = self.provider.get_param('ombh2') / h ** 2

        # Currently, CCL requires the (ill-defined) linear "matter" perturbation
        # growth factor and rate. Because it's ill-defined, we can't get it from
        # Boltzmann code in general; ultimately CCL should use more physical
        # inputs for anything of use in general models.
        # For now just compute from CCL itself to keep it happy:
        cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h,
                              n_s=self.provider.get_param('ns'),
                              A_s=self.provider.get_param('As'))
        # Array z is sorted in ascending order. CCL requires an ascending scale
        # factor as input
        a = 1. / (1 + self._z_sampling[::-1])
        growth = ccl.background.growth_factor(cosmo, a)
        fgrowth = ccl.background.growth_rate(cosmo, a)
        # In order to use CCL with input arrays, the cosmology object needs
        # to be reset. This should be improved...
        cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h,
                              n_s=self.provider.get_param('ns'),
                              A_s=self.provider.get_param('As'))
        cosmo._set_background_from_arrays(a_array=a,
                                          chi_array=distance,
                                          hoh0_array=E_of_z,
                                          growth_array=growth,
                                          fgrowth_array=fgrowth)

        if self._kmax:
            for pair in self._var_pairs:
                # Get the matter power spectrum:
                k, z, Pk_lin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)
                k, z, Pk_nonlin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=True)

                # np.flip(arr, axis=0) flips the rows of arr, thus making Pk with z
                # in descending order.
                Pk_lin = np.flip(Pk_lin, axis=0)
                Pk_nonlin = np.flip(Pk_nonlin, axis=0)

                cosmo._set_linear_power_from_arrays(a, k, Pk_lin)
                cosmo._set_nonlin_power_from_arrays(a, k, Pk_nonlin)

        state['cosmology'] = cosmo
        state['cache'] = {}

    def get_CCL_cosmology(self):
        """
        Get the CCL Cosmology object for the current state, and an additional dict for cacheing any directly derived
        results.

        :return: tuple (CCL Cosmology object. cache dict)
        """
        return self._current_state['cosmology'], self._current_state['cache']

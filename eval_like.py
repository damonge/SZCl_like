from szcl_like import SZClLike
from collections import OrderedDict as odict

cosmo_params = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.67,
    "n_s": 0.96,
    "sigma8": 0.8}
nuisance_params = {
    "b_hydro": 0.2}

szcllike_config = {"cl_file": "data/cl_yy.fits",
                   "map_name": "SO_y",
                   "l_min": 100,
                   "l_max": 3000}
szcllike = SZClLike(szcllike_config)

def logp(Omega_c, Omega_b, h, n_s, sigma8, b_hydro):
    p = {'Omega_c': Omega_c,
         'Omega_b': Omega_b,
         'h': h, 'n_s': n_s,
         'sigma8': sigma8,
         'b_hydro': b_hydro}
    return szcllike.logp(**p)

info = {"params": {**cosmo_params, **nuisance_params},
        "likelihood": {'szcl': logp}}

from cobaya.model import get_model
model = get_model(info)
loglikes, derived = model.loglikes({})
print(-2*loglikes, len(szcllike.data))

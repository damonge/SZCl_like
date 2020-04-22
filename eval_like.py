from .szcl_like.szcl_like import SZClLike
from .szcl_like.ccl import CCL
from cobaya.model import get_model

cosmo_params = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.67,
    "n_s": 0.96,
    "sigma8": 0.8}
nuisance_params = {
    "b_hydro": 0.2}

info = {"params": {"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                   "ombh2": cosmo_params['Omega_b'] * cosmo_params['h'] ** 2.,
                   "H0": cosmo_params['h'] * 100,
                   "ns": cosmo_params['n_s'],
                   "As": cosmo_params['A_s'],
                   "tau": 0,
                   **nuisance_params},
        "likelihood": {'szcl': SZClLike},
        "theory": {
            "camb": None,
            "ccl": {"exernal": CCL, "Pk": {"kmax": 10}}
        }}

model = get_model(info)
loglikes, derived = model.loglikes({})
print(-2 * loglikes)

from szcl_like.szcl_like import SZClLike
from szcl_like.ccl import CCL
from cobaya.model import get_model

cosmo_params = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.67,
    "n_s": 0.96
}
nuisance_params = {
    "b_hydro": 0.2}

info = {"params": {"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                   "ombh2": cosmo_params['Omega_b'] * cosmo_params['h'] ** 2.,
                   "H0": cosmo_params['h'] * 100,
                   "ns": cosmo_params['n_s'],
                   "As": 2.2e-9,
                   "tau": 0,
                   **nuisance_params},
        "likelihood": {'szcl': SZClLike},
        "theory": {
            "camb": None,
            "ccl": {"external": CCL}
        }}

model = get_model(info)
loglikes, derived = model.loglikes({})
print(-2 * loglikes)

import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from szcl_like.theory import HaloProfileArnaud, SZTracer
from scipy.interpolate import interp1d
import sacc

# read noise
l_n_t, _, nl_yy_t = np.loadtxt("data/AdvACT_T_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj0_mask_16000_ell_TT_yy.txt",
                               unpack=True)
lnl_yy_i = interp1d(np.log(l_n_t), np.log(nl_yy_t), fill_value=-100, bounds_error=False)
def nl_yy_f(l):
    out = np.zeros(len(l))
    ind_in = (l < l_n_t[-1]) & (l > l_n_t[0])
    out[ind_in]=np.exp(lnl_yy_i(np.log(l[ind_in])))
    ind_lo = l <= l_n_t[0]
    out[ind_lo] = nl_yy_t[0]
    ind_hi = l >= l_n_t[-1]
    out[ind_hi] = nl_yy_t[-1]
    return out
    
    
# CCL setup
cosmo = ccl.Cosmology(Omega_c=0.25,
                      Omega_b=0.05,
                      h=0.67,
                      n_s=0.96,
                      sigma8=0.8)
mdef = ccl.halos.MassDef(500, 'critical')
hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mdef)
hmb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef, mass_def_strict=False)
hmc = ccl.halos.HMCalculator(cosmo, hmf, hmb, mdef)
ks = np.geomspace(1E-4, 100, 256)
a_s = np.linspace(0.1, 1, 10)
prof = HaloProfileArnaud(0.2)
szk = SZTracer(cosmo)

# Compute pressure P(k)
pk2d = ccl.halos.halomod_Pk2D(cosmo, hmc, prof,
                              lk_arr=np.log(ks),
                              a_arr=a_s,
                              get_2h=False)

# Limber integral
lmax = 15001
ells = np.unique(np.geomspace(2, lmax, 400).astype(int)).astype(float)
cl_t = ccl.angular_cl(cosmo, szk, szk, ells, p_of_k_a=pk2d)
lcl_i = interp1d(np.log(ells), np.log(cl_t), bounds_error=False, fill_value=-200)
l_all = np.arange(2, lmax+1)
cl = np.exp(lcl_i(np.log(l_all)))
nl = nl_yy_f(l_all)

# Beam
beam_fwhm_amin = 1.4
beam = np.exp(-0.5 * l_all * (l_all + 1) * (beam_fwhm_amin * np.pi / 180 / 60)**2)
cl*=beam**2

# Bandpower window
d_ell = 100
n_bands = 10000 // d_ell
windows = np.zeros([n_bands, lmax-1])
for i in range(n_bands):
    windows[i, i*d_ell+2:(i+1)*d_ell+2] = 1. / d_ell
win = sacc.Window(l_all, windows.T)

# Convolve and Gaussian covariance
bpws = np.dot(windows, cl)
bpws_n = np.dot(windows, nl)
leff = np.dot(windows, l_all)
fsky = 0.1
cov = np.diag((bpws+bpws_n)**2 / ((leff + 0.5) * fsky * d_ell))

# Add statistical noise
#bpws = np.random.multivariate_normal(bpws, cov)

s = sacc.Sacc()
s.add_tracer('Map', 'SO_y', quantity='cmb_tSZ',
             spin=0, ell=l_all, beam=beam)
s.add_ell_cl('cl_00', 'SO_y', 'SO_y',
             leff, bpws, window=win,
             window_id=range(len(leff)))
s.add_covariance(cov)
s.save_fits("data/cl_yy.fits", overwrite=True)

plt.plot(ells, cl_t, 'k-')
plt.plot(l_all, cl, 'r--')
plt.errorbar(leff, bpws, yerr=np.sqrt(np.diag(cov)), fmt='b.')
plt.show()

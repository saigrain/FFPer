import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from george import GP, kernels
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('/Users/aigrain/OneDrive - Nexus365/Documents/Soft/l1periodogram/l1periodogram_codes/')
import l1periodogram_v1, covariance_matrices

MIN_SPAN_CYCLE = 8
    
def extract_activity_basis(x, z, zerr = None, 
                           do_plot = True, do_save = False, \
                           save_name = 'FFper', verbose = False):
    '''Model activity time-series to extract basis terms to use in RV fitting'''

    # divide into seasons
    s = np.floor((x-min(x)+30) / 365).astype(int)
    S_MAX = max(s)
    su = np.unique(s)
    ns = len(su)
    nobs = len(x)

    # Fit long-term component of activity variations
    span = x.max() - x.min()
    if span < (MIN_SPAN_CYCLE * 365): # just use 2nd order polynomial
        x0 = x.min()
        act_long_term = np.polyval(np.polyfit(x-x0, z, 2), x-x0)
        per_cyc = -1
        if verbose: 
            print(f'Time span is only {span/365.35:.2f} years, using polynomial')
        if do_plot:
            fig_LS_act, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,5))
    else: # fit LS periodogram 
        ls = LombScargle(x, z, zerr, nterms = 3)
        freq, pow = ls.autopower(minimum_frequency = 1./(x.max()-x.min()), \
            maximum_frequency=1e-2, samples_per_peak=10)
        per = 1./freq
        ipk = np.argmax(pow)
        per_cyc = per[ipk]
        act_long_term = ls.model(x, freq[ipk])
        if verbose: 
            print(f'Estimated activity cycle period: {per_cyc/365.25:.2f} years')
        if do_plot:
            fig_LS_act, axes = plt.subplots(nrows = 1, ncols = 2, \
                                            figsize = (15,5), width_ratios=[2, 1], sharey = True)
            ax = axes[0]
            per_yr = per / 365.25
            axes[1].axvline(per_cyc/365.25, color = 'C2', ls = '--')
            axes[1].semilogx(per_yr, pow, 'C0-')
            axes[1].set_xlabel('Period (years)')
            axes[1].set_xlim(per_yr.min(), per_yr.max())
    z_corr = z - act_long_term

    # Compute LSper of long-term-corrected activity indicator 
    # season by season to get first estimate of Prot
    freq, pow_ = LombScargle(x, z_corr, zerr).autopower(minimum_frequency = 1e-2, \
        maximum_frequency = 0.3, samples_per_peak = 10)
    per = 1./freq
    if do_plot:
        ax.plot(per, pow_, 'C0-')
        ax.set_ylabel('Power')
        ax.set_xlabel('Period (days)')
        ax.set_xlim(per.min(), per.max())
    tmp = np.zeros_like(per)
    for s_ in su:
        l = s == s_
        ls = LombScargle(x[l],z[l], zerr[l])
        pow_ = ls.power(freq)
        tmp += pow_
        if do_plot:
            ax.plot(per, pow_, lw = 0.5, c = 'k', alpha = 0.3)
    tmp /= len(su)
    if do_plot:
        ax.plot(per, tmp, 'C1-', label = 'average over seasons')
    ipk = np.argmax(tmp)
    per_guess = per[ipk]
    if verbose: 
        print(f'First estimate of rotation period: {per_guess:.2f} days')
    if do_plot:
        ax.axvline(per_guess, alpha = 0.5, color = 'C1', ls = ':')

    # Fit quasi-periodic GP to the long-term-corrected activity time-series
    k = np.var(z_corr) * \
        kernels.ExpSine2Kernel(log_period = np.log(per_guess), gamma=1.0) * \
        kernels.ExpSquaredKernel(metric = (3*per_guess)**2)
    gp = GP(k, mean = z_corr.mean(), fit_mean = True)
    gp.compute(x, zerr)
    def nll(p):
        if np.exp(p[-2]) > 5: # stop period from drifting to excessively large values
            return 1e25
        if np.exp(p[-2]) < 0: # stop period from drifting to excessively small values
            return 1e25
        gp.set_parameter_vector(p)
        try:
            return -gp.log_likelihood(z_corr)
        except:
            return 1e25
    p0 = gp.get_parameter_vector()
    res = minimize(nll, p0)
    gp.set_parameter_vector(res.x)
    per_rot = np.exp(res.x[3])
    if verbose:
        print(f'Refined estimate of rotation period: {per_rot:.2f} days')
    if do_plot:
        ax.axvline(per_guess, color = 'C1', ls = '--')
        plt.tight_layout()
        if do_save:
            plt.savefig(save_name + '_activity_rotation.png')

    # Evaluate G and dG (season by season to avoid edge effects)
    x0 = min(x) - 30
    G = np.zeros(nobs)
    dG = np.zeros(nobs)
    if do_plot:
        fig, ax = plt.subplots(ns,1,sharex = True, sharey = True,\
                               figsize = (8, 1.5 * ns))
    for i,s_ in enumerate(su):
        l = s == s_
        xoff = x0 + i * 365
        gp = GP(k, mean = z_corr[l].mean(), fit_mean = True)
        gp.compute(x[l],zerr[l])
        g = gp.predict(z_corr[l], x[l], return_var=False, return_cov = False)
        G[l] = g
        x_pred = np.linspace(min(x[l]) - 5, max(x[l]) + 5, 1000)
        mu, var = gp.predict(z_corr[l], x_pred, return_var=True)
        dy = (mu[1:] - mu[:-1]) / (x_pred[1] - x_pred[0])
        dx = 0.5 * (x_pred[1:] + x_pred[:-1])
        dg = interp1d(dx, dy, bounds_error = False, fill_value = 0.0)
        dG[l] = dg(x[l])  
        if do_plot:
            dmu = dg(x_pred)
            fac = 2* (mu.max() - mu.min())
            yoff = mu.mean() - fac
            sig = np.sqrt(var)
            ax[i].plot(x[l]-xoff, z_corr[l], '.')
            ax[i].plot(x_pred-xoff, mu, 'C1-')
            ax[i].fill_between(x_pred - xoff, mu + sig, mu - sig, \
                               color = 'C1', \
                               alpha = 0.2, lw = 0)
            ax[i].plot(x_pred-xoff, dmu + yoff, 'C2-')
    if do_plot:
        ax[-1].set_xlabel('time % 1 yr (days)')
        plt.tight_layout()
        if do_save:
            plt.savefig(save_name + '_GP_fit.png')
        
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        ax.plot(x, z, 'k.', ms = 2, label = 'data')
        ax.plot(x, act_long_term, 'C0-', label = 'long-term')
        zm = z.mean()
        off = zm + 1.1 * (min(z-zm) - max(z_corr))
        ax.plot(x, z_corr + off, 'C1.', ms=2,label = 'long-term-corrected')
        ax.plot(x, G + off, 'C2-', lw = 0.5, label = 'G')
        off = zm + 1.1 * (min(z-zm) -  2 * (max(z_corr) - min(z_corr)))
        scal = (G.max() - G.min()) / (dG.max() - dG.min())
        ax.plot(x, dG * scal + off, 'C3-', lw = 0.5, label = 'dG')
        plt.legend(loc=0)
        plt.xlabel('time (days)')
        plt.ylabel('activity indicator')
        plt.xlim(x.min(), x.max())
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name + '_activity_fit.png')

    return (act_long_term, G, dG), (per_cyc, per_rot), res.x

def construct_basis(x, activity_terms, \
                    transiting_planets = None,
                    other_periods = None): 
    n_obs = len(x)    
    n_act = len(activity_terms)
    if transiting_planets is None:
        n_trpl = 0
    else:
        n_trpl = len(transiting_planets)
    if other_periods is None:
        n_other = 0
    else:
        n_other = len(other_periods)
    n_sin = n_trpl + 2 * n_other
    Basis = np.zeros((1 + n_act + n_sin, n_obs))
    Basis[0,:] = np.ones(n_obs)
    for i in range(n_act):
        tmp = activity_terms[i]
        mi, ma = tmp.min(), tmp.max()
        tmp -= mi
        tmp /= (ma-mi)
        Basis[1+i,:] = tmp
    for i in range(n_trpl):
        period, tc = transiting_planets[i]
        Basis[1 + n_act + i] = np.sin(2 * np.pi * (x - tc) / period)        
    for i in range(n_other):
        Basis[1 + n_act + n_trpl + i] = \
            np.sin(2 * np.pi * (x / other_periods[i]))
        Basis[1 + n_act + n_trpl + i + 1] = \
            np.cos(2 * np.pi * (x / other_periods[i]))
    return Basis

def fit_basis(x, y, yerr, basis, flags = None,
              do_plot = True, do_save = False, save_name = 'FFper',
              verbose = False):
    if np.ndim(yerr) == 0:
        weights = np.ones(len(x)) / yerr**2
    else:
        weights = 1.0 / yerr**2
    if flags is None:
        flags = np.zeros(basis.shape[0], 'bool')
        flags[:4] = True
    ymean = (y * weights).sum() / weights.sum()
    chi2_constant = (((y - ymean) * weights)**2).sum()
    model = LinearRegression(fit_intercept = False)
    reg = model.fit(basis.T, y, sample_weight = weights)
    yfit = reg.predict(basis.T)
    resid = y - yfit
    coeff = reg.coef_
    yrect = y - coeff[0]
    l = np.copy(flags) 
    l[0] = False
    yfit_act_only = np.dot(coeff[l], basis[l,:])
    l = ~flags    
    yfit_pl_only = np.dot(coeff[l], basis[l,:])
    if do_plot:
        freq, pow = LombScargle(x,yrect, yerr).autopower(samples_per_peak=10)
        per = 1/freq
        pow_act = LombScargle(x,yfit_act_only, yerr).power(freq)
        pow_noact = LombScargle(x,yrect-yfit_act_only, yerr).power(freq)
        pow_pl = LombScargle(x,yfit_pl_only, yerr).power(freq)
        pow_resid = LombScargle(x,resid, yerr).power(freq)
        fig, ax = plt.subplots(2, 1, figsize = (10, 8))
        ax[0].plot(per, pow, 'k-', alpha = 0.5, label = 'data')
        ax[0].plot(per, pow_act- 0.1, 'C0-', alpha = 0.5, label = 'activity')
        ax[0].plot(per, pow_noact - 0.2, 'C1-', alpha = 0.5, label = 'activity-corrected')
        ax[0].plot(per, pow_pl - 0.3, 'C2-', alpha = 0.5, label = 'planets')
        ax[0].plot(per, pow_resid - 0.4, 'C3-', alpha = 0.5, label = 'residuals')
        ax[0].set_ylabel('power')
        ax[0].set_xlabel('period (days)')
        plt.sca(ax[0])
        plt.semilogx()
        plt.xlim(per.min(), per.max())
        plt.legend(loc = 0)
        ax[1].plot(x, yrect, 'k.', ms = 2)
        ax[1].plot(x, yfit_act_only, 'C0.', ms = 2)
        off = 1.1 * (yrect.min() - (yrect - yfit_act_only).max())
        ax[1].plot(x, yrect - yfit_act_only + off, 'C1.', ms = 2)
        ax[1].plot(x, yfit_pl_only + off, 'C2.', ms = 2)
        off += 1.1 * ((yrect - yfit_act_only).min() - resid.max())
        ax[1].plot(x, resid + off, 'C3.', ms = 2)
        ax[1].set_ylabel('RV (m/s)')
        ax[1].set_xlabel('time (MJD)')
        plt.tight_layout()
        if do_save:
            plt.savefig(save_dir + save_name + '_fit.png')
    return resid, coeff

def run_l1_periodogram(x, y, s, yerr, basis = None, per_min = 1.1,
                       sig_add_w = 0.5, fap_threshold = 0.05,
                       n_pk_eval_max = 10):
    
    if basis is None:
        basis = np.zeros((1, len(x)))

    x0 = np.floor(x.min())
    t = x - x0
    
    c = l1periodogram_v1.l1p_class(t,y)
    sigmaW = sig_add_w # add in quadrature 0.5 m/s to the nominal uncertainties 
    sigmaR, tau = 0. ,0. # no red noise
    sigma_calib = 0. # no calibration noise
    V = covariance_matrices.covar_mat(t, np.ones(len(c.t)) * yerr, \
                                      sigmaW, sigmaR, sigma_calib, tau)
    c.set_model(omegamax = 2*np.pi/per_min, 
                V = V,
                MH0 = basis.T, 
                verbose=0)
    c.l1_perio(numerical_method='lars',
               significance_evaluation_methods = ['fap','evidence_laplace'],
               max_n_significance_tests=n_pk_eval_max,
               verbose=0, plot_output = False)

    periods = 2*np.pi/c.omega_peaks
    npk = len(periods)
    if npk > n_pk_eval_max:
        periods = periods[:n_pk_eval_max]
    faps = c.significance['log10faps']
    l = faps < np.log10(fap_threshold)
    periods = periods[l]
    faps = faps[l]
    so = np.argsort(faps)
    return periods[so], faps[so]

if __name__ == "__main__":

    # TODO:
    # - read in from .mat, .sav or .rdb file
    # - user specifies whether to produce plots and where to save them
    do_plot = True
    do_save = False
    save_name = 'FFPer'
    verbose = True
    # - user specifies which activity indicator is used
    # - user specifies how much WN to add (both to data and to uncerts)
    WN = 0.3
    # - user specifies how many seasons to use
    S_MAX = 4
    # - user specifies any transiting planets
    transiting_planets = []
    # - user specifies whether to run a fit with no planets /
    #   transiting planets only
    fit_known_only = True
    # - user specifies max no. peaks and FAP threshold in L1 periodogram
    max_n_peaks = 1
    FAP_threshold = 0.05
    
    # read in data
    from scipy.io import loadmat
    root = '/Users/aigrain/Data/meunier2024/blind/data/'
    fl = root + 'my_serie_res_bt_G2_1000_4m_NOISE0.09_OGS_MAG_real3.mat'    
    d = loadmat(fl)
    x = np.array(d['tt']).flatten()
    y = np.array(d['rv']).flatten()
    z = np.array(d['ca']).flatten()
    y_sig = np.zeros_like(y) + 0.09 # see M+23, section 2.1.3
    z_sig = np.zeros_like(z) + 5e-4 # see M+23, section 2.1.3
    s = np.floor((x-min(x)+30) / 365).astype(int)

    # add white noise if requested
    if WN > 0:
        y += np.random.normal(0, WN, len(y))    
        y_sig = np.sqrt(y_sig**2 + WN**2)
        
    # select seasons
    l = s < S_MAX
    x = x[l]
    y = y[l]
    y_sig = y_sig[l]
    z = z[l]
    z_sig = z_sig[l]

    # specify (known) transiting planets
    transiting_planets = []
    per_inj = []
    
    # extract activity basis
    activity_terms, activity_periods, GP_par = \
        extract_activity_basis(x, z, zerr = z_sig,
                               do_plot = do_plot, do_save = do_save,
                               save_name = save_name, verbose = verbose)
    per_cyc, per_rot = activity_periods
    known_basis = construct_basis(x, activity_terms, transiting_planets)

    if fit_known_only:
        resid, coeff = fit_basis(x, y, y_sig, known_basis, \
                                 do_plot = do_plot, do_save = do_save, \
                                 save_name = save_name + '_known', \
                                 verbose = verbose)

    # Run L1 periodogram, including activity basis set as unpenalised vectors
    if max_n_peaks > 0:
        new_periods, faps = run_l1_periodogram(x, y, s, y_sig, \
                                               basis = known_basis, 
                                               sig_add_w = 0.0, \
                                               fap_threshold = FAP_threshold, \
                                               n_pk_eval_max = max_n_peaks)
        if verbose:
            for i, per in enumerate(new_periods):
                print(f'New signal {i+1}: period {per:.3f}, log10FAP {faps[i]:.2e}')
    else:
        new_periods = []
        
    # Add newly found period(s) to basis and perform fit again
    if len(new_periods) > 0:
        extended_basis = construct_basis(x, activity_terms, \
                                         transiting_planets, new_periods)

        # *** will need to specify flags here ***
        
        resid, coeff = fit_basis(x, y, y_sig, extended_basis, \
                                 do_plot = do_plot, do_save = False, \
                                 verbose = verbose)

    if do_plot:
        # Add newly vertical lines to periodogram in last figure
        plt.sca(plt.gcf().axes[0])
        for i in range(len(new_periods)):
            plt.axvline(new_periods[i], color = 'C2', ls = 'dashed', alpha = 0.5)
        plt.axvline(per_rot, color = 'C0', ls = 'dashed', alpha = 0.5)
        plt.axvline(per_rot/2, color = 'C0', ls = 'dashed', alpha = 0.5)
        if per_cyc>0:
            plt.axvline(per_cyc, color = 'C0', ls = 'dotted', alpha = 0.5)
        for i in range(len(per_inj)):
            plt.axvline(per_inj[i], color = 'C3', ls = 'dotted', alpha = 0.5)
        if do_save:
            plt.savefig(save_dir + save_name + '_fit.png')

    
    # amp = np.sqrt(coeff[4]**2 + coeff[5]**2) # factor 2 bc basis term normalised to 1 pk2pk
    # pha = np.arctan2(coeff[4], coeff[5]) / np.pi /2
    # if pha < 0:
    #     pha += 1
    # lin = f"{i:3d} {per_cyc:7.1f} {per_rot:6.2f} {per_inj[i]:8.3f} {periods[0]:8.3f} "
    # lin += f"{amp_inj[i]:6.2f} {amp:6.2f} {pha_inj[i]:7.2f} {pha:7.2f} {faps[0]:10.2e}"
    # with open(outfl, 'a') as f:
    #     f.write(lin + "\n")
    # print(lin)


    
    plt.show()

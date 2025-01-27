import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from george import GP, kernels
from scipy.optimize import minimize
from scipy.interpolate import interp1d

MIN_SPAN_CYCLE = 8
    
def extract_activity_basis(x, z, zerr = None, 
                           do_plot = True, do_save = False, save_name = 'FFper', verbose = False):
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
        fig, ax = plt.subplots(ns,1,sharex = True, sharey = True,figsize=(8,1.5*ns))
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
            fac = 3* (mu.max() - mu.min())
            yoff = mu.mean() - fac
            sig = np.sqrt(var)
            ax[i].plot(x[l]-xoff, z_corr[l], '.')
            ax[i].plot(x_pred-xoff, mu, 'C1-')
            ax[i].fill_between(x_pred-xoff, mu+sig, mu-sig, color='C1', alpha=0.2,lw=0)
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

if __name__ == "__main__":

    from scipy.io import loadmat
    root = '/Users/aigrain/Data/meunier2024/blind/data/'
    fl = root + 'my_serie_res_bt_G2_1000_4m_NOISE0.09_OGS_MAG_real1.mat'    
    d = loadmat(fl)
    x = np.array(d['tt']).flatten()
    y = np.array(d['rv']).flatten()
    z = np.array(d['ca']).flatten()
    WN = 0.3 # White noise to add to data to make it HARPS3-like
    y += np.random.normal(0, WN, len(y))    
    y_sig = np.zeros_like(y) + np.sqrt(0.09**2 + WN**2) # see M+23, section 2.1.3
    z_sig = np.zeros_like(z) + 5e-4 # see M+23, section 2.1.3
    s = np.floor((x-min(x)+30) / 365).astype(int)

    l = s < 10
    x = x[l]
    y = y[l]
    y_sig = y_sig[l]
    z = z[l]
    z_sig = z_sig[l]
    
    res = extract_activity_basis(x, z, zerr = z_sig,
                                 do_plot = True, do_save = False,
                                 save_name = 'FFper', verbose = True)
    plt.show()

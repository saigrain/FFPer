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
        fit = np.polyval(np.polyfit(x-x0, z, 2), x-x0)
        per_cyc = -1
        if verbose: 
            print(f'Time span is only {span/365.35:.2f} years, using polynomial')
        if do_plot:
            fig_LS_act, ax = plt.subplots(nrows = 1, ncol = 1, figsize = (10,5))
    else: # fit LS periodogram 
        ls = LombScargle(x, z, zerr, nterms = 3)
        freq, pow = ls.autopower(minimum_frequency = 1./(x.max()-x.min()), \
            maximum_frequency=1e-2, samples_per_peak=10)
        per = 1./freq
        ipk = np.argmax(pow)
        per_cyc = per[ipk]
        fit = ls.model(x, freq[ipk])
        if verbose: 
            print(f'Estimated activity cycle period: {per_cyc/365.25:.2f} years')
        if do_plot:
            fig_LS_act, axes = plt.subplots(nrows = 1, ncol = 2, \
                figsize = (15,5), width_ratios=[2, 1])
            ax = axes[0]
            axes[1].axvline(per_cyc, color = 'C2', ls = '--', alpha = 0.5)
            axes[1].semilogx(per, pow, 'C0-')
            axes[1].set_xlabel('Power')
            axes[1].set_xlabel('Period (days)')
            axes[1].set_xlim(per.min(), per.max())
    z_corr = z - act_long_term

    # Compute LSper of long-term-corrected activity indicator 
    # season by season to get first estimate of Prot
    freq, pow = LombScargle(x, z_corr, zerr).autopower(minimum_frequency = 1e-2, \
        maximum_frequency = 0.3, samples_per_peak = 10)
    per = 1./freq
    if do_plot:
        fig, ax = plt.subplots(1,1,sharex = True, figsize=(8,3))
        ax.plot(per, pow_ca, 'C0-', lw = 0.5)
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
        plt.axvline(per_guess, alpha = 0.5, color = 'C1')
        plt.tight_layout()
        if do_save:
            plt.savefig(save_name + '_activity_rotation.png')

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

    # Evaluate G and dG (season by season to avoid edge effects)
    x0 = min(x) - 30
    G = np.zeros(nobs)
    dG = np.zeros(nobs)
    if do_plot:
        fig, ax = plt.subplots(ns,1,sharex = True, sharey = True,figsize=(8,1.5*ns))
    for i,s_ in enumerate(su):
        l = s == s_
        xoff = x0 + i * 365
        gp = GP(k, mean = z[l].mean(), fit_mean = True)
        gp.compute(x[l],zerr[l])
        g = gp.predict(z[l], x[l], return_var=False, return_cov = False)
        G[l] = g
        x_pred = np.linspace(min(x[l]) - 5, max(x[l]) + 5, 1000)
        mu, var = gp.predict(z[l], x_pred, return_var=True)
        dy = (mu[1:] - mu[:-1]) / (x_pred[1] - x_pred[0])
        dx = 0.5 * (x_pred[1:] + x_pred[:-1])
        dg = interp1d(dx, dy, bounds_error = False, fill_value = 0.0)
        dG[l] = dg(x[l])  
        if do_plot:
            dmu = dg(x_pred)
            fac = 3* (mu.max() - mu.min())
            yoff = mu.mean() - fac
            sig = np.sqrt(var)
            ax[i].plot(x[l]-xoff, z[l], '.')
            ax[i].plot(x_pred-xoff, mu, 'C1-')
            ax[i].fill_between(x_pred-xoff, mu+sig, mu-sig, color='C1', alpha=0.2,lw=0)
            ax[i].plot(x_pred-xoff, dmu + yoff, 'C2-')
    if do_plot:
        ax[-1].set_xlabel('time % 1 yr (days)')
        plt.tight_layout()
        if do_save:
            plt.savefig(save_name + '_GP_fit.png')
        
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax.plot(x, z, 'k.', ms = 2, label = 'data')
        ax.plot(x, act_long_term, 'C0.', ms = 2, label = 'long-term')
        zm = z.mean()
        off = zm + 1.1 * (min(z-zm) - max(z_corr))
        ax.plot(x, z_corr + off, 'C1', label = 'long-term-corrected')
        ax.plot(x, G + off, 'C2', label = 'G')
        off = zm + 1.1 * (min(z-zm) -  2 * (max(z_corr) - min(z_corr)))
        scal = (G.max() - G.min()) / (dG.max() / dG.min())
        ax.plot(x, dG * scal + off, 'C3', label = 'dG')
        plt.legend(loc=0)
        plt.xlabel('time (days)')
        plt.ylabel('activity indicator')
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name + '_activity_fit.png')

    return (act_long_term, G, dG), (per_cyc, per_rot), res.x

if __name__ == "__main__":
    print("Hello, World!")
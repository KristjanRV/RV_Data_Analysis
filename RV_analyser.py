import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import seaborn as sns


class RV:
    def __init__(self, data, telescope_names=None, colors=None, f_minimum=None, f_maximum=None):
        """
        Initializes the RV class with input data (time, radial velocity, uncertainties).
        Allows for setting custom telescope names, plot colors, and frequency limits.
        """
        self.telescope_names = telescope_names
        self.colors = colors if colors else sns.color_palette("husl", len(data))

        # Unpack time, radial velocity, and uncertainty data for each telescope
        self.t, self.rv, self.u_rv = [], [], []
        for i in range(len(data)):
            t_i, rv_i, u_rv_i = data[i][:, 0], data[i][:, 1], data[i][:, 2]
            self.t.append(t_i)
            self.rv.append(rv_i)
            self.u_rv.append(u_rv_i)

        # Flattened arrays for calculations across all datasets
        self.flat_t = np.concatenate(self.t)
        self.flat_rv = np.concatenate(self.rv)
        self.flat_u_rv = np.concatenate(self.u_rv)

        # Print statistics on uncertainties (can be removed if not needed)
        print('Median uncertainty:', np.median(self.flat_u_rv))
        print('Mean uncertainty:', np.mean(self.flat_u_rv))

        # Frequency grid calculation based on the input time range
        T = np.max(self.flat_t) - np.min(self.flat_t)  # Total time span
        f_min = 1 / T  # Minimum frequency
        delta_f = 0.1 * f_min  # Frequency step based on user-defined nu

        # Optionally adjust min/max frequency
        delta_t = np.diff(self.flat_t)
        f_max = np.median(1 / delta_t)
        if f_minimum is not None:
            f_min = f_minimum
        if f_maximum is not None:
            f_max = f_maximum

        self.f = np.arange(f_min, f_max, delta_f)  # Frequency grid
        
        print('Min period:', 1/f_min, 'Max period:', 1/f_max)


    def remove_linear_trend(self):
        """
        Removes a linear trend from the RV data for each telescope using weighted linear fit.
        """
        for i in range(len(self.rv)):
            # Fit a linear trend to each dataset and subtract it
            coefficients_i = np.polyfit(self.t[i], self.rv[i], deg=1, w=1 / self.u_rv[i] ** 2)
            fit_line_i = np.poly1d(coefficients_i)
            self.rv[i] -= fit_line_i(self.t[i])

        # Update flattened radial velocity array
        self.flat_rv = np.concatenate(self.rv)


    def remove_weighted_mean(self):
        """
        Removes the weighted mean from the RV data for each telescope dataset.
        """
        for i in range(len(self.rv)):
            wm = np.average(self.rv[i], weights=1 / self.u_rv[i] ** 2)
            self.rv[i] -= wm
        self.flat_rv = np.concatenate(self.rv)
    

    def plot_data(self):
        """
        Plots the RV data with error bars for each telescope.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot each dataset with error bars
        for i in range(len(self.t)):
            telescope_label = self.telescope_names[i] if self.telescope_names else f'Telescope {i+1}'
            ax.errorbar(self.t[i], self.rv[i], yerr=self.u_rv[i], fmt='o', label=telescope_label, 
                        color=self.colors[i], markersize=5)

        # Configure plot appearance
        ax.set_xlabel('Time [days]', fontsize=12)
        ax.set_ylabel('Radial velocity [m/s]', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


    def periodogram(self, t=None, rv=None, u_rv=None):
        """
        Computes the Lomb-Scargle periodogram and plots it. Optionally, specify custom time, RV, and uncertainties.
        """
        rv_data = rv if rv is not None else self.flat_rv
        t_data = t if t is not None else self.flat_t
        u_rv_data = u_rv if u_rv is not None else self.flat_u_rv

        # Compute the Lomb-Scargle periodogram
        p = LombScargle(t_data, rv_data, dy=u_rv_data).power(self.f)

        # Plot the periodogram
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(self.f, p, color='tab:purple')

        # Mark the frequency with the highest power
        max_power_index = np.argmax(p)
        ax1.vlines(self.f[max_power_index], ymax=np.max(p), ymin=0, color='tab:orange', 
                   label=f'P_max = {round(1/self.f[max_power_index], 1)} days')

        # Compute and plot false alarm probabilities (FAP)
        p_synth = self.non_parametric_bootstrap(rv_data)
        fap_10 = np.percentile(p_synth, 90)
        fap_1 = np.percentile(p_synth, 99)
        ax1.hlines(y=fap_10, xmin=0, xmax=np.max(self.f), color='yellow', label='10% FAP', linestyles='dashed')
        ax1.hlines(y=fap_1, xmin=0, xmax=np.max(self.f), color='tab:red', label='1% FAP', linestyles='dashed')

        # Add secondary x-axis for periods
        ax2 = ax1.secondary_xaxis('top')
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels([f'{1 / tick:.1f}' for tick in ax1.get_xticks()])
        ax2.set_xlabel('Period [days]', fontsize=12)

        ax1.legend(loc='lower left')
        ax1.set_xlabel('Frequency [1/days]', fontsize=12)
        ax1.set_ylabel('Power', fontsize=12)
        plt.tight_layout()
        plt.show()

        return p


    def non_parametric_bootstrap(self, rv=None, nsim=1000):
        """
        Performs non-parametric bootstrapping to generate synthetic datasets and compute the false alarm probability (FAP).
        """
        from sklearn.utils import resample
        rv_data = rv if rv is not None else self.flat_rv
        p_synth = []

        for _ in range(nsim):
            rv_bootstrap = resample(rv_data, replace=True)
            p = LombScargle(self.flat_t, rv_bootstrap, dy=self.flat_u_rv).power(self.f)
            p_synth.append(np.max(p))

        return np.sort(p_synth)


    def fold_and_plot(self, period):
        """
        Folds the RV data over a specified period and plots it.
        """
        phase_fold = self.flat_t % period  # Calculate phase for folding
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot folded data
        start_idx = 0
        for i in range(len(self.t)):
            telescope_label = self.telescope_names[i] if self.telescope_names else f'Telescope {i+1}'
            end_idx = start_idx + len(self.t[i])
            ax.errorbar(phase_fold[start_idx:end_idx], self.flat_rv[start_idx:end_idx],
                        yerr=self.flat_u_rv[start_idx:end_idx], fmt='o', label=telescope_label, 
                        color=self.colors[i], markersize=5)
            start_idx = end_idx

        ax.set_xlabel('Phase [days]', fontsize=12)
        ax.set_ylabel('Radial velocity [m/s]', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

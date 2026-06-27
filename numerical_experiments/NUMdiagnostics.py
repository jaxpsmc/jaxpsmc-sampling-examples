"""
Diagnostics for NUMnumerical_experiments.py. This file contains only code rlated to diagnostic.
The class expects the experiment runner to provide attributes such as self.params, 
self.true_samples, self.samples, and self.out so that the diagnostics can be calculated
"""

import json
import logging
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class Diagnostics:
    """Diagnostics, plots, sample export, and sample statistics."""

    def get_true_and_mcmc_samples(self, discard=0, thin=1):
        dim = int(self.params["n_dims"])
        # true samples
        if not hasattr(self, "true_samples") or self.true_samples is None:
            raise ValueError("No true samples found. Ensure self.true_samples is set.")

        true_np = np.asarray(self.true_samples).reshape(-1, dim)

        # sampler samples
        if hasattr(self, "samples") and self.samples is not None:
            samp = np.asarray(self.samples).reshape(-1, dim)
            samp = samp[int(discard)::int(thin), :]
            mcmc_np = samp
        else:
            raise ValueError(
                "No sampler samples found. Run run_experiment() first. "
            )

        return true_np, mcmc_np
    
    #===========================================================
    # 1.1. PLOT CORNER
    #===========================================================
    def plot_corner(self, seed=2046):
        """
        Corner plot: ground truth vs posterior samples
        """
        # get samples 
        true_np, mcmc_np = self.get_true_and_mcmc_samples()

        dim = int(self.params["n_dims"])
        labels = [f"x{i}" for i in range(dim)]

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        # plot posterior samples from sampler first
        fig = corner.corner(mcmc_np, color="blue", hist_kwargs={"color": "blue", "density": True},
                            show_titles=True, labels=labels,)

        # Overlay with ground truth samples
        corner.corner(true_np, fig=fig, color="red", hist_kwargs={"color": "red", "density": True},
                      show_titles=True, labels=labels,)

        # Legend
        handles = [plt.Line2D([], [], color="blue", label="sampler"),
                   plt.Line2D([], [], color="red", label="True Normal"),]
        fig.legend(handles=handles, loc="upper right")

        save_name = os.path.join(outdir, "true_vs_mcmc_corner_plot.pdf")
        fig.savefig(save_name, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved overlay corner plot to {save_name}")



    #===========================================================
    # 1.2. PLOT DIAGNOSTICS 
    #=========================================================== 
    def plot_diagnostics(
        self,
        filename: str = "diagnostics.pdf",
    ) -> None:
        """
        Two-page PDF with core SMC diagnostics.
        Page 1:
            1. beta(t)
            2. ESS(t)
            3. logZ(t)
        Page 2:
            4. acceptance(t)
            5. proposal scale sigma(t)
        """
        if not hasattr(self, "out") or self.out is None:
            raise ValueError("No JAX sampler output found. Run run_experiment() first.")

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        T = int(np.asarray(self.out.state.t))
        if T < 2:
            raise ValueError(f"Not enough iterations recorded (t={T}).")

        # trajectory arrays
        it = np.arange(T)

        beta = np.asarray(self.out.state.beta[:T]).reshape(-1)
        ess = np.asarray(self.out.state.ess[:T]).reshape(-1)
        accept = np.asarray(self.out.state.accept[:T]).reshape(-1)
        logz = np.asarray(self.out.state.logz[:T]).reshape(-1)
        eff = np.asarray(self.out.state.efficiency[:T]).reshape(-1)

        # experiment parameters
        n_active = int(self.params["n_active"])
        n_dims = int(self.params["n_dims"])

        # proposal scale normalization, same convention as mutate()
        norm_ref = 2.38 / np.sqrt(n_dims)

        # sigma is meaningful only once beta > 0
        mask_sigma = beta > 0.0
        it_sigma = it[mask_sigma]
        sigma = eff[mask_sigma] * norm_ref

        # diagnostic ratio
        ess_ratio = ess / max(1, n_active)

        # figure size
        page_width = 13.0      
        panel_height = 4.0     
        extra_height = 1.0   

        figsize_page1 = (page_width, 3 * panel_height + extra_height)  
        figsize_page2 = (page_width, 2 * panel_height + extra_height)  

        marker_size = 3
        line_width = 1.0

        save_path = os.path.join(outdir, filename)

        with PdfPages(save_path) as pdf:

        # PAGE 1: beta, ESS, logZ
            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=figsize_page1,
                sharex=False,
                constrained_layout=True,
            )
            fig.suptitle("SMC core diagnostics: page 1", fontsize=14)

            # 1. beta(t)
            ax = axes[0]
            ax.plot(it, beta, marker="o", markersize=marker_size, linewidth=line_width)
            ax.set_title("beta(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("beta")
            ax.set_ylim(min(-0.02, beta.min()), max(1.02, beta.max()))
            ax.grid(True, alpha=0.3)

            # 2. ESS(t)
            ax = axes[1]
            ax.plot(it, ess, marker="o", markersize=marker_size, linewidth=line_width, label="ESS")
            ax.plot(
                it,
                ess_ratio * n_active,
                linestyle="--",
                linewidth=line_width,
                label="ESS/N_active × N_active",
            )
            ax.axhline(n_active, linestyle=":", linewidth=line_width, label="N_active")
            ax.set_title("ESS(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("ESS")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            # 3. logZ(t)
            ax = axes[2]
            ax.plot(it, logz, marker="o", markersize=marker_size, linewidth=line_width)
            ax.set_title("logZ(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("logZ")
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # PAGE 2: acceptance, sigma
            fig, axes = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=figsize_page2,
                sharex=False,
                constrained_layout=True,
            )
            fig.suptitle("SMC core diagnostics: page 2", fontsize=14)

            # 4. acceptance(t)
            ax = axes[0]
            ax.plot(it, accept, marker="o", markersize=marker_size, linewidth=line_width)
            ax.set_title("acceptance rate")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("accept")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

            # 5. sigma(t)
            ax = axes[1]
            ax.plot(it_sigma, sigma, marker="o", markersize=marker_size, linewidth=line_width)
            ax.set_title("proposal scale sigma(t)  (beta > 0)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("sigma")
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved core diagnostics PDF to {save_path}")


    #===========================================================
    # 1.3. KL METRIC USING DIAGONAL COVARIANCE APPROXIMATION 
    #===========================================================
    def gau_kl(self, pm: np.ndarray, pv: np.ndarray, qm: np.ndarray, qv: np.ndarray) -> float:
        """
        Parametric Gaussian KL using diagonal covariance approximation.
        Computes KL(N(pm, diag(pv)) || N(qm, diag(qv))).
        """
        eps = 1e-12
        pv = np.asarray(pv, dtype=float) + eps
        qv = np.asarray(qv, dtype=float) + eps
        pm = np.asarray(pm, dtype=float)
        qm = np.asarray(qm, dtype=float)

        d = pm.shape[0]
        return 0.5 * (
            np.sum(np.log(qv / pv))
            + np.sum(pv / qv)
            + np.sum((qm - pm) ** 2 / qv)
            - d
        )


    def _gaussian_component_logpdf(
        self,
        xs: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        jitter: float = 1e-10,
    ) -> np.ndarray:
        """
        Computes unweighted Gaussian component log densities.

        Returns:
            logpdfs: shape (N, K)
        """
        xs = np.asarray(xs, dtype=float)
        means = np.asarray(means, dtype=float)
        covs = np.asarray(covs, dtype=float)

        n, d = xs.shape
        k = means.shape[0]

        logpdfs = np.empty((n, k), dtype=float)
        eye = np.eye(d)

        for j in range(k):
            cov_j = covs[j] + jitter * eye
            chol_j = np.linalg.cholesky(cov_j)

            diff = xs - means[j]
            sol = np.linalg.solve(chol_j, diff.T).T
            quad = np.sum(sol * sol, axis=1)
            logdet = 2.0 * np.sum(np.log(np.diag(chol_j)))

            logpdfs[:, j] = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

        return logpdfs



    #===========================================================
    # 1.4. MODE MASS RECOVERY
    #===========================================================
    def _compute_mode_mass_recovery(self, true_samples: np.ndarray, mcmc_samples: np.ndarray) -> dict:
        """
        Mode mass recovery using paper method partition:

            S_k = {x : k = argmax_j log N(x; mu_j, Sigma_j)}

        The true mode mass is estimated from the generated reference samples.
        The sampler mode mass is estimated from posterior samples.
        """
        means = np.asarray(self.mcmc_means, dtype=float)
        covs = np.asarray(self.mcmc_covs, dtype=float)
        n_components = means.shape[0]

        true_logpdf = self._gaussian_component_logpdf(true_samples, means, covs)
        mcmc_logpdf = self._gaussian_component_logpdf(mcmc_samples, means, covs)

        true_labels = np.argmax(true_logpdf, axis=1)
        mcmc_labels = np.argmax(mcmc_logpdf, axis=1)

        true_mass = np.bincount(true_labels, minlength=n_components) / true_labels.size
        sampler_mass = np.bincount(mcmc_labels, minlength=n_components) / mcmc_labels.size

        abs_error = np.abs(sampler_mass - true_mass)

        return {
            "true_mass": true_mass,
            "sampler_mass": sampler_mass,
            "abs_error": abs_error,
            "l1_error": float(np.sum(abs_error)),
            "mean_abs_error": float(np.mean(abs_error)),
            "max_abs_error": float(np.max(abs_error)),
        }



    #===========================================================
    # 1.5. MAXIMUM MEAN DISCREPANCY
    #===========================================================
    def _rbf_kernel_matrix(self, x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x2 = np.sum(x * x, axis=1)[:, None]
        y2 = np.sum(y * y, axis=1)[None, :]
        dist2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)

        return np.exp(-dist2 / (2.0 * sigma * sigma))


    def _median_bandwidth(self, x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> float:
        z = np.vstack([x, y])

        if z.shape[0] > max_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(z.shape[0], size=max_points, replace=False)
            z = z[idx]

        z2 = np.sum(z * z, axis=1)[:, None]
        dist2 = np.maximum(z2 + z2.T - 2.0 * z @ z.T, 0.0)

        upper = dist2[np.triu_indices_from(dist2, k=1)]
        upper = upper[upper > 0.0]

        if upper.size == 0:
            return 1.0

        return float(np.sqrt(np.median(upper)))


    def _mmd2_unbiased_rbf(self, x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        Unbiased quadratic-time MMD^2 estimator with RBF kernel.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        m = x.shape[0]
        n = y.shape[0]

        if m < 2 or n < 2:
            raise ValueError("MMD requires at least two true samples and two sampler samples.")

        k_xx = self._rbf_kernel_matrix(x, x, sigma)
        k_yy = self._rbf_kernel_matrix(y, y, sigma)
        k_xy = self._rbf_kernel_matrix(x, y, sigma)

        sum_xx = np.sum(k_xx) - np.trace(k_xx)
        sum_yy = np.sum(k_yy) - np.trace(k_yy)
        sum_xy = np.sum(k_xy)

        mmd2 = (
            sum_xx / (m * (m - 1))
            + sum_yy / (n * (n - 1))
            - 2.0 * sum_xy / (m * n)
        )

        return float(mmd2)


    def _compute_mmd_diagnostics(self, true_samples: np.ndarray, mcmc_samples: np.ndarray) -> dict:
        """
        MMD between true target samples and sampler samples.
        Uses RBF kernels with bandwidths based on the median heuristic.
        """
        sigma_med = self._median_bandwidth(true_samples, mcmc_samples)

        bandwidths = {
            "0.5 * median": 0.5 * sigma_med,
            "1.0 * median": 1.0 * sigma_med,
            "2.0 * median": 2.0 * sigma_med,
            "4.0 * median": 4.0 * sigma_med,
        }

        out = {
            "median_bandwidth": sigma_med,
            "values": {},
        }

        for name, sigma in bandwidths.items():
            out["values"][name] = {
                "sigma": float(sigma),
                "mmd2_unbiased": self._mmd2_unbiased_rbf(true_samples, mcmc_samples, sigma),
            }

        return out



    #===========================================================
    # 1.6. SLICED WASSERSTEIN DISTANCE
    #===========================================================
    def _sliced_wasserstein_2(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_projections: int = 128,
        seed: int = 0,
    ) -> dict:
        """
        Empirical sliced Wasserstein distance with p=2.
        Implements:
        SW_2^2 = mean_theta mean_i |sort(theta^T x)_i - sort(theta^T y)_i|^2
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        n = min(x.shape[0], y.shape[0])
        if n < 2:
            raise ValueError("Sliced Wasserstein requires at least two samples per distribution.")

        rng = np.random.default_rng(seed)

        if x.shape[0] != n:
            x = x[rng.choice(x.shape[0], size=n, replace=False)]
        if y.shape[0] != n:
            y = y[rng.choice(y.shape[0], size=n, replace=False)]

        d = x.shape[1]

        directions = rng.normal(size=(n_projections, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        x_proj = x @ directions.T
        y_proj = y @ directions.T

        x_proj = np.sort(x_proj, axis=0)
        y_proj = np.sort(y_proj, axis=0)

        sw2_squared = float(np.mean((x_proj - y_proj) ** 2))
        sw2 = float(np.sqrt(sw2_squared))

        return {
            "n_samples": int(n),
            "n_projections": int(n_projections),
            "sw2_squared": sw2_squared,
            "sw2": sw2,
        }



    def write_diagnostic_statistics(
        self,
        filename: str = "statistics.txt",
    ) -> str:
        """
        Writes one combined diagnostic file containing:
            sample statistics
            mode mass recovery
            parametric Gaussian KL divergence
            maximum mean discrepancy
            sliced Wassertein distance
        """
        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        true_samples, mcmc_samples = self.get_true_and_mcmc_samples()

        # 1. Sample statistics
        self.pm = mcmc_samples.mean(axis=0)
        self.pv = mcmc_samples.var(axis=0)
        self.ps = mcmc_samples.std(axis=0)

        self.qm = true_samples.mean(axis=0)
        self.qv = true_samples.var(axis=0)
        self.qs = true_samples.std(axis=0)

        self.mcmc_samples = mcmc_samples
        self.true_samples_np = true_samples

        # 2. Mode mass recovery
        mass = self._compute_mode_mass_recovery(true_samples, mcmc_samples)
        # 3. KL divergence
        kl_val = float(self.gau_kl(self.pm, self.pv, self.qm, self.qv))
        # 4. MMD
        mmd = self._compute_mmd_diagnostics(true_samples, mcmc_samples)
        # 6. SLICED WASSERSTEIN 
        sw = self._sliced_wasserstein_2(true_samples, mcmc_samples)

        # Save one file with metrics
        out_path = os.path.join(outdir, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("1. SAMPLE STATISTICS\n")
            f.write("====================\n\n")

            f.write("MCMC sample mean:\n")
            f.write(np.array2string(self.pm, precision=6, suppress_small=True))
            f.write("\n\n")

            f.write("MCMC sample variance:\n")
            f.write(np.array2string(self.pv, precision=6, suppress_small=True))
            f.write("\n\n")

            f.write("MCMC sample std:\n")
            f.write(np.array2string(self.ps, precision=6, suppress_small=True))
            f.write("\n\n")

            f.write("True sample mean:\n")
            f.write(np.array2string(self.qm, precision=6, suppress_small=True))
            f.write("\n\n")

            f.write("True sample variance:\n")
            f.write(np.array2string(self.qv, precision=6, suppress_small=True))
            f.write("\n\n")

            f.write("True sample std:\n")
            f.write(np.array2string(self.qs, precision=6, suppress_small=True))
            f.write("\n\n\n\n")

            f.write("2. MODE MASS RECOVERY\n")
            f.write("=====================\n\n")
            f.write("Mode partition: S_k = argmax_j log N(x; mean_j, cov_j)\n\n")
            f.write("component    true_mass    sampler_mass    abs_error\n")

            for k in range(len(mass["true_mass"])):
                f.write(
                    f"{k:<12d}"
                    f"{mass['true_mass'][k]:<13.8f}"
                    f"{mass['sampler_mass'][k]:<16.8f}"
                    f"{mass['abs_error'][k]:.8f}\n"
                )

            f.write("\n")
            f.write(f"L1 error:            {mass['l1_error']:.8f}\n")
            f.write(f"Mean absolute error: {mass['mean_abs_error']:.8f}\n")
            f.write(f"Max absolute error:  {mass['max_abs_error']:.8f}\n")
            f.write("\n\n\n\n")

            f.write("3. PARAMETRIC GAUSSIAN KL DIVERGENCE\n")
            f.write("====================================\n\n")
            f.write("KL(N_mcmc || N_true), diagonal Gaussian approximation:\n")
            f.write(f"{kl_val:.8f}\n")
            f.write("\n\n\n\n")            

            f.write("4. MAXIMUM MEAN DISCREPANCY\n")
            f.write("===========================\n\n")
            f.write("Estimator: unbiased quadratic-time MMD^2 with RBF kernel\n")
            f.write(f"Median bandwidth: {mmd['median_bandwidth']:.8f}\n\n")
            f.write("bandwidth        sigma           MMD^2_unbiased\n")

            for name, vals in mmd["values"].items():
                f.write(
                    f"{name:<16s}"
                    f"{vals['sigma']:<16.8f}"
                    f"{vals['mmd2_unbiased']:.8f}\n"
                )


            f.write("\n\n\n\n")
            f.write("5. SLICED WASSERSTEIN DISTANCE\n")
            f.write("==============================\n\n")
            f.write("Estimator: empirical sliced Wasserstein distance with p=2\n")
            f.write("Procedure: random 1D projections, sorting, average 1D W2 distance\n\n")
            f.write(f"Number of matched samples: {sw['n_samples']}\n")
            f.write(f"Number of projections:     {sw['n_projections']}\n")
            f.write(f"Sliced W2^2:               {sw['sw2_squared']:.8f}\n")
            f.write(f"Sliced W2:                 {sw['sw2']:.8f}\n")

        self.diagnostic_statistics_path = out_path
        print(f"Combined diagnostic statistics saved to {out_path}")
        return out_path


    def compute_statistics(self) -> None:
        """
        Writes one file with statistics.
        """
        self.write_diagnostic_statistics()


    def metrics(self, outdir: str | None = None, filename: str = "diagnostic_statistics.txt") -> None:
        if hasattr(self, "diagnostic_statistics_path"):
            return

        self.write_diagnostic_statistics(filename=filename)
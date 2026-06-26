"""
Diagnostics for NUMnumerical_experiments.py. This file contains only diagnostic code.
The methods expect the experiment runner to provide attributes such as self.params, 
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
    # 1.2. PLOT DIAGNOSTICS 
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

   

    def plot_diagnostics(
        self,
        filename: str = "diagnostics_core_long.pdf",
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

        The page is made wider, and the panel height is kept the same
        across both pages.
        """
        if not hasattr(self, "out") or self.out is None:
            raise ValueError("No JAX sampler output found. Run run_experiment() first.")

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        T = int(np.asarray(self.out.state.t))
        if T < 2:
            raise ValueError(f"Not enough iterations recorded (t={T}).")

        # SMC trajectory arrays
        it = np.arange(T)

        beta = np.asarray(self.out.state.beta[:T]).reshape(-1)
        ess = np.asarray(self.out.state.ess[:T]).reshape(-1)
        accept = np.asarray(self.out.state.accept[:T]).reshape(-1)
        logz = np.asarray(self.out.state.logz[:T]).reshape(-1)
        eff = np.asarray(self.out.state.efficiency[:T]).reshape(-1)

        # Experiment parameters
        n_active = int(self.params["n_active"])
        n_dims = int(self.params["n_dims"])

        # Proposal scale normalization, same convention as mutate()
        norm_ref = 2.38 / np.sqrt(n_dims)

        # Sigma is meaningful only once beta > 0
        mask_sigma = beta > 0.0
        it_sigma = it[mask_sigma]
        sigma = eff[mask_sigma] * norm_ref

        # Useful diagnostic ratio
        ess_ratio = ess / max(1, n_active)

        # ------------------------------------------------------------
    # Figure sizing:
    # make page wider, and give every subplot the same height
    # ------------------------------------------------------------
        page_width = 13.0      # wider than before
        panel_height = 4.0     # same height per subplot on every page
        extra_height = 1.0     # room for suptitle / margins

        figsize_page1 = (page_width, 3 * panel_height + extra_height)  # 3 panels
        figsize_page2 = (page_width, 2 * panel_height + extra_height)  # 2 panels

        marker_size = 3
        line_width = 1.0

        save_path = os.path.join(outdir, filename)

        with PdfPages(save_path) as pdf:
        # ============================================================
        # PAGE 1: beta, ESS, logZ
        # ============================================================
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

            # ============================================================
            # PAGE 2: acceptance, sigma
            # ============================================================
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
    # 1.3. SAMPLE STATISTICS
    #===========================================================
    def save_samples_json(self):
        # output directory 
        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        # get samples once
        true_np, mcmc_np = self.get_true_and_mcmc_samples()

        # save generated samples
        mcmc_path = os.path.join(outdir, "mcmc_samples.json")
        with open(mcmc_path, "w", encoding="utf-8") as f:
            json.dump(mcmc_np.tolist(), f)
        print(f"MCMC samples saved to {mcmc_path}")

        # save true samples
        true_path = os.path.join(outdir, "true_samples.json")
        with open(true_path, "w", encoding="utf-8") as f:
            json.dump(true_np.tolist(), f)
        print(f"True samples saved to {true_path}")


    def compute_statistics(self):
        """
        Computes and saves means and variances per dimension for
        ground truth and posterior samples
        """
        # get samples 
        true_samples, mcmc_samples = self.get_true_and_mcmc_samples()
        # MCMC stats
        self.pm = mcmc_samples.mean(axis=0)
        self.pv = mcmc_samples.var(axis=0)
        self.ps = mcmc_samples.std(axis=0)
        # True stats
        self.qm = true_samples.mean(axis=0)
        self.qv = true_samples.var(axis=0)
        self.qs = true_samples.std(axis=0)
        # store arrays 
        self.mcmc_samples = mcmc_samples
        self.true_samples_np = true_samples
        np.set_printoptions(precision=4, suppress=True)

        stats_str = ("pm (mean of MCMC samples):\n" + str(self.pm) +
            "\n\npv (variance of MCMC samples):\n" + str(self.pv) +
            "\n\nps (std dev of MCMC samples):\n" + str(self.ps) +
            "\n\nqm (mean of true samples):\n" + str(self.qm) +
            "\n\nqv (variance of true samples):\n" + str(self.qv) +
            "\n\nqs (std dev of true samples):\n" + str(self.qs) + "\n")

        outdir = self.params["outdir"]
        os.makedirs(outdir, exist_ok=True)

        stats_path = os.path.join(outdir, "sample_statistics.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(stats_str)

        print(f"Sample statistics saved to {stats_path}")


    #-----------------------------------------------------------------------------
    # 1.4. KL DIVERGENCE
    #-----------------------------------------------------------------------------
    @staticmethod
    def gau_kl(pm: np.ndarray, pv: np.ndarray,
               qm: np.ndarray, qv: np.ndarray) -> float:
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
         of Gaussians qm,qv.
        Diagonal covariances are assumed. Divergence is expressed in nats.
        """
        if (len(qm.shape) == 2):
            axis = 1
        else:
            axis = 0
        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod()
        dqv = qv.prod(axis)
        # Inverse of diagonal covariance qv
        iqv = 1. / qv
        # Difference between means pm, qm
        diff = qm - pm
        return (0.5 * (
            np.log(dqv / dpv)                 # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)            # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis)   # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)                         # - N
        ))
    

    def kl_metrics(
        self,
        outdir: str | None = None,
        filename: str = "kl_metrics.txt",
    ) -> None:

        # define outdir
        outdir = (
            outdir
            or (getattr(self, "params", {}) or {}).get("outdir", None)
            or getattr(self, "outdir", None)
        )
        if outdir is None:
            raise ValueError("No output directory specified (pass outdir=... or set params['outdir']).")
        os.makedirs(outdir, exist_ok=True)

        true_np, mcmc_np = self.get_true_and_mcmc_samples() 

        # Parametric Gaussian stats (diagonal covariance assumed)
        pm = mcmc_np.mean(axis=0)
        pv = mcmc_np.var(axis=0)
        qm = true_np.mean(axis=0)
        qv = true_np.var(axis=0)

        kl_val = self.gau_kl(pm, pv, qm, qv)  # scalar for 1D qm/qv

        out_path = os.path.join(outdir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            if np.isscalar(kl_val):
                f.write(f"Parametric KL (Gaussian): {float(kl_val):.8f}\n")
            else:
                kl_arr = np.asarray(kl_val).ravel()
                f.write("Parametric KL (Gaussian):\n")
                for i, v in enumerate(kl_arr):
                    f.write(f"  [{i}] {float(v):.8f}\n")

        print(f"KL metrics saved to {out_path}")

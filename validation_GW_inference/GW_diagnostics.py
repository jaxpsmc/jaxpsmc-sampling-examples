import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner
import matplotlib as mpl
import h5py

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams["axes.grid"] = False


class GWDiagnostics:
    TRUE_FILE = "/home/obevza/jaxpsmc/examples/validation_GW_inference/GW150914_095045_data0_1126259462-391_analysis_H1L1_result.hdf5"

    name_map = {
        "M_c":      "chirp_mass",
        "q":        "mass_ratio",
        "s1_mag":   "a_1",
        "s1_theta": "tilt_1",
        "s1_phi":   "phi_1",
        "s2_mag":   "a_2",
        "s2_theta": "tilt_2",
        "s2_phi":   "phi_2",
        "iota":     "iota",
        "d_L":      "luminosity_distance",
        "t_c":      "geocent_time",
        "phase_c":  "phase",
        "psi":      "psi",
        "ra":       "ra",
        "dec":      "dec",
    }

    gps_ref = 1126259462.4

    labels_latex = [
        r"$\mathcal{M}_c\ [M_\odot]$",
        r"$q$",
        r"$s_{1,\mathrm{mag}}$",
        r"$\theta_1$",
        r"$\phi_1$",
        r"$s_{2,\mathrm{mag}}$",
        r"$\theta_2$",
        r"$\phi_2$",
        r"$\iota$",
        r"$d_L\ \mathrm{[Mpc]}$",
        r"$t_c$",
        r"$\phi_c$",
        r"$\psi$",
        r"$\alpha$",
        r"$\delta$",
    ]

    def __init__(self, prior, true_file=None):
        self.prior = prior
        if true_file is not None:
            self.TRUE_FILE = true_file

    def next_run_dir(self, root: str, prefix: str = "run") -> str:
        os.makedirs(root, exist_ok=True)
        k = 0
        while True:
            outdir = os.path.join(root, f"{prefix}_{k:03d}")
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=False)
                return outdir
            k += 1

    def plot_diagnostics(
        self,
        out,
        n_active,
        n_dims,
        outdir,
        filename="diagnostics_core_long.pdf",
    ):
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
        os.makedirs(outdir, exist_ok=True)

        T = int(np.asarray(out.state.t))
        if T < 2:
            raise ValueError(f"Not enough iterations recorded (t={T}).")

        # SMC trajectory arrays
        it = np.arange(T)

        beta = np.asarray(out.state.beta[:T]).reshape(-1)
        ess = np.asarray(out.state.ess[:T]).reshape(-1)
        accept = np.asarray(out.state.accept[:T]).reshape(-1)
        logz = np.asarray(out.state.logz[:T]).reshape(-1)
        eff = np.asarray(out.state.efficiency[:T]).reshape(-1)

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
        page_width = 13.0
        panel_height = 4.0
        extra_height = 1.0

        figsize_page1 = (page_width, 3 * panel_height + extra_height)
        figsize_page2 = (page_width, 2 * panel_height + extra_height)

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
            ax.plot(
                it,
                beta,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
            )
            ax.set_title("beta(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("beta")
            ax.set_ylim(min(-0.02, beta.min()), max(1.02, beta.max()))
            ax.grid(True, alpha=0.3)

            # 2. ESS(t)
            ax = axes[1]
            ax.plot(
                it,
                ess,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                label="ESS",
            )
            ax.plot(
                it,
                ess_ratio * n_active,
                linestyle="--",
                linewidth=line_width,
                label="ESS/N_active × N_active",
            )
            ax.axhline(
                n_active,
                linestyle=":",
                linewidth=line_width,
                label="N_active",
            )
            ax.set_title("ESS(t)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("ESS")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            # 3. logZ(t)
            ax = axes[2]
            ax.plot(
                it,
                logz,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
            )
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
            ax.plot(
                it,
                accept,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
            )
            ax.set_title("acceptance rate")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("accept")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)

            # 5. sigma(t)
            ax = axes[1]
            ax.plot(
                it_sigma,
                sigma,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
            )
            ax.set_title("proposal scale sigma(t)  (beta > 0)")
            ax.set_xlabel("SMC iteration")
            ax.set_ylabel("sigma")
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved core diagnostics PDF to {save_path}")




    def load_true_samples(self, true_file: str, names: list[str]) -> np.ndarray:
        with h5py.File(true_file, "r") as f_true:
            post_true = f_true["posterior"]
            cols = []
            for nm in names:
                true_nm = self.name_map[nm]
                arr = post_true[true_nm][:]
                if nm == "t_c":
                    arr = arr - self.gps_ref
                cols.append(arr)
        return np.column_stack(cols)

    def save_outputs(
        self,
        *,
        event_name: str,
        out_root: str,
        out,
        theta_physical: np.ndarray,
        n_active: int,
        n_dims: int,
        meta: dict,
    ):
        # save
        outdir = self.next_run_dir(os.path.join(out_root, event_name))
        # create diagnostics PDF
        self.plot_diagnostics(out, n_active=n_active, n_dims=n_dims, outdir=outdir)


        # save posterior in hdf5 file 
        h5_path = os.path.join(outdir, "posterior.hdf5")

        with h5py.File(h5_path, "w") as f_h5:
            f_h5.create_dataset("samples", data=theta_physical)
            f_h5.create_dataset("names", data=np.asarray(list(self.prior.parameter_names), dtype="S"))
            for k, v in meta.items():
                if k == "parameter_names":
                    f_h5.attrs[k] = ",".join(v)
                else:
                    f_h5.attrs[k] = v

        print(f"[{event_name}] wrote posterior HDF5: {h5_path}")

        #1. download jims posterior HDF5 path for comparison
        TRUE_FILE = self.TRUE_FILE
        # match my parameters with true posteriors

        # load true samples and convert jaxpsmc samples
        samples_true = self.load_true_samples(TRUE_FILE, list(self.prior.parameter_names))
        samples_ours = theta_physical

        # print parameter ranges
        for i, name in enumerate(self.prior.parameter_names):
            col_true = samples_true[:, i]
            col_ours = samples_ours[:, i]
            print(f"True {name}: min={col_true.min():.4f}, max={col_true.max():.4f}, unique={np.unique(col_true).size}")
            print(f"Ours {name}: min={col_ours.min():.4f}, max={col_ours.max():.4f}, unique={np.unique(col_ours).size}")


        labels_latex = self.labels_latex

        fig = plt.figure(figsize=(22, 22))

        fig = corner.corner(samples_true, fig=fig,
            labels=labels_latex if len(labels_latex) == len(self.prior.parameter_names) else list(self.prior.parameter_names),
            show_titles=True, plot_datapoints=False, plot_density=True, fill_contours=True,
            bins=30, color="red", hist_kwargs={"density": True},)

        corner.corner(samples_ours, fig=fig, plot_datapoints=False, plot_density=True, fill_contours=False,
                      bins=30, color="blue", hist_kwargs={"density": True},)

        handles = [
            plt.Line2D([], [], color="blue", label="Sampler"),
            plt.Line2D([], [], color="red", label="Jim"),]
        
        fig.legend(handles=handles, loc="upper right")

        for ax in fig.get_axes():
            ax.grid(False)

        save_path = os.path.join(outdir, "corner_true_vs_mine.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("Saved overlay corner:", save_path)
        print(f"[{event_name}] saved {theta_physical.shape[0]} samples to: {outdir}")
        return outdir, theta_physical


class GW_diagnostics(GWDiagnostics):
    pass

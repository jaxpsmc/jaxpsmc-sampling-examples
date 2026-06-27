import jax
jax.config.update("jax_enable_x64", True)

# diagnostics
import os
import sys
import re
import argparse
import numpy as np
# jax 
import jax
import jax.numpy as jnp
# my helpers
from NUMlikelihood import *
from NUMgaussian_mixture import *
from NUMdiagnostics import Diagnostics
# jaxpsmc
from jaxpsmc import (
    Prior,
    UNIFORM,
    SamplerConfigJAX,
    SamplerJAX,
    IdentityFlowJAX,
    posterior_jax,
)


"""
Code for running Gaussian numerical experiments with jaxpsmc
"""

SUPPORTED_EXPERIMENTS = ["gaussian"]
### the argparse is used to store and process any user input we want to pass on
parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
parser.add_argument("--experiment-type", choices=["gaussian", "dualmoon", "rosenbrock"], required=True, 
                    help="Which experiment to run.")
parser.add_argument("--n-dims", type=int, required=True, 
                    help="Number of dimensions.")
parser.add_argument("--outdir", type=str, required=True, 
                    help="The output directory, where things will be stored")
# everything below here are hyperparameters for the Gaussian experiment
parser.add_argument("--nr-of-samples", type=int, default=10000, 
                    help="Number of samples to be geerated")
parser.add_argument("--nr-of-components", type=int, default=2, 
                    help="Number of components to be geerated")
parser.add_argument("--width-mean", type=float, default=10.0, 
                    help="The width of mean")
parser.add_argument("--width-cov", type=float, default=3.0, 
                    help="The width of cov")
parser.add_argument("--weights-of-components", nargs="+", type=float, default=None, 
                    help="If omitted, uses equal weights.")
# everything below here are hyperparameters for sampler
parser.add_argument("--prior-low", type=float, default=-20.0, 
                    help="Prior lower bound.")
parser.add_argument("--prior-high", type=float, default=20.0, 
                    help="Prior upper bound.")
parser.add_argument("--n-effective", type=int, required=True)
parser.add_argument("--n-active", type=int, required=True)
parser.add_argument("--n-prior", type=int, required=True)
parser.add_argument("--n-total", type=int, default=4096)
parser.add_argument("--pc-n-steps", type=int, default=8)
parser.add_argument("--pc-n-max-steps", type=int, default=80)
parser.add_argument("--keep-max", type=int, default=4096)
parser.add_argument("--sampling-mode", type=str, default="truncated_persistent", choices=["persistent", "truncated_persistent"],)
parser.add_argument("--random-state", type=int, default=0)
parser.add_argument("--precondition", action="store_true", default=True)  # keep True by default
parser.add_argument("--no-precondition", action="store_false", dest="precondition")
parser.add_argument("--dynamic", action="store_true", default=True)
parser.add_argument("--no-dynamic", action="store_false", dest="dynamic")
parser.add_argument("--metric", type=str, default="ess", choices=["ess", "uss"])
parser.add_argument("--resample", type=str, default="mult", choices=["mult", "syst"])
parser.add_argument("--transform", type=str, default="probit", choices=["probit", "logit"])
parser.add_argument("--use-identity-flow", action="store_true", default=True)
# delayed acceptance arguments
parser.add_argument("--delayed-acceptance", action="store_true", default=False,)
parser.add_argument("--da-c-const", type=float, default=0.01,)
parser.add_argument("--da-d-const", type=float, default=2.0,)
# available kernels 
parser.add_argument("--kernel", type=str, default="pcn", choices=["pcn", "li_pcn", "dili_pcn"],
                    help="Mutation kernel: pcn, li_pcn, dili_pcn.")
# empirical likelihood-informed pCN options
parser.add_argument("--li-rank", type=int, default=8, 
                    help="Rank of empirical likelihood-informed subspace for li_pcn.")
parser.add_argument("--li-lis-scale", type=float, default=1.0,
                    help="Proposal scale multiplier inside empirical LIS.")
parser.add_argument("--li-cs-scale", type=float, default=1.0,
                    help="Proposal scale multiplier in complement subspace.")
parser.add_argument("--li-var-floor", type=float, default=1e-8,
                    help="Variance floor for empirical LI-pCN covariance eigenvalues.")
parser.add_argument("--li-complement-var", type=float, default=1.0,
                    help="Reference variance used in the complement subspace.")
# Hessian/GNH-based DILI-pCN options
parser.add_argument("--dili-rank", type=int, default=4)
parser.add_argument("--dili-n-lis-particles", type=int, default=8)
parser.add_argument("--dili-lis-scale", type=float, default=1.0)
parser.add_argument("--dili-cs-scale", type=float, default=1.0)
parser.add_argument("--dili-gnh-floor", type=float, default=1e-10)
parser.add_argument("--dili-cov-floor", type=float, default=1e-8)
parser.add_argument("--dili-complement-var", type=float, default=1.0)
parser.add_argument("--dili-autodiff-gnh", action="store_true", default=False,
                    help="Use autodiff Hessian of negative log-likelihood to construct the DILI/GNH geometry.")
# additional sampling params
parser.add_argument("--proposal-scale", type=float, default=0.0)
parser.add_argument("--trim-ess", type=float, default=0.99)
parser.add_argument("--bins", type=int, default=1000)
parser.add_argument("--bisect-steps", type=int, default=1000)







##################################################################################
# 1. EXPERIMENT RUNNER
##################################################################################
class SequentialMCExperimentRunner(Diagnostics):
    """
    Base class storing everything shared between experiment
    """
    def __init__(self, args):
        # process the argparse args into params:
        self.params = vars(args)
        # automatically create a unique output directory: results_1, results_2, ...
        base_results_dir = self.params["outdir"]
        unique_outdir = self.get_next_available_outdir(base_results_dir)
        print(f"Using output directory: {unique_outdir}")
        os.makedirs(unique_outdir, exist_ok=False)
        self.params["outdir"] = unique_outdir

        # check if experiment type is allowed/supported:
        if self.params["experiment_type"] not in SUPPORTED_EXPERIMENTS:
            raise ValueError(
                f"Experiment type {self.params['experiment_type']} is not supported. "
                f"Supported types are: {SUPPORTED_EXPERIMENTS}"
            )

        # show the parameters to the screen/log file
        print("Passed parameters:")
        for key, value in self.params.items():
            print(f"{key}: {value}")

        # specify the desired target function based on the experiment type
        if self.params["experiment_type"] == "gaussian":
            print("Setting the target function to a standard Gaussian distribution.")

            # defining parameters for smc sampler 
            np.random.seed(2)
            D = self.params["n_dims"]
            
            true_samples, means, covariances, weights = GaussianMixtureGenerator.generate_gaussian_mixture(
                n_dim=D,
                n_gaussians=args.nr_of_components,
                n_samples = args.nr_of_samples,
                width_mean = args.width_mean,
                width_cov = args.width_cov,
                weights= args.weights_of_components,
            ) 

            # store true samples for diagnostics later on
            self.true_samples = true_samples

            self.mcmc_means   = jnp.stack(means, axis=0)         # (K, D)
            self.mcmc_covs    = jnp.stack(covariances, axis=0)   # (K, D, D)
            self.mcmc_weights = jnp.asarray(weights)             # (K,)

            # define Likelihood 
            self.likelihood = GaussianMixtureLikelihood(
                means=self.mcmc_means,
                covs=self.mcmc_covs,
                weights=self.mcmc_weights,
            )

            self.target_fn = self.target_normal
      

    def get_next_available_outdir(self, base_dir: str, prefix: str = "results") -> str:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        matches = [re.match(rf"{prefix}_(\d+)", name) for name in existing]
        numbers = [int(m.group(1)) for m in matches if m]
        next_number = max(numbers, default=0) + 1
        return os.path.join(base_dir, f"{prefix}_{next_number}")


    def target_normal(self, x):
        # computes log-probability (log-density) of target dist at a given point
        return self.likelihood.log_prob(x)   


    @staticmethod
    def make_auto_bounds_inflated(means, covs, inflate=9.0, nsig=12.0, pad=1e-6,
                                    prior_low=None, prior_high=None):
        """
        Function creates per dimension uniform bounds [low[d], high[d]] that contain all 
        mass of a Gaussian mixture, if needed. Created by means and cov and achieved by:
            * find smallest and largest component means in dim k:
            * for each component k, take the marginal sd in dim k 
            * set bounds
        """
        means = np.asarray(means, dtype=float)                                          # (K, D)
        covs  = np.asarray(covs,  dtype=float) * float(inflate)                         # make variance bigger
        # find smallest and largest component means in dim k
        mu_min = means.min(axis=0)                                                      # (D,)
        mu_max = means.max(axis=0)                                                      # (D,)
        # for each component k, take the marginal sd in dim k 
        std_max = np.sqrt(np.stack([np.diag(C) for C in covs], axis=0)).max(axis=0)     # (D,)
        # set bounds
        low  = mu_min - nsig * std_max - pad
        high = mu_max + nsig * std_max + pad
        # if bounds are provided, make bigger bounds, if necessary
        if prior_low  is not None: low  = np.minimum(low,  float(prior_low))
        if prior_high is not None: high = np.maximum(high, float(prior_high))
        return low, high
    

    #===========================================================
    # 1.1. RUN EXPERIMENT
    #===========================================================
    def run_experiment(self):
        """
        Function defines the experiment.
        """
        dim = int(self.params["n_dims"])
        means_np = np.asarray(self.mcmc_means)
        covs_np  = np.asarray(self.mcmc_covs)

        # initialize prior bounds
        low_np, high_np = self.make_auto_bounds_inflated(
            means_np, covs_np,
            inflate=9.0,
            nsig=12.0,
            prior_low=float(self.params["prior_low"]),
            prior_high=float(self.params["prior_high"]),
        )

        low  = jnp.asarray(low_np, dtype=jnp.float64)
        high = jnp.asarray(high_np, dtype=jnp.float64)

        # sampler compatible Uniform Prior [low[d], high[d]]
        kinds  = jnp.full((dim,), UNIFORM, dtype=jnp.int32)        # UNIFORM constant from prior_jax
        params = jnp.stack([low, high], axis=1)                    # (D,2): [low, high]
        prior  = Prior.create(kinds, params)
        self.prior = prior

        if hasattr(self, "likelihood") and self.likelihood is not None:
            loglike_single = self.likelihood.loglike_single  # use likelihood

            # first DA correctness test: surrogate = full likelihood
            loglike_approx_single = self.likelihood.loglike_single

        # Read sampler params
        n_effective  = int(self.params.get("n_effective", 512))
        n_active     = int(self.params.get("n_active", 256))
        n_prior_in   = int(self.params.get("n_prior", 512))
        n_prior      = int(np.ceil(n_prior_in / n_active) * n_active)
        n_total      = int(self.params.get("n_total", 4096))
        n_steps      = int(self.params.get("pc_n_steps", 8))
        n_max_steps  = int(self.params.get("pc_n_max_steps", 80))
        keep_max     = int(self.params.get("keep_max", 4096))
        sampling_mode = str(self.params.get("sampling_mode", "truncated_persistent"))
        precond      = bool(self.params.get("precondition", True))
        dynamic      = bool(self.params.get("dynamic", True))
        metric       = str(self.params.get("metric", "ess"))
        resample     = str(self.params.get("resample", "mult"))
        transform    = str(self.params.get("transform", "probit"))
        delayed_acceptance = bool(self.params.get("delayed_acceptance", False))
        da_c_const = float(self.params.get("da_c_const", 0.01))
        da_d_const = float(self.params.get("da_d_const", 2.0))

        cfg = SamplerConfigJAX(
            n_dim=dim,
            n_effective=n_effective,
            n_active=n_active,
            n_prior=n_prior,
            n_total=n_total,
            n_steps=n_steps,
            n_max_steps=n_max_steps,
            proposal_scale=float(self.params.get("proposal_scale", 0.0)),
    
            # kernel options
            kernel=str(self.params.get("kernel", "pcn")),
            li_rank=int(self.params.get("li_rank", 8)),
            li_lis_scale=float(self.params.get("li_lis_scale", 1.0)),
            li_cs_scale=float(self.params.get("li_cs_scale", 1.0)),
            li_var_floor=float(self.params.get("li_var_floor", 1e-8)),
            li_complement_var=float(self.params.get("li_complement_var", 1.0)),
    
            # DILI options
            dili_rank=int(self.params.get("dili_rank", 4)),
            dili_n_lis_particles=int(self.params.get("dili_n_lis_particles", 8)),
            dili_lis_scale=float(self.params.get("dili_lis_scale", 1.0)),
            dili_cs_scale=float(self.params.get("dili_cs_scale", 1.0)),
            dili_gnh_floor=float(self.params.get("dili_gnh_floor", 1e-10)),
            dili_cov_floor=float(self.params.get("dili_cov_floor", 1e-8)),
            dili_complement_var=float(self.params.get("dili_complement_var", 1.0)),
            dili_autodiff_gnh=bool(self.params.get("dili_autodiff_gnh", False)),
    
            sampling_mode=sampling_mode,
            keep_max=keep_max,
            trim_ess=float(self.params.get("trim_ess", 0.99)),
            bins=int(self.params.get("bins", 1000)),
            bisect_steps=int(self.params.get("bisect_steps", 1000)),
            blob_dim=0,
            preconditioned=precond,
            dynamic=dynamic,
            metric=metric,
            resample=resample,
            transform=transform,
            enable_flow_evidence=False,
    
            # delayed acceptance
            delayed_acceptance=delayed_acceptance,
            da_c_const=da_c_const,
            da_d_const=da_d_const,
        )       

        # use dummy flow
        flow_obj = IdentityFlowJAX(cfg.n_dim) if bool(self.params.get("use_identity_flow", True)) else self.flow
        sampler = SamplerJAX(prior, loglike_single, cfg, flow=flow_obj, loglike_approx_single_fn=loglike_approx_single,)

        # run sampler
        random_state = int(self.params.get("random_state", 0))
        key = jax.random.PRNGKey(random_state)
        out = sampler.run(key, n_total)   

        # draw posterior samplers for diagnostics
        key_post = jax.random.fold_in(key, 1)
        resample_method = jnp.int32(0 if resample == "mult" else 1)

        post = posterior_jax(
            out.state,
            key=key_post,
            do_resample=True,
            resample_method=resample_method,
            trim_importance_weights=True,
            ess_trim=jnp.asarray(cfg.trim_ess, dtype=jnp.float64),
            bins_trim=int(cfg.bins),
            beta_final=jnp.asarray(1.0, dtype=jnp.float64),
        )

        # choose how many samples you want to keep
        n_keep = int(self.params.get("nr_of_samples", min(cfg.keep_max, int(post.samples_resampled.shape[0]))))
        samples = np.asarray(post.samples_resampled[:n_keep])
        logl    = np.asarray(post.logl_resampled[:n_keep])
        logp    = np.asarray(post.logp_resampled[:n_keep])
        logZ    = float(np.asarray(out.logz))
        logZerr = float(np.asarray(out.logz_err))  # will be nan cause real flow is not trained

        # save results
        self.samples = samples
        self.logl = logl
        self.logp = logp
        self.logZ = logZ
        self.logZerr = logZerr
        self.out = out
        self.posterior = post

        self.results = {"samples": samples, "logl": logl, "logp": logp, "logZ": logZ,
                        "logZerr": logZerr, "out": out, "posterior": post, "params": self.params,}

        print("Sampling complete!")
        print("sampling_mode =", sampling_mode)
        print("n_prior (adjusted) =", n_prior, "(input was", n_prior_in, ")")
        print("samples.shape =", samples.shape)
        print("logZ =", logZ, "logZerr =", logZerr)

        return self.results
 

sys.argv = [

    # where to save
    "notebook",
    "--experiment-type", "gaussian",
    "--outdir", "/home/obevza/jaxpsmc/numerical_experiments/gaussian_10",

    # parameters of the experiments
    "--n-dims", "5",
    "--nr-of-samples", "10000",
    "--nr-of-components", "2",
    "--width-mean", "10.0",
    "--width-cov", "1.0",
    "--weights-of-components", "0.5", "0.5", 

    # define bounds
    "--prior-low", "-30.0",
    "--prior-high", "30.0",

    # define number of particles
    "--n-effective", "2000",
    "--n-active", "1000",
    "--n-prior", "20000",

    # define steps
    "--n-total", "10000",
    "--pc-n-steps", "200",
    "--pc-n-max-steps", "800",
    "--keep-max", "20000",
    "--sampling-mode", "truncated_persistent",  #  "truncated_persistent",
    "--random-state", "0",

    # define metrics
    "--metric", "ess",
    "--resample", "mult",
    "--transform", "probit",
    "--use-identity-flow",

    # proposal params (add these)
    "--proposal-scale", "0.0",
    "--trim-ess", "0.99",
    "--bins", "1000",
    "--bisect-steps", "1000",

    # delayed acceptance
    "--delayed-acceptance",
    #"--da-c-const", "0.01",
    #"--da-d-const", "2.0",

    # kernel - test li_pcn or dili_pcn
    "--kernel", "pcn",   # "pcn" "li_pcn"
    "--dili-autodiff-gnh",  # for DILI
]

def main():
    args = parser.parse_args()
    runner = SequentialMCExperimentRunner(args)
    runner.run_experiment()
    runner.plot_corner()
    runner.save_samples_json()
    runner.compute_statistics()
    runner.kl_metrics()
    runner.plot_diagnostics()

main()



#def main():
#    args = parser.parse_args()
#    runner = SequentialMCExperimentRunner(args)
#    runner.run_experiment()
#    runner.plot_true_vs_mcmc_corner()
#    runner.save_samples_json()
#    runner.compute_and_save_sample_statistics()
#    runner.kl_metrics()
#    runner.plot_top6_diagnostics_a4_2pages()


#if __name__ == "__main__":
#    main()
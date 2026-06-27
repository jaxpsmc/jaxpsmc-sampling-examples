import os
os.environ["JAX_ENABLE_X64"] = "True"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxpsmc import *
import jimgw
import time

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
)
from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.transforms import BoundToUnbound
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)

from jax.tree_util import tree_map
from GW_diagnostics import GW_diagnostics
import json
import re
import sys
import argparse
import logging
import numpy as np


### the argparse is used to store and process any user input we want to pass on
parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
parser.add_argument("--outdir", type=str, required=True, help="The output directory")
parser.add_argument("--nr-of-samples", type=int, default=10000, help="Number of samples to generate")
# everything below here are hyperparameters for sampler
parser.add_argument("--prior-low", type=float, default=-20.0, help="Prior lower bound.")
parser.add_argument("--prior-high",  type=float, default=20.0, help="Prior upper bound.")
parser.add_argument("--n-total", type=int, default=4096)
parser.add_argument("--pc-n-steps", type=int, default=8)
parser.add_argument("--pc-n-max-steps", type=int, default=80)
parser.add_argument("--keep-max", type=int, default=4096)
parser.add_argument("--sampling-mode", type=str, default="truncated_persistent",  choices=["persistent", "truncated_persistent"],)
parser.add_argument("--random-state", type=int, default=0)
parser.add_argument("--precondition", action="store_true", default=True)  # True by default
parser.add_argument("--no-precondition", action="store_false", dest="precondition")
parser.add_argument("--dynamic", action="store_true", default=True)
parser.add_argument("--no-dynamic", action="store_false", dest="dynamic")
parser.add_argument("--metric", type=str, default="ess", choices=["ess", "uss"])
parser.add_argument("--resample", type=str, default="mult", choices=["mult", "syst"])
parser.add_argument("--transform", type=str, default="probit", choices=["probit", "logit"])
# parser.add_argument("--use-identity-flow", action="store_true", default=True)
parser.add_argument("--n-effective", type=int, required=True)
parser.add_argument("--n-active", type=int, required=True)
parser.add_argument("--n-prior", type=int, required=True)
parser.add_argument("--proposal-scale", type=float, default=0.0)
parser.add_argument("--trim-ess", type=float, default=0.99)
parser.add_argument("--bins", type=int, default=1000)
parser.add_argument("--bisect-steps", type=int, default=1000)
# delayed acceptance
parser.add_argument("--delayed-acceptance", action="store_true", default=False,)
parser.add_argument("--da-c-const", type=float, default=0.01,)
parser.add_argument("--da-d-const", type=float, default=2.0, help="Conservative delayed-acceptance d constant.",)
# mutation kernel
parser.add_argument("--kernel", type=str, default="pcn", choices=["pcn", "li_pcn", "dili_pcn"],
    help="Mutation kernel: pcn, li_pcn, dili_pcn.",)
# empirical likelihood-informed pCN options
parser.add_argument("--li-rank", type=int, default=8, 
    help="Rank of empirical likelihood-informed subspace for li_pcn.",)
parser.add_argument("--li-lis-scale", type=float, default=1.0,
    help="Proposal scale multiplier inside empirical LIS.",)
parser.add_argument("--li-cs-scale", type=float, default=1.0,
    help="Proposal scale multiplier in complement subspace.",)
parser.add_argument("--li-var-floor", type=float, default=1e-8,
    help="Variance floor for empirical LI-pCN covariance eigenvalues.",)
parser.add_argument("--li-complement-var", type=float, default=1.0,
    help="Reference variance used in the complement subspace.",)



# Hessian/GNH-based DILI-pCN options
parser.add_argument("--dili-rank", type=int, default=4)
parser.add_argument("--dili-n-lis-particles", type=int, default=8)
parser.add_argument("--dili-lis-scale", type=float, default=1.0)
parser.add_argument("--dili-cs-scale", type=float, default=1.0)
parser.add_argument("--dili-gnh-floor", type=float, default=1e-10)
parser.add_argument("--dili-cov-floor", type=float, default=1e-8)
parser.add_argument("--dili-complement-var", type=float, default=1.0)
parser.add_argument("--dili-autodiff-gnh", action="store_true", default=False,
    help="Use autodiff Hessian of negative log-likelihood to construct the DILI/GNH geometry.",)



##############################################################################################
# 1. CONFIGURATIONS OF THE EXPERIMENT
##############################################################################################
# grab data
total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914 for the analysis
gps = 1126259462.4
start = gps - 2
end = gps + 2

# fetch 4096s of data to estimate the PSD (to be
# careful we should avoid the on-source segment,
# but we don't do this in this example)
psd_start = gps - 2048
psd_end = gps + 2048

# define frequency integration bounds for the likelihood
# we set fmax to 87.5% of the Nyquist frequency to avoid
# data corrupted by the GWOSC antialiasing filter
# (Note that Data.from_gwosc will pull data sampled at
# 4096 Hz by default)
fmin = 20.0
fmax = 1024

# initialize detectors
ifos = [get_H1(), get_L1()]

for ifo in ifos:
    # set analysis data
    data = Data.from_gwosc(ifo.name, start, end)
    ifo.set_data(data)

    # set PSD (Welch estimate)
    psd_data = Data.from_gwosc(ifo.name, psd_start, psd_end)
    # set an NFFT corresponding to the analysis segment duration
    psd_fftlength = data.duration * data.sampling_frequency
    ifo.set_psd(psd_data.to_psd(nperseg=psd_fftlength))


###########################################
########## Set up waveform ################
###########################################
# initialize waveform
waveform = RippleIMRPhenomPv2(f_ref=20)


###########################################
########## Set up priors ##################
###########################################
prior = []

# Mass prior
M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
Mc_prior = UniformPrior(M_c_min, M_c_max, parameter_names=["M_c"])
q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

prior = prior + [Mc_prior, q_prior]

# Spin prior
s1_prior = UniformSpherePrior(parameter_names=["s1"])
s2_prior = UniformSpherePrior(parameter_names=["s2"])
iota_prior = SinePrior(parameter_names=["iota"])

prior = prior + [
    s1_prior,
    s2_prior,
    iota_prior,
]

# Extrinsic prior
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = prior + [
    dL_prior,
    t_c_prior,
    phase_c_prior,
    psi_prior,
    ra_prior,
    dec_prior,
]

prior = CombinePrior(prior)

# Defining Transforms

sample_transforms = [
    DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
    BoundToUnbound(
        name_mapping=(["M_c"], ["M_c_unbounded"]),
        original_lower_bound=M_c_min,
        original_upper_bound=M_c_max,
    ),
    BoundToUnbound(
        name_mapping=(["q"], ["q_unbounded"]),
        original_lower_bound=q_min,
        original_upper_bound=q_max,
    ),
    BoundToUnbound(
        name_mapping=(["s1_phi"], ["s1_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_phi"], ["s2_phi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["iota"], ["iota_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_theta"], ["s1_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s2_theta"], ["s2_theta_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["s1_mag"], ["s1_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
    ),
    BoundToUnbound(
        name_mapping=(["s2_mag"], ["s2_mag_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=1.0,
    ),
    BoundToUnbound(
        name_mapping=(["phase_det"], ["phase_det_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["psi"], ["psi_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["zenith"], ["zenith_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=jnp.pi,
    ),
    BoundToUnbound(
        name_mapping=(["azimuth"], ["azimuth_unbounded"]),
        original_lower_bound=0.0,
        original_upper_bound=2 * jnp.pi,
    ),
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,
    SphereSpinToCartesianSpinTransform("s1"),
    SphereSpinToCartesianSpinTransform("s2"),
]


likelihood = BaseTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps,
    f_min=fmin,
    f_max=fmax,
)





##############################################################################################
# 2. TRANSFORM PRIOR FOR jaxpsmc IN jims UNCONSTARINT SPACE
##############################################################################################
# compute final parameter names after all sample_transforms
current_names = list(prior.parameter_names)
for t in sample_transforms:
    current_names = t.propagate_name(current_names)
final_names = current_names
D_transformed = len(final_names)
print(f"Physical parameters ({len(prior.parameter_names)}): {prior.parameter_names}")
print(f"Transformed parameters ({D_transformed}): {final_names}")


# helper functions to convert between dict and array
def dict_to_array(d, names):
    return jnp.array([d[name] for name in names])

def array_to_dict(x, names):
    return {name: x[i] for i, name in enumerate(names)}


# sample from transformed prior (physical into forward transforms)
def sample_transformed_prior(key, n):
    phys_dict = prior.sample(key, n)                     
    # apply forward transforms sequentially
    current_dict = phys_dict
    for t in sample_transforms:
        current_dict = jax.vmap(t.forward)(current_dict) 
    # convert to array in final_names order
    return jnp.stack([current_dict[name] for name in final_names], axis=1)

# log‑probability in transformed space (physical log‑prob + log|J|)
def logpdf_transformed(x):
    """
    x : array (D_transformed,) or (n, D_transformed)
    returns log probability of x in the transformed space.
    """
    def _logpdf_one(xi):
        d = array_to_dict(xi, final_names)
        # inverse transforms accumulate log Jacobian
        logjac = 0.0
        for t in reversed(sample_transforms):
            d, ld = t.inverse(d)
            logjac += ld
        phys_logprob = prior.log_prob(d)
        return phys_logprob + logjac

    if x.ndim == 1:
        return _logpdf_one(x)
    else:
        return jax.vmap(_logpdf_one)(x)


# likelihood for unconstrained samples
def gw_loglike_unconstrained(x):
    """
    x : array (D_transformed,) or (n, D_transformed)
    returns log likelihood
    """
    def _like_one(xi):
        d = array_to_dict(xi, final_names)
        # inverse transforms to get physical parameters
        for t in reversed(sample_transforms):
            d, _ = t.inverse(d)     
        # apply likelihood transforms (original list, already instances)
        for t in likelihood_transforms:
            d = t.forward(d)
        return likelihood.evaluate(d, {})

    if x.ndim == 1:
        return _like_one(x)
    else:
        return jax.vmap(_like_one)(x)


# convert unconstrained samples back to physical space (for posterior comparison)
def unconstrained_to_physical(x):
    """
    x : array (n, D_transformed) or (D_transformed,)
    returns physical samples in same order as prior.parameter_names.
    """
    def _to_phys_one(xi):
        d = array_to_dict(xi, final_names)
        for t in reversed(sample_transforms):
            d, _ = t.inverse(d)
        # return array in original physical order
        return jnp.array([d[name] for name in prior.parameter_names])

    if x.ndim == 1:
        return _to_phys_one(x)
    else:
        return jax.vmap(_to_phys_one)(x)


# create a prior object that matches the jaxpsmc expected input
class TransformedPrior:
    def __init__(self, sample_fn, logpdf_fn, dim):
        self.sample = sample_fn
        self.logpdf = logpdf_fn
        self.dim = dim
        self.params = jnp.zeros((dim, 2))   # not used

    def bounds(self):
        # all parameters are unbounded after transforms
        return jnp.array([[-jnp.inf, jnp.inf]] * self.dim)
    
    def logpdf1(self, x):
        """
        log probability for a single point (D,). Returns a scalar.
        """
        return self.logpdf(x)
    
prior_smc = TransformedPrior(
    sample_fn=lambda key, n: sample_transformed_prior(key, n),
    logpdf_fn=logpdf_transformed,
    dim=D_transformed
)





##############################################################################################
# 3. DIAGNOSTIC HELPERS
##############################################################################################
def _block_tree(x):
    def _b(a):
        return a.block_until_ready() if hasattr(a, "block_until_ready") else a
    return tree_map(_b, x)



##############################################################################################
# 4. EXPERIMENT RUNNER
############################################################################################## 
def run_event_and_save_posteriors(
    *,
    event_name: str,
    prior_u,
    loglike_x,
    loglike_approx_x=None,
    D: int,
    names: list[str],
    ranges,
    periodic_idx=None,
    args,
):
    
    # read sampler params 
    n_effective = int(args.n_effective)
    n_active    = int(args.n_active)
    n_prior_in  = int(args.n_prior)

    n_prior = int(np.ceil(n_prior_in / n_active) * n_active)

    n_total     = int(args.n_total)
    n_steps     = int(args.pc_n_steps)
    n_max_steps = int(args.pc_n_max_steps)
    keep_max    = int(args.keep_max)
    sampling_mode = str(args.sampling_mode)    

    precond   = bool(args.precondition)
    dynamic   = bool(args.dynamic)
    metric    = str(args.metric)
    resample  = str(args.resample)
    transform = str(args.transform)

    proposal_scale = float(args.proposal_scale)
    trim_ess       = float(args.trim_ess)
    bins           = int(args.bins)
    bisect_steps   = int(args.bisect_steps)

    delayed_acceptance = bool(args.delayed_acceptance)
    da_c_const = float(args.da_c_const)
    da_d_const = float(args.da_d_const)
    if loglike_approx_x is None:
        loglike_approx_x = loglike_x

    kernel = str(args.kernel)
    li_rank = int(args.li_rank)
    li_lis_scale = float(args.li_lis_scale)
    li_cs_scale = float(args.li_cs_scale)
    li_var_floor = float(args.li_var_floor)
    li_complement_var = float(args.li_complement_var)

    dili_rank = int(args.dili_rank)
    dili_n_lis_particles = int(args.dili_n_lis_particles)
    dili_lis_scale = float(args.dili_lis_scale)
    dili_cs_scale = float(args.dili_cs_scale)
    dili_gnh_floor = float(args.dili_gnh_floor)
    dili_cov_floor = float(args.dili_cov_floor)
    dili_complement_var = float(args.dili_complement_var)
    dili_autodiff_gnh = bool(args.dili_autodiff_gnh)

    if kernel == "dili_pcn":
        if not dili_autodiff_gnh:
            raise ValueError(
                "For this experiment script, use --dili-autodiff-gnh with kernel='dili_pcn', "
                "unless you modify the script to pass a custom local_gnh_fn."
            )
        if dili_n_lis_particles > keep_max:
            raise ValueError("--dili-n-lis-particles must be <= --keep-max.")


    seed     = int(args.random_state)
    n_keep   = int(args.nr_of_samples)   
    out_root = str(args.outdir)          

    # define sampler
   
    cfg = SamplerConfigJAX(
        n_dim=D,
        n_active=n_active,
        n_effective=n_effective,
        n_prior=n_prior,
        n_total=n_total,
        n_steps=n_steps,
        n_max_steps=n_max_steps,
        proposal_scale=proposal_scale,

        # kernel
        kernel=kernel,
        li_rank=li_rank,
        li_lis_scale=li_lis_scale,
        li_cs_scale=li_cs_scale,
        li_var_floor=li_var_floor,
        li_complement_var=li_complement_var,

        # Hessian/GNH-based DILI-pCN
        dili_rank=dili_rank,
        dili_n_lis_particles=dili_n_lis_particles,
        dili_lis_scale=dili_lis_scale,
        dili_cs_scale=dili_cs_scale,
        dili_gnh_floor=dili_gnh_floor,
        dili_cov_floor=dili_cov_floor,
        dili_complement_var=dili_complement_var,
        dili_autodiff_gnh=dili_autodiff_gnh,

        sampling_mode=sampling_mode,
        keep_max=keep_max,
        trim_ess=trim_ess,
        bins=bins,
        bisect_steps=bisect_steps,
        preconditioned=precond,
        dynamic=dynamic,
        metric=metric,
        resample=resample,
        transform=transform,
        periodic=periodic_idx,
        reflective=None,
        blob_dim=0,

        # delayed acceptance
        delayed_acceptance=delayed_acceptance,
        da_c_const=da_c_const,
        da_d_const=da_d_const,
    )

    t0 = time.time()

    # run sampler
    sampler = SamplerJAX(
        prior_u,
        loglike_x,
        cfg,
        flow=IdentityFlowJAX(D),
        loglike_approx_single_fn=loglike_approx_x,
    )
      
    out = sampler.run(jax.random.PRNGKey(seed))
    out = _block_tree(out)
    print(f"[{event_name}] sampler.run: {(time.time()-t0)/60:.2f} min")

    # draw posterior samples
    key_post = jax.random.PRNGKey(seed + 1)
    resample_method = jnp.int32(0 if resample == "mult" else 1)

    t1 = time.time()
    post = posterior_jax(out.state, key=key_post, do_resample=True,
                        resample_method=resample_method,
                        trim_importance_weights=True,
                        ess_trim=jnp.asarray(cfg.trim_ess, dtype=jnp.float64),
                        bins_trim=int(cfg.bins),
                        beta_final=jnp.asarray(1.0, dtype=jnp.float64),)
    
    post = _block_tree(post)
    print(f"[{event_name}] posterior_jax: {(time.time()-t1)/60:.2f} min")

    # samples are in the transformed (unconstrained) space
    theta_transformed = np.asarray(post.samples_resampled[:n_keep])
    # convert to physical space for saving and comparison
    theta_physical = np.asarray(unconstrained_to_physical(theta_transformed))  

    logZ = float(np.asarray(out.logz))
    logZerr = float(np.asarray(out.logz_err))

    print("Sampling complete!")
    print("kernel =", kernel)
    if kernel == "li_pcn":
        print("li_rank =", li_rank)
        print("li_lis_scale =", li_lis_scale)
        print("li_cs_scale =", li_cs_scale)
        print("li_var_floor =", li_var_floor)
        print("li_complement_var =", li_complement_var)

    if kernel == "dili_pcn":
        print("dili_rank =", dili_rank)
        print("dili_n_lis_particles =", dili_n_lis_particles)
        print("dili_lis_scale =", dili_lis_scale)
        print("dili_cs_scale =", dili_cs_scale)
        print("dili_gnh_floor =", dili_gnh_floor)
        print("dili_cov_floor =", dili_cov_floor)
        print("dili_complement_var =", dili_complement_var)
        print("dili_autodiff_gnh =", dili_autodiff_gnh)
        print("sampling_mode =", sampling_mode)
        print("n_prior (adjusted) =", n_prior, "(input was", n_prior_in, ")")
        print("samples.shape =", theta_physical.shape)
        print("logZ =", logZ, "logZerr =", logZerr)

    meta = {
        "event": event_name,
        "parameter_names": list(prior.parameter_names),   # physical names
        "n_samples": int(theta_physical.shape[0]),
        "logz": float(logZ),
        "logz_err": float(logZerr),
        "seed": int(seed),
        "sampling_mode": str(sampling_mode),
        "kernel": str(kernel),
        "li_rank": int(li_rank),
        "li_lis_scale": float(li_lis_scale),
        "li_cs_scale": float(li_cs_scale),
        "li_var_floor": float(li_var_floor),
        "li_complement_var": float(li_complement_var),
        "note": "Posterior draws from sampler (physical space)",

        "dili_rank": int(dili_rank),
        "dili_n_lis_particles": int(dili_n_lis_particles),
        "dili_lis_scale": float(dili_lis_scale),
        "dili_cs_scale": float(dili_cs_scale),
        "dili_gnh_floor": float(dili_gnh_floor),
        "dili_cov_floor": float(dili_cov_floor),
        "dili_complement_var": float(dili_complement_var),
        "dili_autodiff_gnh": bool(dili_autodiff_gnh),
    }

    diagnostics = GW_diagnostics(prior=prior)
    outdir, theta_physical = diagnostics.save_outputs(
        event_name=event_name,
        out_root=out_root,
        out=out,
        theta_physical=theta_physical,
        n_active=n_active,
        n_dims=D,
        meta=meta,
    )
    return outdir, theta_physical





##############################################################################################
# 5. RUN
##############################################################################################
prior_u = prior                      
bounds = np.asarray(prior_smc.bounds()) 
ranges = [tuple(map(float, b)) for b in bounds]
prior_smc = TransformedPrior(
    sample_fn=lambda key, n: sample_transformed_prior(key, n),
    logpdf_fn=logpdf_transformed,
    dim=D_transformed
)




def main(argv=None):
    args = parser.parse_args(argv)

    outdir, theta = run_event_and_save_posteriors(
        event_name="GW150914",
        prior_u=prior_smc,
        loglike_x=gw_loglike_unconstrained,
        loglike_approx_x=gw_loglike_unconstrained,  # first correctness test
        D=D_transformed,
        names=final_names,
        ranges=None,
        periodic_idx=None,
        args=args,
    )
    return outdir, theta


#if __name__ == "__main__":
#    main()



sys.argv = [
    "notebook",
    "--outdir", "/home/obevza/jaxpsmc/GW_examples_23_05",       
    "--nr-of-samples", "10000",        

    "--n-effective", "8000",
    "--n-active", "4000",
    "--n-prior", "180000",

    "--n-total", "10000",
    "--pc-n-steps", "450",
    "--pc-n-max-steps", "1000",
    "--keep-max", "30000",
    "--sampling-mode", "truncated_persistent",    # "persistent", "truncated_persistent",
    "--random-state", "0",

    "--metric", "ess",
    "--resample", "mult",
    "--transform", "probit",

    "--proposal-scale", "0.0",
    "--trim-ess", "0.99",
    "--bins", "1000",
    "--bisect-steps", "1000",

    # delayed acceptance    
    "--delayed-acceptance",

    # mutation kernel
    "--kernel", "pcn",        #"li_pcn", "pcn", "dili_pcn",
    #"--dili-autodiff-gnh",          # you have to use it with dili_pcn
]

args = parser.parse_args()


outdir, theta = run_event_and_save_posteriors(
    event_name="GW150914",
    prior_u=prior_smc,                     # use transformed prior
    loglike_x=gw_loglike_unconstrained,    # use likelihood
    D=D_transformed,                       # dimension in transformed space
    names=final_names,                     # transformed parameter names
    ranges=None,                           # will be computed inside or set to None
    periodic_idx=None,
    args=args,
)
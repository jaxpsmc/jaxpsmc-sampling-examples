# Examples

1. **Numerical experiments:** this example constructs synthetic Gaussian-mixture targets with user-controlled dimension, number of mixture components, component means, component covariances, and mixture weights. The target likelihood is known exactly and can be evaluated directly, which makes the example suitable for checking whether the sampler behaves correctly on multimodal continuous distributions with nontrivial geometry.

2. **Gravitational-wave validation:** this example uses LIGO detector data for the GW150914 event and sets up an inference problem with a frequency-domain waveform model, detector PSD estimation, physically structured priors, and deterministic transforms from physical parameters to an unconstrained sampling space.
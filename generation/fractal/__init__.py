"""Monophonic melody DETAILER (fractalizer) — a conditional diffusion model that takes a coarse
melodic line (a sparse beat-skeleton, as under-elaborating agents produce) and adds surface detail
(16ths, triplets, passing/neighbour ornaments) toward the corpus distribution, conditioned on a goal.

This is the project's pivot away from neural theme *composition* (see docs/research/17): the model's
real strength is detailing, not inventing structure, and diffusion samples the distribution's detailed
tail instead of regressing to the bland mean (the failure that killed the autoregressive melody model).
"""

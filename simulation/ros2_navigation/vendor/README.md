# CPU reciprocal solver provenance

`RVO2/` came from the user-supplied `sfm_orca_pack_20261029` snapshot, whose
provenance declares RVO2 v2.0.2, commit `c96a6ec8328492bae0721e59774026a119a60b4a`.
Its Apache-2.0 license and original source headers remain in place. The reference
already includes passive-agent responsibility changes; it is not claimed to be
an unmodified upstream checkout.

`provenance.json` records the supplied files' original hashes, installed hashes
and local changes. Local per-agent fields separate static map clearance from
pair clearance, avoiding a hidden coupling between wall width and human spacing.
Moving convex footprints use full-responsibility obstacle half-planes, translated
by the explicitly observed obstacle velocity. KD-tree edge splits preserve that
velocity. The C ABI version is checked before any native call.
ABI v3 passes pedestrian and obstacle prediction horizons independently. Rebuild
precompiled ABI v2 libraries before using the new parameter.

`bridge.cpp` is original BSD-3-Clause integration code. It exposes a C ABI for
real RVO2 solving and checks executable heading-constrained velocities against
the solver's actual half-planes. `solver.py` builds it using the existing C++
compiler, caching by source content. No system installation or Python wheel is
required. An explicit prebuilt library can be supplied to the ROS node.

The heading check applies to every nonzero candidate, including numerical
residuals, and a zero speed cap admits only zero velocity. Rebuild any explicitly
supplied prebuilt library to receive these fixes; the automatic content cache
already selects a new library. The ABI v3 argument layout is unchanged.
When the heading sector has no feasible ORCA candidate, the bridge returns an
infeasible status and a zero vector. The execution layer independently checks the
next swept step; that fallback is not a full-horizon ORCA feasibility certificate.

The unavailable LightSFM and HuNav behavior-tree sources were not copied.
Python computes explicit desired, social and obstacle-force preferences before
RVO2; this is not a claim of complete LightSFM behavioral equivalence.
Occupancy swept-disk guards cover raster geometry, not arbitrary unobserved
physical shapes or an inaccurate map.
For a changing footprint, the present certificate assumes translation at its
measured velocity during the next step. It is not a bound on unobserved angular
acceleration or arbitrary future rotation; those require a supplied swept hull
or vertex-motion bound. Missing velocity observations reject the control tick.

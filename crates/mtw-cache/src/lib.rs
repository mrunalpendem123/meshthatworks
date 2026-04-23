//! Usage-adaptive expert caching for MeshThatWorks.
//!
//! Maintains a rolling 500-prompt histogram of expert activations, tiers
//! experts into hot/warm/cold, and decides the RAM budget: pinned, mmap,
//! or streamed from SSD on demand. Feeds mesh-level re-sharding.

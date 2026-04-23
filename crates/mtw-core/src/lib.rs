//! Mesh coordination layer for MeshThatWorks.
//!
//! Owns peer discovery, shard assignment, and inference request routing
//! across the home mesh. Uses iroh for transport and identity.

pub mod echo;

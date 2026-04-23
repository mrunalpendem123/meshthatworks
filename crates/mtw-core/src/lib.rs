//! Mesh coordination layer for MeshThatWorks.
//!
//! Owns peer discovery, shard assignment, and inference request routing
//! across the home mesh. Uses iroh for transport and identity.

pub mod echo;
pub mod health;
pub mod identity;
pub mod pair;
pub mod peers;
pub mod serve;
pub mod status;

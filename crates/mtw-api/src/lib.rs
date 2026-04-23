//! Agent integration layer for MeshThatWorks.
//!
//! Serves an OpenAI-compatible HTTP API on `localhost:9337` with SSE
//! streaming, routing requests into the mesh via `mtw-core`.

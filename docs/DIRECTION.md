# Direction (decided 8 Sep 2026, after landscape research)

Full researched memo with sources: see the "Off the Critical Path" report
(published artifact). Summary:

**Rule learned from testing + literature: the network belongs off the critical
path.** Answers happen locally and instantly; the network moves work and trust
in the background. Live collective inference is structurally slow (our own
measurements; Petals' death; NVIDIA PAIR explicitly avoiding pooled inference;
SWARM-LLM publishing our architecture as research). It stays as the paper —
finish harness stages 1–3.

## Three acts

1. **The confidence router (now).** OpenAI-compatible proxy in front of
   Ollama/LM Studio: answer locally, compute calibrated probe-based semantic
   entropy, escalate ONLY genuinely uncertain queries to a cloud key — showing
   why. Verified white space: GPT-5 normalized routing (Aug 2025) and its
   backlash was about opacity; Apple's split is static task-based; semantic
   entropy has zero commercial deployments 2.5 years after Nature; no
   confidence-gated local proxy exists (Ollama fallback = crash/timeout only).
   The harness's AUROC benchmarks are the product's credibility.

2. **Your devices are the cloud (next).** Escalation targets over iroh:
   phone → own laptop → trusted friend's Mac. Mesh returns, but only for the
   uncertain minority of queries. Moat vs PAIR/EXO: they are LAN-only; iroh
   NAT-traverses. (This absorbs the "compute co-op" idea as a feature —
   standalone co-op killed: DePIN sector ≈ $180–220M total, demand thin.)

3. **The agent address book (the bet).** Same key-based reachability
   generalized to people's agents: person → agent endpoint directory, MCP/A2A
   formats over iroh, E2E encrypted, no platform in the middle. Verified gap:
   no person-to-person agent reachability exists anywhere (A2A assumes known
   HTTPS endpoints; NANDA not live; Entra intra-tenant). Enter via OpenClaw
   plugin. First killer verb: India-first agent payments (NPCI UAP / Reserve
   Pay still pre-approval — our Indus expertise). Caveat: consumer demand
   projected, not proven — Acts 1–2 fund the learning.

**Parked**: p2p shared group memory ("shared brain") — cell verified empty and
demand real (Mem0 cloud-only, Tencent centralized, Anytype no AI, Rewind
killed), but the risk is on-device retrieval quality + iroh-docs immaturity,
which wastes our networking edge. Revisit.

**Killed**: live mesh as product; compute co-op as company; anything that makes
a user wait on the slowest phone.

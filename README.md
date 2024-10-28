# Solana Aii

This repository is a brain dump of my thoughts and experiments while researching and learning about AI inference and model verification using blockchain.

For the overall vision and use cases, please see click here: [Vision](layman_readme.md)

## Basic Building Blocks

Regardless of the architectural approach taken by adapting an existing blockchain to support AI, these are the core things needed:

- support for verifiable execution proofs, training and learning, model and inference verification at the contract level
- GPU support for proof of useful work computations
- Proof of useful work consensus
- State compression
- Model registry

Types of useful work

- Logit inspection
- Model weight verification
- Training

## High level ideas

Based on initial research and thought, approaches 2 and 5 are the most viable.

### Solana Fork Options:

#### 1. Replace proof of history (PoH) with proof of logits (PoL)

This is the process of forking Solana and modifying its consensus mechanism from Proof of History (PoH) to Proof of Logits (PoL). This change transforms Solana into a blockchain platform that validates AI model inferences as part of its consensus mechanism.

Proof of history is core to Solana. Removing PoH and using different consensus is a challenging task. My gut tells me this is the least feasible approach.

#### 2. Replace proof of state (PoS) with proof of logits (PoL)

This approach maintains the proof of history, but replaces the proof of stake mechanism with proof of useful work (logit inspection). Though this would require extensive changes to validator selection and rewards, I do like this idea.
A PoH + PoUW hybrid Solana would provide a high-throughput, energy-efficient blockchain network that also performs valuable computation. In this design, the "useful work" are calculations that can be offloaded to GPUs such as logit inspection and eventually training.

See here for a more detailed design: [POH with PoUW L1](poh_pouw_design.md)

#### 3. Hybrid Dual Consensus

The hybrid approach integrates Proof of Logits validation directly into Solana's existing Proof of History mechanism, creating a single unified system that handles both traditional and AI-related transactions as well as proof of useful work (proof of logits).
An issue with this design is Solana’s PoH is integral to its high throughput, so PoUW must integrate without disrupting the existing PoH-PoS flow. This may involve limiting PoUW to certain stages, running it in parallel with PoH processing or running in the GPU

See here for a more detailed design: [Hybrid L1](poh_pouw_hybrid.md)

#### 4. Parallel Consensus

This approach differs from the hybrid solution by maintaining two distinct consensus layers that separate PoH and PoL while ensuring coordination through a bridge protocol.

Refactor Solana to combine Proof of History (PoH) with Proof of Useful Work (PoUW) in a parallel approach, the goal would be to maintain Solana’s high throughput while adding a layer of useful work computation. By running PoH and PoUW in parallel, the blockchain can achieve fast transaction ordering through PoH while validators contribute useful computations to earn rewards, thereby creating a dual-purpose consensus and utility model.

See here for a more detailed design: [Parallel L1](poh_pouw_parallel.md)

#### 5. PoUW L2

By combining Solana’s PoH-based L1 with an SVM-based PoUW L2, this architecture provides a powerful system for decentralized computation. L2 focuses on computationally intensive tasks, while L1 ensures finality and security. This setup expands Solana’s ecosystem, accommodating applications from AI training to model and inference verification.

See here for a more detailed design: [PoUW L2](L2_design.md)

### Random Thoughts

- The solana-aii src contains some examples of how logit verifications/inspections might work
- The schemas directory contains schemas on potential logit structures and how they can be used. These may be included in the model registry.
- Model Registry - This represents a system for managing models (versioning, forking, etc...). May also include metadata specifying where data is stored off-chain, for example with model weights. Maybe model weight can be processed in chunks in parallel by multiple validators.
- Tiered processing/rewards. Lower powered validators can participate with less intensive computations like inference inspection, where others might perform training.
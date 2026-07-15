# R12 Self-Authenticating State No-Go

**Status:** exact fault-model theorem; reject as a novel reasoning mechanism.

## Candidate

Carry a compact causal state together with a locally checkable certificate so
each reasoning step detects or repairs corruption before it compounds.

## Collapse theorem

Let `E:X->{0,1}^N` encode causal states and let
`V:{0,1}^N -> X union {reject}` satisfy `V(E(x))=x`.

1. If every corruption of at most `t` bits must avoid acceptance as a different
   state, then distinct codewords have Hamming distance at least `t+1`.
2. If every such corruption must be corrected to the original state, the
   distance is at least `2t+1`.
3. Against unrestricted substitution, public self-authentication is impossible:
   replacing `E(x)` with another valid `E(y)` passes completeness. Detection
   therefore needs a bounded-distance fault model or an external root, secret,
   counter, checkpoint, or trusted prior state.
4. A recurrent control with the same `N` bits and transition work can execute
   the identical map `z -> E(U_a(D(z)))`, including verification and recovery.

The first two statements are exactly error-detecting and error-correcting code
distance. The third identifies the hidden trust source. The fourth is a
resource-preserving identity simulation under the corrected R12 gate.

## Smallest witnesses

For one causal bit with update `x <- x xor a`, the code `0->00, 1->11` is the
smallest one-bit-error detector. The repetition code `0->000, 1->111` with
majority decoding is the smallest one-bit-error corrector. A recurrent control
given two or three bits and the same repair work reproduces either exactly.

Under independent boundary noise `BSC(p)`, threefold repetition fails per step
with probability `3p^2-2p^3`. Longer codes can extend the reliable horizon, but
the resource is redundancy plus a trusted repair boundary.

## Prior-art and resource boundary

- accepted packets form an error-detecting/correcting code;
- constant-query local checks are locally testable codes and import proof-oracle
  storage plus soundness error;
- noisy verification/repair is fault-tolerant computation;
- recursive execution certificates are proof-carrying data or incrementally
  verifiable computation and certify a specified update, not its semantic truth;
- detection without correction is checkpoint/restart or fail-stop recovery;
- cryptographic authentication imports a key/root and replay protection imports
  a trusted counter or history commitment.

The only exposed resource is fault-domain-separated trust. Reopen only for a
separation against coded/proof-carrying controls with identical bits, precision,
trusted boundaries, checkpoints, FLOPs, and correlated-noise exposure. No CPU
falsifier, Shohin fit, or H100 job is authorized.

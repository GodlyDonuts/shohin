# Shohin Research Outcomes: Public Summary

This document records high-level results that can be discussed publicly
without releasing the active research architecture, implementation, task
generator, or training protocol.

## Scope

The research extension asks a narrow question: can a small model convert a
bounded language source into an executable program, carry a structured state
through several operations, and answer only after that state has been
computed? The source is removed before recurrent execution, so the later
answer must be produced from the retained structured representation.

The experiments use synthetic finite domains so every program, intermediate
state, and answer can be checked exactly. They are component-level scientific
tests, not substitutes for open-domain benchmarks.

## Language-to-program and state reuse

The protocol was frozen before a single disjoint 2,048-example confirmation
board was opened. No second board, post-confirmation repair, or rescore was
used.

| Confirmation measurement | Exact result |
|---|---:|
| Programs | 2,003 / 2,048 = **97.803%** |
| Recurrent states | 2,015 / 2,048 = **98.389%** |
| Final answers | 2,020 / 2,048 = **98.633%** |

Two matched causal controls independently rotate essential internal
representations while preserving the surrounding evaluation process. Each
control produces **0/2,048 exact programs**. This distinguishes use of the
compiled representation from a direct-answer or marginal shortcut on the
frozen board.

## Primitive-to-composed transition generalization

A separate learned transition module contains **4,934 parameters**. It receives
only six primitive one-step training cells and no two-step transition,
recurrent-program, confirmation-program, or answer supervision.

On the sole frozen confirmation:

| Measurement | Result |
|---|---:|
| Never-trained two-step transitions | **36 / 36 exact** |
| End-to-end exact programs | **96.924%** |
| End-to-end exact state | **97.607%** |
| End-to-end correct answers | **98.096%** |

The learned module exactly matches the fixed exact-executor upper bound on
program, state, and answer accuracy. Matched law, direction, and state-reset
interventions substantially degrade recurrent state, providing evidence that
the learned transitions are causally consumed.

## What can be claimed

The results establish:

- high-accuracy compilation on one frozen, disjoint bounded-language board;
- recurrent reuse of a model-produced structured state;
- exact composition of all 36 held-out two-step transitions from six primitive
  examples; and
- causal dependence on the compiled representation and learned transition
  content.

They do not establish:

- unrestricted natural-language reasoning;
- open-ended planning or learned halting;
- improved GSM8K, MATH-500, HumanEval, or MBPP performance;
- a general-purpose replacement for transformer inference; or
- publication priority or broad architectural novelty by themselves.

The public repository therefore exposes the complete 300K pretrained baseline
while limiting the active research extension to this result-level summary.

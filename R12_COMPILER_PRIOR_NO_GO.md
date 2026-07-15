# R12 Compiler-Prior No-Go

**Status:** exact collapse theorem. Recurrence can provide a useful compiler
prior or optimization bias, but it has no intrinsic sample, parameter-bit, or
compute generalization advantage over a fair uniform acyclic realization.

## 1. Theorem

Let `theta` contain `p` learned bits and let a recurrent evaluator process a
length-`L` input by

```
s_i = U_theta(s_(i-1), x_i),
y = O_theta(s_L),
```

where the state uses `b` bits at fixed precision, one transition costs work
`c` and depth `d`, and the learner infers `theta` from dataset `D` and prior
`pi`.

There is a uniform acyclic compiler that, for every `L`, emits

```
O_theta o U_theta[x_L] o ... o U_theta[x_1].
```

For every dataset, learner random seed, and input, the compiled evaluator has
the identical output. The constructor receives no scale-specific target advice
and shares the same learned parameter source nodes across all copied cells.

## 2. Resource ledger

| Resource | Recurrent evaluator | Uniform compiled evaluator |
|---|---:|---:|
| Learned parameter bits | `p` | `p` |
| Precision | same | same |
| Samples and prior | same | same |
| Oracle calls | same | same |
| Work | `Lc + c_O` | `Lc + c_O` |
| Sequential depth | `Ld + d_O` | `Ld + d_O` |
| Peak scheduled state | `b + scratch` | `b + scratch` |
| Instantiated graph area | reused cell | `Theta(Lc)` |

Materializing every activation in parallel can use `Theta(Lb)` memory;
sequentially scheduling the acyclic graph recovers recurrent peak memory. The
real difference is reusable program description or instantiated graph area,
not learned information or function class.

## 3. Consequences

- A single acyclic forward invocation is not a meaningful comparator if it is
  forbidden to contain the compiled transition chain.
- A streaming one-pass comparator already includes recurrence.
- Fixed parallel depth can yield classical circuit-depth separations, but that
  is a depth claim and may reach open `TC0` versus `NC1` questions for strong
  transformer-like comparators.
- Any claimed sample or extrapolation advantage must come from the training
  protocol, structural prior, optimization dynamics, or a separately counted
  resource, not from the presence of a loop alone.

## 4. Small finite collapse witness

One state bit, one learned bit, two events, and horizon two already demonstrate
the identity. Let `theta=0` assign `a=NOT, b=RESET0` and `theta=1` assign
`a=RESET0, b=NOT`, starting from zero. One observed transition identifies
`theta`; `ab` returns `theta` and `ba` returns `1-theta`. A recurrent evaluator
uses two calls to one cell. The compiler uses two copied cells sharing the same
learned bit. There is no finite horizon at which this equivalence fails.

## 5. Decision

Reject recurrence itself as the outstanding nonlinear-learnability
separation. Preserve tied recurrence as a favorable matched control and count
its compact reusable program description honestly. No CPU experiment can
falsify this identity theorem; experiments may only test whether a frozen
training protocol makes the shared algorithm easier to learn at matched
resources.

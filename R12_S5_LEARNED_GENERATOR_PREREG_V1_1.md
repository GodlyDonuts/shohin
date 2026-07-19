# R12 S5.1 Learned Generator-Factored Executor Preregistration

**Status:** frozen after the S5 v1 pre-board custody failure and before any new
seed, board, fit, model load, or score.

S5 v1.1 imports the architecture, matched controls, fit contract, all 18 gates,
and claim boundary from `R12_S5_LEARNED_GENERATOR_PREREG.md` unchanged. Its sole
correction is the board exclusion contract.

## Retired Attempt

Seed `107732609041319044` produced no board, fit, model load, or score. The
builder was incorrectly passed the sealed S4 v5 confirmation as an exclusion
input. It opened enough of the file to detect the confirmation split, failed
closed, and created no output directory. S5 v1 is retired and cannot authorize
a result.

## Corrected Custody

After this file and the S5 v1 closure receipt are committed, draw one new seed.
Build one 512-group development board excluded against:

- the admitted S4 source training corpus;
- the factorized source training corpus;
- the public self-delimiting S4 development board; and
- every S4 v2--v5 **development** board.

Do not pass, open, hash, or otherwise inspect any sealed confirmation board.
The old confirmation's aggregate public result may remain in documentation,
but neither its row bytes nor any derived overlap statistic may enter S5.1.
Random nonces, factor exclusion, and the existing exact-prompt/13-gram/name/
roster-multiset gates provide the lawful development isolation.

Exactly one matched six-cell fit and one serial H100 evaluation may access the
new development board. All original gates remain exact. A pass authorizes only
one newly seeded confirmation protocol that similarly excludes prior public and
development data without opening any old confirmation bytes.

## Closure

Seed `7741142465189679834` built 2,048 rows / 512 groups without confirmation
access or model access. Exact-prompt, 13-gram, factor, nonce, context, depth,
balance, and independent-executor gates pass, but 4/2,048 rows reuse a public
roster token multiset. `all_gates_pass` is false, so this board is sealed and
cannot be scored. S5.1 closes with no fit, model load, or result. S5.2 may draw
one replacement seed after commit with architecture, fit, controls, evaluator,
assessor, gates, and exclusions unchanged.

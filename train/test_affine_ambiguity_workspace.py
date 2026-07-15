#!/usr/bin/env python3
"""Exact soundness gates for the R10 affine ambiguity workspace."""

import itertools
import random

from affine_ambiguity_workspace import (
    AffineAmbiguityWorkspace,
    build_workspace,
    compose_workspaces,
    workspace_from_operations,
    workspace_from_transforms,
)
from version_space_product_tree import (
    ExactAffineTransform,
    build_tree,
    opcode_candidate,
    operation_transform,
    query_agreement,
)


def exact_products(candidate_sets):
    products = []
    for path in itertools.product(*candidate_sets):
        total = operation_transform(*path[0])
        for opcode, value in path[1:]:
            total = total.followed_by(operation_transform(opcode, value))
        products.append(total)
    return tuple(set(products))


def test_leaf_hull_and_query_annihilator():
    workspace = workspace_from_operations(
        (("add_0", 5), ("add_1", 5)), source_index=7,
    )
    assert workspace.ambiguity_rank == 1
    assert workspace.retained_source_indices == (7,)
    assert workspace.retrieval_source_indices == (7,)
    assert workspace.contains(operation_transform("add_0", 5))
    assert workspace.contains(operation_transform("add_1", 5))

    total = workspace.query_certificate((2, 7), "sum")
    assert total.certified and total.integer_answer == 14
    assert not total.hot_context_evictable
    query_set = workspace.query_set_certificate((2, 7), ("sum", "read_0"))
    assert not query_set.certified

    first = workspace.query_certificate((2, 7), "read_0")
    assert not first.certified and first.answer is None


def test_rank_zero_is_the_only_general_source_drop():
    workspace = workspace_from_operations(
        (("swap", 0), ("swap", 0)), source_index=3,
    )
    assert workspace.ambiguity_rank == 0
    assert workspace.hot_context_evictable
    assert workspace.retained_source_indices == ()
    assert workspace.retrieval_source_indices == (3,)
    assert workspace.query_certificate((4, 9), "read_0").integer_answer == 9


def test_product_hull_contains_every_exact_product():
    earlier_candidates = (("add_0", 2), ("move_0_1", 2), ("swap", 0))
    later_candidates = (("sub_1", 1), ("merge_0_1", 0), ("swap", 0))
    earlier = workspace_from_operations(earlier_candidates, source_index=0)
    later = workspace_from_operations(later_candidates, source_index=1)
    product = compose_workspaces(earlier, later)
    assert product.ambiguity_rank <= 9
    assert product.retained_source_indices == (0, 1)
    for transform in exact_products((earlier_candidates, later_candidates)):
        assert product.contains(transform)


def test_balanced_workspace_is_sound_under_either_grouping():
    candidate_sets = (
        (("add_0", 1), ("swap", 0)),
        (("sub_0", 1), ("merge_0_1", 0)),
        (("add_1", 2), ("swap", 0)),
    )
    balanced = build_workspace(candidate_sets)
    first = workspace_from_operations(candidate_sets[0], source_index=0)
    second = workspace_from_operations(candidate_sets[1], source_index=1)
    third = workspace_from_operations(candidate_sets[2], source_index=2)
    left = compose_workspaces(compose_workspaces(first, second), third)
    right = compose_workspaces(first, compose_workspaces(second, third))
    exact = exact_products(candidate_sets)
    for transform in exact:
        assert balanced.contains(transform)
        assert left.contains(transform)
        assert right.contains(transform)


def test_affine_certificates_never_disagree_with_exact_version_space():
    candidate_sets = (
        (("add_0", 5), ("add_1", 5)),
        (("swap", 0),),
    )
    exact_tree = build_tree(tuple(
        tuple(opcode_candidate(opcode, value) for opcode, value in candidates)
        for candidates in candidate_sets
    ), cap=32)
    workspace = build_workspace(candidate_sets)
    for initial in ((0, 0), (2, 7), (11, 3)):
        for query in ("read_0", "read_1", "sum", "difference_0_1", "difference_1_0"):
            exact = query_agreement(exact_tree, initial, query)
            affine = workspace.query_certificate(initial, query)
            if affine.certified:
                assert exact.query_agrees
                assert affine.integer_answer == exact.answer


def test_current_query_agreement_does_not_survive_every_continuation():
    workspace = workspace_from_transforms(
        (ExactAffineTransform.identity(), operation_transform("merge_0_1", 0)),
        source_index=4,
    )
    assert workspace.query_certificate((1, 0), "read_0").certified
    continued = compose_workspaces(
        workspace,
        workspace_from_operations((("swap", 0),), source_index=5),
    )
    assert not continued.query_certificate((1, 0), "read_0").certified
    assert workspace.retained_source_indices == (4,)


def test_wrong_singleton_remains_retrievable_and_candidate_conditional():
    wrong = workspace_from_operations((("add_0", 1),), source_index=9)
    truth = operation_transform("sub_0", 1)
    assert wrong.hot_context_evictable
    assert wrong.retained_source_indices == ()
    assert wrong.retrieval_source_indices == (9,)
    assert not wrong.contains(truth)


def test_homogeneous_rank_contract_rejects_malformed_directions():
    anchor = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    malformed = ((0, 0, 0), (0, 0, 0), (0, 0, 1))
    try:
        AffineAmbiguityWorkspace(anchor, (malformed,), (0,), (0,))
    except ValueError:
        pass
    else:
        raise AssertionError("nonzero homogeneous ambiguity row was accepted")


def test_bilinear_cross_term_blocks_false_single_leaf_attribution():
    identity = ExactAffineTransform.identity()
    earlier = workspace_from_transforms(
        (identity, operation_transform("merge_0_1", 0)), source_index=0,
    )
    later = workspace_from_transforms(
        (identity, operation_transform("merge_1_0", 0)), source_index=1,
    )
    product = compose_workspaces(earlier, later)
    assert earlier.query_certificate((1, 0), "read_0").certified
    assert later.query_certificate((1, 0), "read_0").certified
    assert not product.query_certificate((1, 0), "read_0").certified


def test_deterministic_product_containment_sweep():
    pool = (
        ("add_0", 1), ("add_1", 2), ("sub_0", 1), ("sub_1", 2),
        ("move_0_1", 1), ("move_1_0", 2),
        ("merge_0_1", 0), ("merge_1_0", 0), ("swap", 0),
    )
    rng = random.Random(20260714)
    for _ in range(128):
        depth = rng.randint(1, 4)
        candidate_sets = tuple(
            tuple(rng.sample(pool, rng.randint(1, 3))) for _ in range(depth)
        )
        workspace = build_workspace(candidate_sets)
        exact = exact_products(candidate_sets)
        assert workspace.ambiguity_rank <= 6
        for transform in exact:
            assert workspace.contains(transform)
        initial = (rng.randint(0, 9), rng.randint(0, 9))
        for query in ("read_0", "read_1", "sum", "difference_0_1", "difference_1_0"):
            certificate = workspace.query_certificate(initial, query)
            if certificate.certified:
                answers = {transform.answer(initial, query) for transform in exact}
                assert answers == {certificate.integer_answer}


def main():
    test_leaf_hull_and_query_annihilator()
    test_rank_zero_is_the_only_general_source_drop()
    test_product_hull_contains_every_exact_product()
    test_balanced_workspace_is_sound_under_either_grouping()
    test_affine_certificates_never_disagree_with_exact_version_space()
    test_current_query_agreement_does_not_survive_every_continuation()
    test_wrong_singleton_remains_retrievable_and_candidate_conditional()
    test_homogeneous_rank_contract_rejects_malformed_directions()
    test_bilinear_cross_term_blocks_false_single_leaf_attribution()
    test_deterministic_product_containment_sweep()
    print("affine ambiguity workspace mechanics: passed (exact rational, sound, fail-closed)")


if __name__ == "__main__":
    main()

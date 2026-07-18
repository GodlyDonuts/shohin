from __future__ import annotations

import copy

import cross_domain_fault_channel_falsifier as cross


def test_fault_neighborhoods_require_redundancy_or_provenance():
    result = cross.disjoint_fault_neighborhood_lemma_certificate()
    assert result["pass"]
    assert result["uncoded_neighborhood_intersection"] == [0, 1]
    assert not result["uncoded_exact_recovery_identifiable"]
    assert result["repetition_radius_one_intersection"] == []
    assert result["repetition_exact_recovery_identifiable"]


def test_triadic_commit_is_repetition_code_and_fails_common_mode_semantics():
    result = cross.triadic_efference_certificate()
    assert result["pass"]
    assert result["one_fault_checks"] == 12
    assert result["one_fault_failures"] == []
    assert len(result["two_lane_consistent_origins"]) == 2
    assert not result["two_lane_fault_localization_identifiable"]
    assert result["three_lane_decoder_equals_repetition_majority"]
    assert result["shared_semantic_failures"] == 4
    assert not result["surviving_advantage_over_repetition_code"]


def test_reversible_cat_map_never_contracts_a_nonzero_error():
    result = cross.reversible_transport_certificate()
    assert result["pass"]
    assert result["determinant_mod5"] == 1
    assert result["states"] == 25
    assert result["nonzero_errors"] == 24
    assert result["checks"] == 6_000
    assert result["collapsed_errors"] == []
    assert not result["reversible_transport_contracts_all_errors"]
    assert result["correction_requires_noninvertible_decoder_or_extra_provenance"]


def test_s3_atlas_is_recurrence_when_complete_and_patchable_when_incomplete():
    result = cross.relation_syndrome_atlas_certificate()
    assert result["pass"]
    assert result["states"] == 6
    assert result["state_generator_pairs"] == 12
    assert all(result["relations"].values())
    assert result["complete_atlas_matches_six_state_recurrence"]
    assert result["admitted_pairs"] == 11
    assert result["patched_atlas_exact_on_all_admitted_pairs"]
    assert result["exact_endpoint"] != result["patched_endpoint"]
    assert result["separating_late_queries"]
    assert not result["finite_incomplete_atlas_identifies_uniform_update"]


def test_report_is_deterministic_and_grants_no_neural_authority():
    first = cross.build_report()
    second = cross.build_report()
    assert first == second
    assert first["all_pass"]
    assert first["candidate_mechanisms_tested"] == 3
    assert first["candidate_mechanisms_surviving"] == 0
    assert not first["neural_preregistration_authorized"]
    without_hash = copy.deepcopy(first)
    digest = without_hash.pop("payload_sha256")
    assert digest == cross.payload_sha256(without_hash)

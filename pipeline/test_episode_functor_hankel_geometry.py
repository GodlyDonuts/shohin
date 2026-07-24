from pipeline.episode_functor_hankel_geometry import (
    commutative_bag_incidence,
    enumerate_action_words,
    prefix_shift_incidence,
    random_shift_incidence,
)
from pipeline.episode_functor_hankel_shift import (
    commutative_bag_incidence as oracle_commutative_bag_incidence,
)
from pipeline.episode_functor_hankel_shift import (
    enumerate_action_words as oracle_enumerate_action_words,
)
from pipeline.episode_functor_hankel_shift import (
    prefix_shift_incidence as oracle_prefix_shift_incidence,
)
from pipeline.episode_functor_hankel_shift import (
    random_shift_incidence as oracle_random_shift_incidence,
)


def test_runtime_geometry_matches_offline_oracle() -> None:
    for depth in range(5):
        assert enumerate_action_words(depth) == oracle_enumerate_action_words(
            depth
        )
        assert prefix_shift_incidence(depth) == oracle_prefix_shift_incidence(
            depth
        )
        assert commutative_bag_incidence(
            depth
        ) == oracle_commutative_bag_incidence(depth)
        assert random_shift_incidence(
            depth,
            seed="runtime-geometry-test",
        ) == oracle_random_shift_incidence(
            depth,
            seed="runtime-geometry-test",
        )

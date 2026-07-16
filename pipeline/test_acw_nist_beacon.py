import copy
import json
import unittest
from pathlib import Path

from pipeline.acw_nist_beacon import derive_confirmation_seed, verify_pulse


class NistBeaconVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = Path(__file__).parent / "testdata" / "acw_nist_beacon_snapshot.json"
        cls.snapshot = json.loads(fixture.read_text())

    def verify(self, pulse=None, previous=None):
        return verify_pulse(
            pulse or self.snapshot["pulse"],
            self.snapshot["certificate_pem"].encode("ascii"),
            previous_pulse=previous or self.snapshot["previous_pulse"],
            expected_chain_index=2,
            expected_pulse_index=1_862_119,
            expected_timestamp="2026-07-16T13:50:00.000Z",
        )

    def test_archived_signed_pulse_replays(self) -> None:
        receipt = self.verify()
        self.assertTrue(receipt["signature_verified"])
        self.assertTrue(receipt["output_hash_verified"])
        self.assertIsNotNone(receipt["previous_link"])

    def test_pulse_and_chain_mutations_fail(self) -> None:
        cases = []
        output = copy.deepcopy(self.snapshot["pulse"])
        output["outputValue"] = "00" * 64
        cases.append((output, self.snapshot["previous_pulse"]))
        signature = copy.deepcopy(self.snapshot["pulse"])
        signature["signatureValue"] = "00" * 512
        cases.append((signature, self.snapshot["previous_pulse"]))
        previous = copy.deepcopy(self.snapshot["previous_pulse"])
        previous["precommitmentValue"] = "00" * 64
        cases.append((self.snapshot["pulse"], previous))
        for pulse, prior in cases:
            with self.subTest(
                field=(pulse["outputValue"][:8], prior["precommitmentValue"][:8])
            ):
                with self.assertRaises(ValueError):
                    self.verify(pulse, prior)

    def test_three_seed_domains_are_deterministic_and_distinct(self) -> None:
        seeds = [
            derive_confirmation_seed(
                authorization_payload_sha256="ab" * 32,
                pulse=self.snapshot["pulse"],
                index=index,
            )
            for index in range(3)
        ]
        self.assertEqual(len(set(seeds)), 3)
        self.assertTrue(all(len(seed) == 32 for seed in seeds))
        self.assertEqual(
            seeds[0],
            derive_confirmation_seed(
                authorization_payload_sha256="ab" * 32,
                pulse=self.snapshot["pulse"],
                index=0,
            ),
        )


if __name__ == "__main__":
    unittest.main()

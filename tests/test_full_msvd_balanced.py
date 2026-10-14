import json
import unittest
from collections import Counter

from scripts.run_full_msvd_balanced import (
    FORMAL_EPOCHS,
    FORMAL_STEPS,
    TRAIN,
    coverage_is_valid,
    make_batch_schedule,
    make_reference_orders,
)


class FullMSVDBalancedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise scheduling independently of third-party dataset availability.
        cls.records = [{"video_id": f"synthetic-{i}", "captions": [f"reference-{j}" for j in range(18 + i % 49)]}
                       for i in range(1200)]

    @unittest.skipUnless(TRAIN.is_file(), "requires local MSVD annotations")
    def test_full_dataset_contract_counts(self):
        records = json.loads(TRAIN.read_text(encoding="utf-8"))
        counts = [len(record["captions"]) for record in records]
        self.assertEqual(len(records), 1200)
        self.assertEqual(sum(counts), 48774)
        self.assertEqual(min(counts), 18)
        self.assertEqual(max(counts), 66)
        self.assertEqual(FORMAL_EPOCHS, 66)
        self.assertEqual(FORMAL_STEPS, 9900)

    def test_batch_schedule_is_deterministic_complete_and_9900_steps(self):
        schedule = make_batch_schedule(1200, FORMAL_EPOCHS)
        self.assertEqual(schedule, make_batch_schedule(1200, FORMAL_EPOCHS))
        self.assertEqual(sum(len(epoch) for epoch in schedule), FORMAL_STEPS)
        for epoch in schedule:
            self.assertEqual(len(epoch), 150)
            self.assertEqual(sorted(index for batch in epoch for index in batch), list(range(1200)))

    def test_reference_orders_are_complete_and_66_epochs_cover_all(self):
        orders = make_reference_orders(self.records)
        self.assertEqual(orders, make_reference_orders(self.records))
        for record in self.records:
            video_id = str(record["video_id"])
            count = len(record["captions"])
            self.assertEqual(sorted(orders[video_id]), list(range(count)))
            visited = {orders[video_id][epoch % count] for epoch in range(FORMAL_EPOCHS)}
            self.assertEqual(visited, set(range(count)))

    def test_schedule_rejects_partial_batches(self):
        with self.assertRaisesRegex(ValueError, "fixed-size"):
            make_batch_schedule(10, 1)

    def test_smoke_and_formal_coverage_requirements_differ(self):
        one = Counter({0: 1})
        complete = Counter({index: 1 for index in range(3)})
        self.assertTrue(coverage_is_valid(one, 3, smoke=True))
        self.assertFalse(coverage_is_valid(one, 3, smoke=False))
        self.assertTrue(coverage_is_valid(complete, 3, smoke=False))


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from scripts.run_repeated_seeds import make_schedule, TRAIN

class RepeatedSeedsTests(unittest.TestCase):
    def test_schedule_pair_reproducible_and_seeds_differ(self):
        records=[{'video_id':f'synthetic-{i}', 'captions':[f'reference-{j}' for j in range(18+i%49)]}
                 for i in range(1200)]
        a=make_schedule(records,124)
        b=make_schedule(records,125)
        self.assertEqual(a,make_schedule(records,124))
        self.assertNotEqual(a['batches'],b['batches'])
        self.assertNotEqual(a['orders'],b['orders'])
        for schedule in (a,b):
            self.assertEqual(sum(map(len,schedule['batches'])),9900)
            for epoch in schedule['batches']:
                self.assertEqual(sorted(i for batch in epoch for i in batch),list(range(1200)))
            for record in records:
                order=schedule['orders'][record['video_id']]
                self.assertEqual(sorted(order),list(range(len(record['captions']))))
                self.assertEqual({order[e%len(order)] for e in range(66)},set(range(len(order))))

    def test_unregistered_seed_rejected(self):
        with self.assertRaises(ValueError):
            make_schedule([],126)

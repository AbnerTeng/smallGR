import unittest

import torch

from src.trie import RouteConstraint


class RetrievalConstraintsTest(unittest.TestCase):
    def setUp(self):
        self.routes = [[4, 5, 6, 7], [4, 5, 6, 8], [9, 10, 11, 12]]
        self.constraint = RouteConstraint(self.routes, eos=1, vocab_size=16, device="cpu")

    def allowed(self, prefixes):
        scores = torch.zeros(len(prefixes), 16)
        result = self.constraint(torch.tensor(prefixes), scores)
        return [set(torch.where(torch.isfinite(row))[0].tolist()) for row in result]

    def test_collision_suffix_distinguishes_documents(self):
        self.assertEqual(self.allowed([[0, 4, 5, 6]]), [{7, 8}])

    def test_routes_share_only_valid_prefixes(self):
        self.assertEqual(self.allowed([[0]]), [{4, 9}])
        self.assertEqual(self.allowed([[0, 4], [0, 9]]), [{5}, {10}])

    def test_complete_route_requires_eos(self):
        self.assertEqual(self.allowed([[0, 4, 5, 6, 8]]), [{1}])

    def test_invalid_prefix_cannot_reenter_corpus(self):
        self.assertEqual(self.allowed([[0, 4, 10], [0, 9, 10]]), [{1}, {11}])


if __name__ == "__main__":
    unittest.main()

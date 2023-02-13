import unittest
from oslo import *


class TestOslo(unittest.TestCase):
    def test1(self):
        model = OsloModel(16)
        for _ in range(400):  # run the model for a long time to reach the steady state
            model.run()

        result = []
        for i in range(3_000_000):  # take average over t=5000
            model.run()
            result.append(model.heights[0])
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.mean(result), 26.5, places=1)

    def test2(self):
        model = OsloModel(32)
        for _ in range(1500):  # run the model for a long time to reach the steady state
            model.run()

        result = []
        for i in range(3_000_000):  # take average over t=5000
            model.run()
            result.append(model.heights[0])
        # noinspection PyTypeChecker
        self.assertAlmostEqual(np.mean(result), 53.9, places=1)

    def test3(self):




if __name__ == '__main__':
    unittest.main()

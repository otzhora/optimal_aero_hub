import unittest

from hub.geodesic import Point, distance


class TestGeodesic(unittest.TestCase):
    def test_distance(self):
        moscow = Point(55.751244, 37.618423)
        spb = Point(59.9342802, 30.3350986)

        self.assertAlmostEqual(633.4538325946983, distance(moscow, spb))

        newport = Point(41.49008, -71.312796)
        cleveland = Point(41.499498, -81.695391)

        self.assertAlmostEqual(864.2144943393627, distance(newport, cleveland))


if __name__ == "__main__":
    unittest.main()

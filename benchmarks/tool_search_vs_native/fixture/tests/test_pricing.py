import unittest

from shop.pricing import apply_discount, order_total


class PricingTest(unittest.TestCase):
    def test_ten_percent(self):
        self.assertEqual(apply_discount(100.0, 10), 90.0)

    def test_zero(self):
        self.assertEqual(apply_discount(59.99, 0), 59.99)

    def test_order_total_with_coupon(self):
        self.assertEqual(order_total([(20.0, 2), (10.0, 1)], coupon_percent=10), 45.0)


if __name__ == "__main__":
    unittest.main()

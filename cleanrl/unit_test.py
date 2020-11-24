import unittest
import torch
import cleanrl.c51 as c51
import cleanrl.rnd_ppo as rnd_ppo
import cleanrl.dqn as dqn


class C51Tests(unittest.TestCase):
    def test_no_params_qNetwork_forward(self):
        with self.assertRaises(TypeError):
            c51.QNetwork().forward()

    def test_no_params_qNetwork_get(self):
        with self.assertRaises(TypeError):
            c51.QNetwork().get_action()

#    def test_qNetwork_forward_params(self):
#        device = torch.device('cpu')
#        self.assert


class RndPpoTests(unittest.TestCase):
    def test_revard_forward_filter_update_with_rews_with_gamma(self):
#       gamma = 1
        filter = rnd_ppo.RewardForwardFilter(1)
        self.assertEqual(25, filter.update(25))
        self.assertEqual(3249, filter.update(3224))
        self.assertEqual(39205923059230523636859242651, filter.update(39205923059230523636859239402))
        self.assertEqual(39205923137642369754773359586577724704, filter.update(39205923098436446695542835949718482053))

    def test_revard_forward_filter_update_with_rews_without_gamma(self):
        filter = rnd_ppo.RewardForwardFilter(None)
        filter.update(25)
        self.assertEqual(50, filter.update(25))

class DqnTests(unittest.TestCase):

    def test_linear_schedule(self):
        self.assertEqual(0.9259,dqn.linear_schedule( 1, 0.05, 16000, 1248))
        self.assertEqual(0.36694375, dqn.linear_schedule( 1, 0.05, 16000, 10662))
        self.assertEqual(0.4188375, dqn.linear_schedule( 1, 0.05, 16000, 9788))
        self.assertEqual(0.52654375, dqn.linear_schedule( 1, 0.05, 16000, 7974))
        self.assertEqual(0.53366875, dqn.linear_schedule( 1, 0.05, 16000, 7854))

    def test_linear_schedule_with_zero_values(self):
        self.assertEqual(0.05, dqn.linear_schedule(0, 0.05, 16000, 10662))
        self.assertEqual(0.33362499999999995, dqn.linear_schedule(1, 0, 16000, 10662))
        self.assertEqual(0, dqn.linear_schedule(1, 0.05, 0, 10662))
        self.assertEqual(1.0, dqn.linear_schedule(1,0.05, 16000, 0))


if __name__ == '__main__':
    unittest.main()


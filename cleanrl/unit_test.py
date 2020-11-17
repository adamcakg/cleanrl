import unittest
import torch
import cleanrl.c51 as c51
import cleanrl.rnd_ppo as rnd_ppo


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

        
#       rews = 1
#        filter = rnd_ppo.RewardForwardFilter(394829482948294829481551616102958)
#        self.assertEqual(25, filter.update(25))
#        self.assertEqual(3224, filter.update(3224))
        
        


if __name__ == '__main__':
    unittest.main()




#
#        self.assertEqual('foo'.upper(), 'FOO')
#
#    def test_isupper(self):
#        self.assertTrue('FOO'.isupper())
#        self.assertFalse('Foo'.isupper())
#
#    def test_split(self):
#        s = 'hello world'
#        self.assertEqual(s.split(), ['hello', 'world'])
#        # check that s.split fails when the separator is not a string
#        with self.assertRaises(TypeError):
#            s.split(2)

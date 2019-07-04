
import torch

from utils.audio_datasets import NoisySpeechFeaturesDataset
from utils.utilities import load_hdf5

import unittest   # The test framework

TEST_DATA = "./test/test_data/data.h5"
TEST_SCALAR = "./test/test_data/scaler.p"

class Test_TestNoisySpeechFeaturesDataset_Init(unittest.TestCase):
    def test_init(self):
        self.dataset = NoisySpeechFeaturesDataset(
            TEST_DATA,
            TEST_SCALAR)


class Test_TestNoisySpeechFeaturesDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = NoisySpeechFeaturesDataset(
            TEST_DATA,
            TEST_SCALAR)

        _,x = load_hdf5(TEST_DATA)
        self.known_length = len(x)

    def test_len(self):
        length = len(self.dataset)
        self.assertEqual(length, self.known_length)
        # Test this length is accessible
        self.dataset[length-1]
        self.assertRaises(IndexError, self.dataset.__getitem__, length)

    def test_iter(self):
        for pair in self.dataset:
            self.assertEqual(len(pair),2)
            (x,y) = pair
            self.assertIsInstance(x,torch.Tensor)
            self.assertIsInstance(y,torch.Tensor)

    def test_normalized(self):
        ''' Test for normalized deviation and mean '''
        places = 3
        for pair in self.dataset:
            (x,y) = pair
            # TODO fails, but passes elsewhere
            # Check standard deviation is 1
            self.assertAlmostEqual(torch.std(x).item(),1,places)
            self.assertAlmostEqual(torch.std(y).item(),1,places)
            # Check mean is 0
            self.assertAlmostEqual(torch.mean(x).numpy(),0,places)
            self.assertAlmostEqual(torch.mean(y).numpy(),0,places)

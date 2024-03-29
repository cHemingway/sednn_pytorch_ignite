import evaluator
import soundfile
import unittest
import sys
import os
from pathlib import Path
sys.path.insert(1, Path(__file__).parent)


class Test_evaluate_metrics(unittest.TestCase):
    def setUp(self):

        our_location = os.path.dirname(__file__)
        self.clean_file = open(os.path.join(
            our_location,"test_data/clean/TEST_DR4_FDMS0_SX48.wav"), 'rb')
        self.dirty_file = open(os.path.join(
            our_location,"test_data/dirty/TEST_DR4_FDMS0_SX48.n64.wav"), 'rb')

    def tearDown(self):
        self.clean_file.close()
        self.dirty_file.close()

    def test_metrics(self):
        metrics = evaluator.evaluate_metrics(self.dirty_file, self.clean_file)
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, evaluator.Metrics)

    def test_stoi(self):
        ''' Compare STOI result with original MATLAB implementation 
        MATLAB version from C.H.Taal, see http://www.ceestaal.nl/code/
        Test files taken from TIMIT, as this is our test set
        '''
        metrics = evaluator.evaluate_metrics(self.dirty_file, self.clean_file)
        # Seems to be accurate to 4dp, which is good enough
        self.assertAlmostEqual(0.693646066121670, metrics.stoi, places=4)

    def test_bss(self):
        ''' Compare BSS results with results from MATLAB BSS_EVAL toolkit
        See http://bass-db.gforge.inria.fr/bss_eval/
        This is what mir_eval.seperation is designed to follow, so should be correct
        '''
        metrics = evaluator.evaluate_metrics(self.dirty_file, self.clean_file)
        
        self.assertAlmostEqual(0.074943771664844, metrics.sdr, places=8)

        # FIXME: Why is this Infinity? 
        # MATLAB gives same, do we need different data for a proper test?
        self.assertEqual(float("inf"), metrics.sir)

        self.assertAlmostEqual(0.074943771664844, metrics.sar, places=8)


class Test_compare_folder(unittest.TestCase):

    def setUp(self):
        our_location = os.path.dirname(__file__)
        self.clean_folder = os.path.join(our_location,"test_data/clean/")
        self.dirty_folder = os.path.join(our_location,"test_data/dirty/")

    def test_compare(self):
        metrics = evaluator.compare_folder(self.clean_folder, self.dirty_folder)
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics), 4) # Should be 4 files in folder

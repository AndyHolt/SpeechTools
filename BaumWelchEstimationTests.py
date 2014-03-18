from BaumWelchEstimation import *
import unittest
from numpy import *

class BaumWelchEstimationTests(unittest.TestCase):

  def test_nonSquareTransitionMatrix(self):
    """Transition matrices must be square."""
    transitionMatrix = array([[0.7, 0.3, 0.0],
                              [0.6, 0.4, 0.0],
                              [0.0, 0.8, 0.2],
                              [0.0, 0.8, 1.0]])
    outputProbabilities = array([[0.0, 0.5],
                                 [1.0, 0.5]])
    observationSequence = array([[0.2, 0.1, 0.1, 0.5, 0.6, 0.8, 0.7]])
    self.assertRaises(BaumWelchInputError, BaumWelch,
                      transitionMatrix, outputProbabilities, observationSequence)

  def test_nonSummingRows(self):
    """Transition matrix rows must sum to one."""
    transitionMatrix = array([[0.7, 0.3, 0.0],
                              [0.6, 0.3, 0.0],
                              [0.1, 0.8, 0.2]])
    outputProbabilities = array([[0.0, 0.5],
                                 [1.0, 0.5]])
    observationSequence = array([[0.2, 0.1, 0.1, 0.5, 0.6, 0.8, 0.7]])
    self.assertRaises(BaumWelchInputError, BaumWelch,
                      transitionMatrix, outputProbabilities, observationSequence)

  def test_outputProbabilitiesWrongRows(self):
    """Must be one less row for output probs than transition matrix"""
    transitionMatrix = array([[0.7, 0.3, 0.0],
                              [0.6, 0.4, 0.0],
                              [0.0, 0.8, 0.2]])
    outputProbabilities = array([[0.0, 0.5],
                                 [1.0, 0.5],
                                 [-1.0, 0.5]])
    observationSequence = array([[0.2, 0.1, 0.1, 0.5, 0.6, 0.8, 0.7]])
    self.assertRaises(BaumWelchInputError, BaumWelch,
                      transitionMatrix, outputProbabilities, observationSequence)

  def test_outputProbabilitiesWrongColumns(self):
    """output probs must have exactly 2 columns"""
    transitionMatrix = array([[0.7, 0.3, 0.0],
                              [0.6, 0.4, 0.0],
                              [0.0, 0.8, 0.2]])
    outputProbabilities = array([[0.0, 0.5, 0.1],
                                 [1.0, 0.5, 0.1]])
    observationSequence = array([[0.2, 0.1, 0.1, 0.5, 0.6, 0.8, 0.7]])
    self.assertRaises(BaumWelchInputError, BaumWelch,
                      transitionMatrix, outputProbabilities, observationSequence)

  def test_observationSequenceWrongSize(self):
    """observation sequence must be row vector"""
    transitionMatrix = array([[0.7, 0.3, 0.0],
                              [0.6, 0.4, 0.0],
                              [0.0, 0.8, 0.2]])
    outputProbabilities = array([[0.0, 0.5],
                                 [1.0, 0.5]])
    observationSequence = array([[0.2, 0.1, 0.1, 0.5, 0.6, 0.8, 0.7],
                                 [0.2, 0.1, 0.1, 0.5, 0.6, 0.8, 0.7]])
    self.assertRaises(BaumWelchInputError, BaumWelch,
                      transitionMatrix, outputProbabilities, observationSequence)


if __name__ == '__main__':
    unittest.main()

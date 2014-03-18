#!/usr/bin/python
"""
Baum-Welch parameter estimation.

Estimate the most probable state sequence for an HMM given the utterance output.
"""
# Author: Andy Holt (andrew.holt@hotmail.co.uk)
# Date: Tue 18 Mar 2014 15:10

from numpy import *

class BaumWelch(object):
  """
  Baum Welch parameter estimation.

  Estimate the most probable state sequence for an HMM given the utterance output.
  """

  def __init__(self, transitionMatrix, outputProbabilities, observationSequence):
    """
    Set up the HMM.

    transitionMatrix is a transition matrix array, so must be square and each
      row sum to one to be valid.
    outputProbabilities classifies output probabilities as Gaussian parameters.
      Must have one less row than transition matrix (because of non-emmitting
        states) and two columns.
      First column of each row is the mean, second column is variance.
    observation sequence is the observed output sequence.
    """
    # check that transition matrix is valid
    if transitionMatrix.shape[0] != transitionMatrix.shape[1]:
      raise BaumWelchInputError('Transition matrix must be square.')
    for row in range(transitionMatrix.shape[0]):
      if transitionMatrix[row, :].sum() != 1.0:
        raise BaumWelchInputError('Transition matrix rows must sum to 1.0')

    # check that outputProbabilities is valid
    if outputProbabilities.shape[0] != (transitionMatrix.shape[0] - 1):
      raise BaumWelchInputError('Output classification must have one row per emmitting state')
    if outputProbabilities.shape[1] != 2:
      raise BaumWelchInputError('Output classification must have exactly 2 columns')

    # check that observation sequence is valid
    if observationSequence.shape[0] != 1:
      raise BaumWelchInputError('Observation sequence should be row vector')


class BaumWelchInputError(exception):
  """
  Exceptions for input errors to BW class
  """
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)
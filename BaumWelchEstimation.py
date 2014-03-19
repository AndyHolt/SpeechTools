#!/usr/bin/python
"""
Baum-Welch parameter estimation.

Estimate the most probable state sequence for an HMM given the utterance output.
"""
# Author: Andy Holt (andrew.holt@hotmail.co.uk)
# Date: Tue 18 Mar 2014 15:10

from numpy import *
import math

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

    # if all are valid, save to object parameters
    self.transitionMatrix = transitionMatrix
    self.outputProbabilities = outputProbabilities
    self.observationSequence = observationSequence

    # other useful parameters
    # max time
    self.T = self.observationSequence.shape[1]
    self.noOfEmmittingStates = self.outputProbabilities.shape[0]

  def outputGenerationProbability(self):
    """
    Calculate probabilities of outputs for each state

    Creates 'b' matrix. Each row corresponds to a state, and each column an
    element of the observation sequence.
    """
    self.b = zeros((self.noOfEmmittingStates, self.T))
    for row in range(self.noOfEmmittingStates):
      for col in range(self.T):
        self.b[row, col] = self.gaussianDist(self.observationSequence[0, col],
                                             self.outputProbabilities[row, 0],
                                             self.outputProbabilities[row, 1])

  def forwardVariableGeneration(self):
    """
    Caclulate forward variables, alpha
    """
    self.alpha = zeros((self.noOfEmmittingStates+2, self.T + 1))

    # initialistation
    self.alpha[0,0] = 1.0
    self.alpha[1:,0] = 0.0
    self.alpha[0,1:] = 0.0

    # main recursion
    for t in range(1, self.T+1):
      for j in range(1, self.noOfEmmittingStates+1):
        partialSum = 0
        for k in range(self.noOfEmmittingStates+1):
          partialSum += (self.alpha[k, t-1] * self.transitionMatrix[k, j-1])
        self.alpha[j, t] = self.b[j-1, t-1] * partialSum
    # since must end in final state, last alpha for states with zero transition
    # prob to last state must be zero?
    for row in range(self.transitionMatrix.shape[0]):
      if self.transitionMatrix[row,-1] == 0.0:
        self.alpha[row,-1] = 0.0
    # fwd prob variable for final state at 'last' timestep gets bumped into the
    # final column to save having a needless column
    partialSum = 0
    for k in range(self.noOfEmmittingStates+1):
      partialSum += (self.alpha[k,-1] * self.transitionMatrix[k,-1])
    self.alpha[-1,-1] = partialSum

  def gaussianDist(self, x, mu, var):
    """
    Return the probability at point x for a gaussian/normal distribuion of given parameters
    """
    val = 1/(math.sqrt(2 * math.pi * var)) * math.exp(-1 * (x - mu)**2 / (2*var))
    return val


class BaumWelchInputError(Exception):
  """
  Exceptions for input errors to BW class
  """
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

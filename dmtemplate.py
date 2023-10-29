from parseQCP import *
import numpy as np
import random as random
from datetime import datetime
import time
from scipy.sparse import lil_matrix

class DMtemplate:
  circ = None

  def __init__(self, circ: QCPcircuit, noise: dict={"bitflip": 0.2, "phaseflip": 0, "bitphaseflip": 0, "depolarization": 0, "amplitudedamping": 0}) -> None:
    self.circ = circ
    self.noise = noise # Noise wird als Dictionary angegeben, für jede Form von Noise ein Eintrag mit Wahrscheinlichkeit, einzutreten

  def buildMatrix_sq(self, smallMatrix, gate):
       
    IdsBefore = gate.target
    IdsAfter = self.circ.numQubits - gate.target - 1
    
    IdMatBefore = np.eye(2**IdsBefore)
    IdMatAfter = np.eye(2**IdsAfter)

    largeMatrix = np.kron(np.kron(IdMatAfter, smallMatrix), IdMatBefore)

    return largeMatrix
  
  def buildMatrix_mq(self, matr, gate):
    mat0 = np.matrix([[1,0], [0,0]])
    mat1 = np.matrix([[0,0], [0,1]])
    matI = np.eye(2)

    tensorMat0 = np.matrix([[1]])
    tensorMat1 = np.matrix([[1]])

    IdsBefore = min(gate.target, gate.control)
    IdsAfter = self.circ.numQubits - (max(gate.target, gate.control)) - 1
    gateSize = abs(gate.target - gate.control) + 1

    IdMatBefore = np.eye(2**IdsBefore)
    IdMatAfter = np.eye(2**IdsAfter)
    
    if gate.control < gate.target:
      tensorMat0 = np.kron(mat0, tensorMat0)
      tensorMat1 = np.kron(mat1, tensorMat1)
      if gateSize > 2:
        tensorMat0 = np.kron(np.eye(2**(gateSize - 2)), tensorMat0)
        tensorMat1 = np.kron(np.eye(2**(gateSize - 2)), tensorMat1)      
      
      tensorMat0 = np.kron(matI, tensorMat0)
      tensorMat1 = np.kron(matr, tensorMat1)
    
    else:
      tensorMat0 = np.kron(matI, tensorMat0)
      tensorMat1 = np.kron(matr, tensorMat1)
      if gateSize > 2:
        tensorMat0 = np.kron(np.eye(2**(gateSize - 2)), tensorMat0)
        tensorMat1 = np.kron(np.eye(2**(gateSize - 2)), tensorMat1)      
      tensorMat0 = np.kron(mat0, tensorMat0)
      tensorMat1 = np.kron(mat1, tensorMat1)

    gateMatrix = tensorMat0 + tensorMat1
    
    return np.kron(np.kron(IdMatAfter, gateMatrix), IdMatBefore)
  
  def storeResult(self, time):
    name = self.circ.circuitName + "_output"

    with open(f'outputs/{name}.txt', 'w') as f:
      f.write(f"{self.circ.circuitName}: \n\n")

      f.write(f"Time taken for execution: {time} (s) \n\n")
      print(f"Time taken for simulation of {self.circ.circuitName}: {time} (s)")

      f.write("Measurement probabilites: \n")
      sumAmplitudes = 0
      for i in range(2**self.circ.numQubits):
        sumAmplitudes = sumAmplitudes + self.DenseMat[i,i]
        f.write(f"|{bin(i)}>: {np.real(np.round(self.DenseMat[i,i],5))} \n")  

      f.write(f"\nSum of amplitudes is: {np.round(sumAmplitudes, 20)}\n\n")  

  def preprocessCircuit(self):  
    gateMinusOne = None
    gateMinusTwo = None
    gateIterator = self.circ.gates.copy()
    for gate in gateIterator:
      # Rotation-Checker
      if (gateMinusOne != None) and gateMinusOne.name == gate.name and gateMinusOne.target == gate.target and (gate.name in ["rx", "ry", "rz"]):
        gateMinusOne.param = gateMinusOne.param + gate.param
        self.circ.gates.remove(gate)

      # Double-Checker
      elif (gateMinusOne != None) and (gate.name == gateMinusOne.name) and gate.target == gateMinusOne.target and gate.name in ["x", "y", "z", "h"]:
        self.circ.gates.remove(gateMinusOne)
        gateMinusOne = None
        self.circ.gates.remove(gate)
        
      # Sandwich-Checker
      elif (gateMinusTwo != None) and (gateMinusOne != None) and gateMinusTwo.name == "h" and gate.name == "h" and gateMinusTwo.target == gateMinusOne.target == gate.target:
        if gateMinusOne.name == "x":
          gateMinusOne.name = "z"
          self.circ.gates.remove(gateMinusTwo)
          self.circ.gates.remove(gate)
        elif gateMinusOne.name == "z":
          gateMinusOne.name = "x"
          self.circ.gates.remove(gateMinusTwo)
          self.circ.gates.remove(gate)
        elif gateMinusOne.name == "cx":
          gateMinusOne.name = "cz"
          self.circ.gates.remove(gateMinusTwo)
          self.circ.gates.remove(gate)
        elif gateMinusOne.name == "cz":
          gateMinusOne.name = "cx"
          self.circ.gates.remove(gateMinusTwo)
          self.circ.gates.remove(gate)
        elif gateMinusOne.name == "rz":
          gateMinusOne.name = "rx"
          self.circ.gates.remove(gateMinusTwo)
          self.circ.gates.remove(gate)
        elif gateMinusOne.name == "rx":
          gateMinusOne.name = "rz"
          self.circ.gates.remove(gateMinusTwo)
          self.circ.gates.remove(gate)

      else:
        gateMinusTwo = gateMinusOne
        gateMinusOne = gate

  def addnoise(self, gate):
    if self.noise["bitflip"] > 0:
      matrices = []
      p = self.noise["bitflip"]
      ErrMatBitFlip = np.matrix([[0 , 1],[1 , 0]]) * np.sqrt(p)
      noErrMatBitFlip = np.eye(2) * np.sqrt(1 - p)
      matrices.append(ErrMatBitFlip)
      matrices.append(noErrMatBitFlip)
      self.applynoise(gate, matrices)

    if self.noise["phaseflip"] > 0:
      matrices = []
      p = self.noise["phaseflip"]
      ErrMatPhaseFlip = np.matrix([[1 , 0],[0 , -1]]) * np.sqrt(p)
      noErrMatPhaseFlip = np.eye(2) * np.sqrt(1 - p)
      matrices.append(ErrMatPhaseFlip)
      matrices.append(noErrMatPhaseFlip)
      self.applynoise(gate, matrices)

    if self.noise["bitphaseflip"] > 0:
      matrices = []
      p = self.noise["bitphaseflip"]
      ErrMatBitPhaseFlip = np.matrix([[0 , -1j],[1j , 0]]) * np.sqrt(p)
      noErrMatBitPhaseFlip = np.eye(2) * np.sqrt(1 - p)
      matrices.append(ErrMatBitPhaseFlip)
      matrices.append(noErrMatBitPhaseFlip)
      self.applynoise(gate, matrices)

    if self.noise["depolarization"] > 0:
      matrices = []
      p = self.noise["depolarization"]
      noErrMatDepol = np.eye(2) * np.sqrt(1 - p)
      ErrMatDepol1 = np.matrix([[0 , 1],[1 , 0]]) * np.sqrt(p/3)
      ErrMatDepol2 = np.matrix([[0 , -1j],[1j , 0]]) * np.sqrt(p/3)
      ErrMatDepol3 = np.matrix([[1 , 0],[0 , -1]]) * np.sqrt(p/3)
      matrices.append(noErrMatDepol)
      matrices.append(ErrMatDepol1)
      matrices.append(ErrMatDepol2)
      matrices.append(ErrMatDepol3)
      self.applynoise(gate, matrices)

    if self.noise["amplitudedamping"] > 0:
      matrices = []
      p = self.noise["amplitudedamping"]
      ampdampmat1 = np.matrix([[1 , 0],[0 , np.sqrt(1 - p)]])
      ampdampmat2 = np.matrix([[0 , np.sqrt(p)],[0 , 0]])
      matrices.append(ampdampmat1)
      matrices.append(ampdampmat2)
      self.applynoise(gate, matrices)

    """
    p = self.noise["phaseflip"]
    ErrMatPhaseFlip = np.matrix([[1 , 0],[0 , -1]]) * np.sqrt(p)
    noErrMatPhaseFlip = np.eye(2) * np.sqrt(1 - p)
    
    p = self.noise["bitphaseflip"]
    ErrMatBitPhaseFlip = np.matrix([[0 , -1j],[1j , 0]]) * np.sqrt(p)
    noErrMatBitPhaseFlip = np.eye(2) * np.sqrt(1 - p)

    IdMatBefore = np.eye(2**IdsBefore)
    IdMatAfter = np.eye(2**IdsAfter)
    
    #Basic FehlerMatrizen anwenden
    if self.noise["bitflip"] > 0:
      largeMatrixBitFlip = np.kron(np.kron(IdMatAfter, ErrMatBitFlip), IdMatBefore)
      nolargeMatrixBitflip = np.kron(np.kron(IdMatAfter, noErrMatBitFlip), IdMatBefore)
      self.DenseMat = (nolargeMatrixBitflip * self.DenseMat * np.conjugate(nolargeMatrixBitflip).T) + (largeMatrixBitFlip * self.DenseMat * np.conjugate(largeMatrixBitFlip).T)
    
    if self.noise["phaseflip"] > 0:
      largeMatrixPhaseFlip = np.kron(np.kron(IdMatAfter, ErrMatPhaseFlip), IdMatBefore)
      nolargeMatrixPhaseflip = np.kron(np.kron(IdMatAfter, noErrMatPhaseFlip), IdMatBefore)
      self.DenseMat = (nolargeMatrixPhaseflip * self.DenseMat * np.conjugate(nolargeMatrixPhaseflip).T) + (largeMatrixPhaseFlip * self.DenseMat * np.conjugate(largeMatrixPhaseFlip).T)
    
    if self.noise["bitphaseflip"] > 0:
      largeMatrixBitPhaseFlip = np.kron(np.kron(IdMatAfter, ErrMatBitPhaseFlip), IdMatBefore)
      nolargeMatrixBitPhaseflip = np.kron(np.kron(IdMatAfter, noErrMatBitPhaseFlip), IdMatBefore)
      self.DenseMat = (nolargeMatrixBitPhaseflip * self.DenseMat * np.conjugate(nolargeMatrixBitPhaseflip).T) + (largeMatrixBitPhaseFlip * self.DenseMat * np.conjugate(largeMatrixBitPhaseFlip).T)

    #Depolarisation
    p = self.noise["depolarization"]
    noErrMatDepol = np.eye(2) * np.sqrt(1 - p)
    ErrMatDepol1 = np.matrix([[0 , 1],[1 , 0]]) * np.sqrt(p/3)
    ErrMatDepol2 = np.matrix([[0 , -1j],[1j , 0]]) * np.sqrt(p/3)
    ErrMatDepol3 = np.matrix([[1 , 0],[0 , -1]]) * np.sqrt(p/3)

    if self.noise["depolarization"] > 0:
      nolargeMatrixDepol = np.kron(np.kron(IdMatAfter, noErrMatDepol), IdMatBefore)
      largeMatrixDepol1 = np.kron(np.kron(IdMatAfter, ErrMatDepol1), IdMatBefore)
      largeMatrixDepol2 = np.kron(np.kron(IdMatAfter, ErrMatDepol2), IdMatBefore)
      largeMatrixDepol3 = np.kron(np.kron(IdMatAfter, ErrMatDepol3), IdMatBefore)

      self.DenseMat = (nolargeMatrixDepol * self.DenseMat * np.conjugate(nolargeMatrixDepol).T) + (largeMatrixDepol1 * self.DenseMat * np.conjugate(largeMatrixDepol1).T) + (largeMatrixDepol2 * self.DenseMat * np.conjugate(largeMatrixDepol2).T) + (largeMatrixDepol3 * self.DenseMat * np.conjugate(largeMatrixDepol3).T)

  	#AmplitudeDamping
    p = self.noise["amplitudedamping"]
    ampdampmat1 = np.matrix([[1 , 0],[0 , np.sqrt(1 - p)]])
    ampdampmat2 = np.matrix([[0 , np.sqrt(p)],[0 , 0]])
    if self.noise["amplitudedamping"] > 0:
      largeMatrixAmpDamp1 = np.kron(np.kron(IdMatAfter, ampdampmat1), IdMatBefore)
      largeMatrixAmpDamp2 = np.kron(np.kron(IdMatAfter, ampdampmat2), IdMatBefore)
      self.DenseMat = (largeMatrixAmpDamp1 * self.DenseMat * np.conjugate(largeMatrixAmpDamp1).T) + (largeMatrixAmpDamp2 * self.DenseMat * np.conjugate(largeMatrixAmpDamp2).T)
    '''
    # Create large Matrix
    largeMatrixNoError = None
    if gate.target == 0:
      largeMatrixNoError = noErrMat
    else:
      largeMatrixNoError = np.eye(2)

    for i in range(1,self.circ.numQubits):
      if i == gate.target:
        # Add error matrix
        largeMatrixNoError = np.kron(noErrMat, largeMatrixNoError)
      else:
        # Add identity
        largeMatrixNoError = np.kron(np.eye(2), largeMatrixNoError)

    largeMatrixError = None
    if gate.target == 0:
      largeMatrixError = ErrMatBitPhaseFlip
    else:
      largeMatrixError = np.eye(2)

    for i in range(1,self.circ.numQubits):
      if i == gate.target:
        # Add error matrix
        largeMatrixError = np.kron(ErrMatBitPhaseFlip, largeMatrixError)
      else:
        # Add identity
        largeMatrixError = np.kron(np.eye(2), largeMatrixError)
  	'''
    # print("ErrorMatrix:")
    # print(largeMatrixError)
    # print(largeMatrixNoError)
    #self.DenseMat = (largeMatrixNoError * self.DenseMat * np.conjugate(largeMatrixNoError).T) + (largeMatrixError * self.DenseMat * np.conjugate(largeMatrixError).T)
   """
    
  def applynoise(self,gate, matrices):
    mats = []
    for m in matrices:
      largeMatrix = self.DenseMat.copy()
      swapped = np.zeros((2**self.circ.numQubits,2**self.circ.numQubits))
      dist = 2**(gate.target)
      
      for i in range(2**self.circ.numQubits):
        for j in range(2**self.circ.numQubits):
          if swapped[i,j] != 1:
            smallMatrix = np.matrix([[self.DenseMat[i,j], self.DenseMat[i, j + dist]], [self.DenseMat[i + dist, j], self.DenseMat[i + dist, j + dist]]])

            smallMatrix = m * smallMatrix * np.conjugate(m).T

            largeMatrix[i, j] = smallMatrix[0, 0]
            largeMatrix[i, j + dist] = smallMatrix[0, 1]
            largeMatrix[i + dist, j] = smallMatrix[1, 0]
            largeMatrix[i + dist, j + dist] = smallMatrix[1, 1]

            swapped[i, j] = 1
            swapped[i , j + dist] = 1
            swapped[i + dist, j] = 1
            swapped[i + dist, j + dist] = 1

      mats.append(largeMatrix)
    res = np.matrix([0])
    for m in mats:
      res = res + m

    self.DenseMat = res

  def iterate_circ(self):
    if (self.circ is None): raise Exception("circ is None") 
    gateCounter = 0
    for gate in self.circ.gates:
      print(f"Applying gate {gateCounter}:")
      matrix = getattr(self, gate.name)(gate)

      if type(matrix) != type(None):
        self.DenseMat = matrix * self.DenseMat * np.conjugate(matrix).T
      self.addnoise(gate)
      gateCounter = gateCounter + 1

    print(self.DenseMat)

  def simulate(self):
    st = time.time()

    # self.DenseMat = lil_matrix((2**self.circ.numQubits, 2**self.circ.numQubits)) * 0j
    # self.DenseMat[0, 0] = 1
    print("gates before preprocessing:")
    print(len(self.circ.gates))

    self.preprocessCircuit()

    print("gates after preprocessing:")
    print(len(self.circ.gates))

    self.DenseMat = np.zeros((2**self.circ.numQubits,2**self.circ.numQubits)) * 0j
    self.DenseMat[0][0] = 1

    self.iterate_circ()
    et = time.time()
    res = et - st
    self.storeResult(res)

  def x(self, gate):
    swapDistance = 2**gate.target
    # Swap rows
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[[i, (i+swapDistance)]] = self.DenseMat[[(i+swapDistance), i]]
        swapped[i] = 1
        swapped[i+swapDistance] = 1

    # Swap Cols
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[:, [i, (i+swapDistance)]] = self.DenseMat[:, [(i+swapDistance), i]]
        swapped[i] = 1
        swapped[i+swapDistance] = 1      
    
    return None
    
    """
    # Alte Implementierung
    return self.buildMatrix_sq(np.matrix([[0, 1], [1, 0]]), gate)
    """
      
  def y(self, gate):
    swapDistance = 2**gate.target
    # Swap rows
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[[i, (i+swapDistance)]] = self.DenseMat[[(i+swapDistance), i]]
        self.DenseMat[i,:] *= -1j
        self.DenseMat[(i+swapDistance),:] *= 1j
        
        swapped[i] = 1
        swapped[i+swapDistance] = 1

    # Swap Cols
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[:, [i, (i+swapDistance)]] = self.DenseMat[:, [(i+swapDistance), i]]
        self.DenseMat[:,i] *= 1j
        self.DenseMat[:,(i+swapDistance)] *= -1j

        swapped[i] = 1
        swapped[i+swapDistance] = 1    
    
    return None

    """
    # Alte Implementierung
    return self.buildMatrix_sq(np.matrix([[0, -1j], [1j, 0]]), gate)
    """
       
  def z(self, gate):
    id = 2**gate.target
    # rows
    for i in range(id, 2**self.circ.numQubits, 2*id):
      for j in range(id):
        self.DenseMat[i+j,:] *= (-1)

    # cols
    for i in range(id, 2**self.circ.numQubits, 2*id):
      for j in range(id):
        self.DenseMat[:, i+j] *= (-1)
    
    """
    # Alte Implementierung:
    return self.buildMatrix_sq(np.matrix([[1, 0], [0, -1]]), gate)
    """
    
  def h(self, gate):
    """
    # Neue Implementierung
    unitary = np.matrix([[1, 1], [1, -1]]) * 1/np.sqrt(2)
    self.applyUnitary(gate, unitary)
    return None
    """
    
    
    # Alte Implementierung
    return self.buildMatrix_sq(np.matrix([[1, 1], [1, -1]]) * 1/np.sqrt(2), gate)
     
  def cx(self, gate):
    swapDistance = 2**gate.target
    ignored = 2**gate.control
    swapped = np.zeros(2**self.circ.numQubits)
    # Swap rows
    for i in range(0, 2**self.circ.numQubits, 2*ignored):
      for j in range(0, ignored):
        swapped[i+j] = 1

    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[[i, (i+swapDistance)]] = self.DenseMat[[(i+swapDistance), i]]
        swapped[i] = 1
        swapped[i+swapDistance] = 1

    # Swap Cols
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(0, 2**self.circ.numQubits, 2*ignored):
      for j in range(0, ignored):
        swapped[i+j] = 1

    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[:, [i, (i+swapDistance)]] = self.DenseMat[:, [(i+swapDistance), i]]
        swapped[i] = 1
        swapped[i+swapDistance] = 1      
    
    return None

    """
    # Alte Implementierung
    return self.buildMatrix_mq(np.matrix([[0,1], [1,0]]), gate)
    """
    
  def cz(self, gate):
    id = 2**gate.target
    ignored = 2**gate.control
    
    # rows
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(0, 2**self.circ.numQubits, 2*ignored):
      for j in range(0, ignored):
        swapped[i+j] = 1

    for i in range(id, 2**self.circ.numQubits, 2*id):
      if swapped[i] == 0:
        for j in range(id):
          self.DenseMat[i+j,:] *= (-1)

    # cols
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(0, 2**self.circ.numQubits, 2*ignored):
      for j in range(0, ignored):
        swapped[i+j] = 1

    for i in range(id, 2**self.circ.numQubits, 2*id):
      if swapped[i] == 0:
        for j in range(id):
          self.DenseMat[:, i+j] *= (-1)

    """
    # Alte Implementierung:
    return self.buildMatrix_mq(np.matrix([[1,0], [0,-1]]), gate)
    """
    
  def cy(self, gate):
    swapDistance = 2**gate.target
    ignored = 2**gate.control

    # Swap rows
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(0, 2**self.circ.numQubits, 2*ignored):
      for j in range(0, ignored):
        swapped[i+j] = 1

    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[[i, (i+swapDistance)]] = self.DenseMat[[(i+swapDistance), i]]
        self.DenseMat[i,:] *= -1j
        self.DenseMat[(i+swapDistance),:] *= 1j
        
        swapped[i] = 1
        swapped[i+swapDistance] = 1

    # Swap Cols
    swapped = np.zeros(2**self.circ.numQubits)
    for i in range(0, 2**self.circ.numQubits, 2*ignored):
      for j in range(0, ignored):
        swapped[i+j] = 1
    for i in range(2**self.circ.numQubits):
      if swapped[i] == 0:
        self.DenseMat[:, [i, (i+swapDistance)]] = self.DenseMat[:, [(i+swapDistance), i]]
        self.DenseMat[:,i] *= 1j
        self.DenseMat[:,(i+swapDistance)] *= -1j

        swapped[i] = 1
        swapped[i+swapDistance] = 1    
    
    return None
  
    """
    # Alte Implementierung
    return self.buildMatrix_mq(np.matrix([[0,-1j], [1j,0]]), gate)
    """
    
  def rx(self, gate):
    """
    unitary = np.matrix([[np.cos(gate.param / 2), np.sin(gate.param / 2) * -1j], [np.sin(gate.param / 2) * -1j, np.cos(gate.param / 2)]])
    self.applyUnitary(gate, unitary)
    return None                   
    """
    # Alte Implementierung
    return self.buildMatrix_sq(np.matrix([[np.cos(gate.param / 2), np.sin(gate.param / 2) * -1j], [np.sin(gate.param / 2) * -1j, np.cos(gate.param / 2)]]), gate)
    
  def ry(self, gate):
    """
    unitary = np.matrix([[np.cos(gate.param / 2), np.sin(gate.param / 2) * (-1)], [np.sin(gate.param / 2) , np.cos(gate.param / 2)]])
    self.applyUnitary(gate, unitary)
    return None
    """
    # Alte Implementierung
    return self.buildMatrix_sq(np.matrix([[np.cos(gate.param / 2), np.sin(gate.param / 2) * (-1)], [np.sin(gate.param / 2) , np.cos(gate.param / 2)]]), gate)
     
  def rz(self, gate):
    """
    unitary = np.matrix([[np.cos(gate.param / 2) - (1j * np.sin(gate.param / 2)), 0], [0, np.cos(gate.param / 2) + (1j * np.sin(gate.param / 2))]])
    self.applyUnitary(gate, unitary)
    return None
    """
    return self.buildMatrix_sq(np.matrix([[np.cos(gate.param / 2) - 1j * np.sin(gate.param / 2), 0], [0, np.cos(gate.param / 2) + 1j * np.sin(gate.param / 2)]]), gate)
     
  def measure(self, gate):
    for i in range(2**self.circ.numQubits):
      for j in range(2**self.circ.numQubits):
        binI = np.binary_repr(i, width=self.circ.numQubits)
        binJ = np.binary_repr(j, width=self.circ.numQubits)
        if binI[len(binI) - 1 - gate.target] != binJ[len(binJ) - 1 - gate.target]:
          self.DenseMat[i,j] = 0

  def applyUnitary(self, gate, unitary):
    swapped = np.zeros((2**self.circ.numQubits,2**self.circ.numQubits))
    dist = 2**(gate.target)

    for i in range(2**self.circ.numQubits):
      for j in range(2**self.circ.numQubits):
        if swapped[i,j] != 1:
          smallMatrix = np.matrix([[self.DenseMat[i,j], self.DenseMat[i, j + dist]], [self.DenseMat[i + dist, j], self.DenseMat[i + dist, j + dist]]])

          smallMatrix = unitary * smallMatrix * np.conjugate(unitary).T

          self.DenseMat[i, j] = smallMatrix[0, 0]
          self.DenseMat[i, j + dist] = smallMatrix[0, 1]
          self.DenseMat[i + dist, j] = smallMatrix[1, 0]
          self.DenseMat[i + dist, j + dist] = smallMatrix[1, 1]

          swapped[i, j] = 1
          swapped[i , j + dist] = 1
          swapped[i + dist, j] = 1
          swapped[i + dist, j + dist] = 1
      
if __name__ == "__main__":
  c = parseQCP("evaluation/circuits/12qb.qcp")
  # c = parseQCP("challenge/hhl_n14.qcp")
  # c = parseQCP("QCPBench/small/deutsch_n2.qcp")
  simulator = DMtemplate(c)
  # simulator = DMsim(c,{"bitflip": 0.2, "phaseflip": 0.2})
  simulator.simulate()

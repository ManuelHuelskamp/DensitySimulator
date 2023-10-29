import cmath
class Gate:
  def __init__(self, name):
    self.name = name
    self.param = None
    self.target = None
    self.control = None
  def __repr__(self):
    return self.name

  def __str__(self):
    str = self.name
    if (self.param is not None): str += f"({self.param})"
    if (self.control is not None): str += f" {self.control}"
    if (self.target is not None): str += f" {self.target}"
    return str

class QCPcircuit:
  def __init__(self) -> None:
    self.numQubits = None
    self.gates = []
    self.circuitName = None
  def __str__(self):
    str = f"{self.numQubits}\n"
    str += "\n".join([i.__str__() for i in self.gates])
    return str


def parseQCP(path):
  with open(path, "r") as fp:
    circ = QCPcircuit()
    circ.circuitName = path.split("/")[-1]
    for line in fp.read().splitlines() :
      # ignore comments
      if line.startswith('//'): continue

      # first line that is no comment has to be num of used qbits
      if (circ.numQubits is None): 
        circ.numQubits=int(line)
        continue

      gate_comp = line.split()
      gate = Gate(gate_comp[0])

      # gates with parameters
      if (line.startswith('r')): 
        gate.param = float(eval(gate_comp[1].replace("pi", str(cmath.pi))))
        gate.target = int(gate_comp[2])
        circ.gates.append(gate)
        continue

      # controlled gates
      if (line.startswith('c')):
        gate.control = int(gate_comp[1])
        gate.target = int(gate_comp[2])
        circ.gates.append(gate)
        continue

      # single qbit gates without parameters
      gate.target = int(gate_comp[1])
      circ.gates.append(gate)
  return circ
        
if __name__ == "__main__":
  c = parseQCP("code/QCPBench/small/grover_n2.qcp")
  print(c)

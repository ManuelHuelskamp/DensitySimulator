from parseQCP import *
from simulator_naive import *
from simulator_opt import *

if __name__ == "__main__":
    with open('evaluation/evaluation.csv', 'w') as e:
        e.write("numQubits, Type, Time\n")
        for j in range(2, 16):
            for i in range(3):
                c = parseQCP(f"evaluation/circuits/{j}qb.qcp")
                simO = DMtemplateOpt(c)
                simN = DMtemplateNaive(c)
                resO = simO.simulate()
                resN = simN.simulate()

                e.write(f"{j}, Optimized, {resO}\n")
                e.write(f"{j}, Naive, {resN}\n")

                
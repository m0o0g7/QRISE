"""Physical programs executed on the same processors for every policy."""
from netsquid.components import QuantumProgram
from netsquid.components.instructions import INSTR_CNOT, INSTR_H, INSTR_MEASURE, INSTR_X, INSTR_Z


class PurifyProgram(QuantumProgram):
    """One endpoint of bilateral BBPSSW; pair 0 is retained."""
    default_num_qubits = 2

    def program(self):
        keep, sacrifice = self.get_qubit_indices(2)
        self.apply(INSTR_CNOT, [keep, sacrifice])
        self.apply(INSTR_MEASURE, sacrifice, output_key="m")
        yield self.run()


class SwapProgram(QuantumProgram):
    """Bell measurement; H-side bit requests Z, other bit requests X."""
    default_num_qubits = 2

    def program(self):
        left, right = self.get_qubit_indices(2)
        self.apply(INSTR_CNOT, [left, right])
        self.apply(INSTR_H, left)
        self.apply(INSTR_MEASURE, left, output_key="z")
        self.apply(INSTR_MEASURE, right, output_key="x")
        yield self.run()


class CorrectProgram(QuantumProgram):
    """Apply correction received over the classical channel."""
    default_num_qubits = 1

    def program(self, x=0, z=0):
        q, = self.get_qubit_indices(1)
        if x:
            self.apply(INSTR_X, q)
        if z:
            self.apply(INSTR_Z, q)
        yield self.run()

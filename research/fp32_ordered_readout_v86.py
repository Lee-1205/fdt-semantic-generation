"""Same ordered fuzzy binding and tensor names; fixed FP32 verbalization."""
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.readout_precision_policy_v81 import readout

class FP32OrderedFuzzyReadout(OrderedFuzzyReadout):
    def from_commit(self,commit):
        return readout(self.readout,commit)

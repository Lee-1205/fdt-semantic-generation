"""CPU candidate: train four language tail blocks, preserving every source tensor.

No architecture, vocabulary, fuzzy mechanism, inference or p50 loss change.
Optimizer ownership migration is a separate, not-yet-complete driver contract.
"""
from research.owt_prefix_hold_v28 import PrefixHoldFuzzyAE,OBJECTIVE

TRAINABILITY={'language_tail_blocks':4,'language_embedding_trainable':False,
              'same_total_parameters':True,'new_parameters':0,
              'reasoning_diffusion_training':False,'inference_changed':False}

class ExtractionTailFuzzyAE(PrefixHoldFuzzyAE):
    def __init__(self,c):
        super().__init__(c)
        for block in self.core.language.blocks[-4:]:block.requires_grad_(True)
        # Keep the embedding frozen: thawing it would require backpropagation
        # through every preceding frozen language block as well.
        self.core.language.embedding.requires_grad_(False)

def new_trainable_names(model,source_parameter_groups):
    old=[name for group in source_parameter_groups for name in group]
    if len(old)!=len(set(old)):raise ValueError('duplicate source optimizer ownership')
    named=dict(model.named_parameters())
    if any(name not in named or not named[name].requires_grad for name in old):raise ValueError('source trainable ownership lost')
    return [name for name,p in model.named_parameters() if p.requires_grad and name not in set(old)]

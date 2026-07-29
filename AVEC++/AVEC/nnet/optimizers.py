# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.optim as optim

# NeuralNets
from nnet import embeddings
from nnet import schedulers

###############################################################################
# Optimizers
###############################################################################

class _MultiSchedulerOptimizerMixin:

    """ Shared step()/state_dict()/load_state_dict() logic for optimizers that
    support an independent Scheduler per param_group (discriminative LR), not just
    one shared scheduler for the whole optimizer.

    Usage: `lr=` (passed to __init__) is the DEFAULT scheduler, used by any
    param_group that doesn't carry its own. A param_group opts into its own
    schedule by including a "scheduler" key (a schedulers.Scheduler instance) in
    the dict passed to `params=` -- e.g. via get_discriminative_param_groups()
    below. This is how AudioVisualPretrainedConformerInterCTC.compile() gives
    newly-unfrozen wav2vec2/SwinLip layers a separate, lower-LR schedule while the
    rest of the model keeps using the original one, untouched.

    Multiple groups may share the same scheduler object (e.g. all "base" groups
    normally do) -- step() below only calls .step() once per distinct scheduler
    per optimizer step (keyed by id()), not once per group, so a shared scheduler's
    model_step increments exactly once per training step regardless of how many
    groups reference it.

    Checkpoint format is backward compatible with the old single-scheduler
    checkpoints (a scalar "model_step" key): loading one of those into a NEW
    multi-scheduler optimizer restores that step only onto the first (base)
    scheduler and leaves any additional (e.g. newly-added finetune) scheduler at
    its fresh model_step=0 -- exactly what you want when introducing a new
    discriminative-LR group partway through training: the pre-existing schedule
    resumes exactly where it left off, the new one starts its own short warmup
    from scratch. Checkpoints saved by a multi-scheduler optimizer use a
    "scheduler_steps" list instead and restore each scheduler positionally. """

    def _init_schedulers(self, lr):

        if isinstance(lr, schedulers.Scheduler):
            self.default_scheduler = lr
        else:
            self.default_scheduler = schedulers.ConstantScheduler(val=lr)

        # Resolve one scheduler per param_group: an explicit per-group "scheduler"
        # takes priority, else fall back to the shared default above. Pop it out of
        # the group dict since torch's Optimizer would otherwise choke on an
        # unrecognized key.
        self.group_schedulers = []
        for group in self.param_groups:
            sched = group.pop("scheduler", None)
            self.group_schedulers.append(sched if sched is not None else self.default_scheduler)

    def _unique_schedulers(self):
        return list({id(s): s for s in self.group_schedulers}.values())

    def _step_lrs(self):
        lr_cache = {}
        for group, sched in zip(self.param_groups, self.group_schedulers):
            if id(sched) not in lr_cache:
                lr_cache[id(sched)] = sched.step()
            group['lr'] = lr_cache[id(sched)]

    def _state_dict_schedulers(self, state_dict):
        unique_scheds = self._unique_schedulers()
        if len(unique_scheds) == 1:
            # Backward-compatible single-scheduler format
            state_dict["model_step"] = unique_scheds[0].model_step
        else:
            state_dict["scheduler_steps"] = [s.model_step for s in unique_scheds]
        return state_dict

    def _load_state_dict_schedulers(self, state_dict):
        unique_scheds = self._unique_schedulers()
        if "model_step" in state_dict:
            step = state_dict.pop("model_step")
            # Old single-scheduler checkpoint: restore it onto the first (base)
            # scheduler only. Any additional scheduler (e.g. a discriminative-LR
            # group just added this run) is left at its fresh model_step=0.
            unique_scheds[0].model_step.fill_(step)
        elif "scheduler_steps" in state_dict:
            steps = state_dict.pop("scheduler_steps")
            assert len(steps) == len(unique_scheds), \
                "checkpoint has {} scheduler(s) but optimizer has {} -- param group / scheduler setup changed since this checkpoint was saved".format(len(steps), len(unique_scheds))
            for s, step in zip(unique_scheds, steps):
                s.model_step.fill_(step)

class SGD(optim.SGD, _MultiSchedulerOptimizerMixin):

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(params=params, lr=torch.tensor(0), momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self._init_schedulers(lr)

    def step(self, closure=None):
        self._step_lrs()
        return super(SGD, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(SGD, self).state_dict()

        # Append Scheduler Step(s)
        state_dict = self._state_dict_schedulers(state_dict)

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step(s)
        self._load_state_dict_schedulers(state_dict)

        # Load State Dict
        super(SGD, self).load_state_dict(state_dict)

class Adam(optim.Adam, _MultiSchedulerOptimizerMixin):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(params=params, lr=torch.tensor(0), betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self._init_schedulers(lr)

    def step(self, closure=None):
        self._step_lrs()
        return super(Adam, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(Adam, self).state_dict()

        # Append Scheduler Step(s)
        state_dict = self._state_dict_schedulers(state_dict)

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step(s)
        self._load_state_dict_schedulers(state_dict)

        # Load State Dict
        super(Adam, self).load_state_dict(state_dict)

class AdamW(optim.AdamW, _MultiSchedulerOptimizerMixin):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        super(AdamW, self).__init__(params=params, lr=torch.tensor(0), betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self._init_schedulers(lr)

    def step(self, closure=None):
        self._step_lrs()
        return super(AdamW, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(AdamW, self).state_dict()

        # Append Scheduler Step(s)
        state_dict = self._state_dict_schedulers(state_dict)

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step(s)
        self._load_state_dict_schedulers(state_dict)

        # Load State Dict
        super(AdamW, self).load_state_dict(state_dict)

###############################################################################
# Utils
###############################################################################

def get_decay_param_groups(model, weight_decay=0.01, decay_modules=(torch.nn.Linear,), no_decay_modules=(torch.nn.LayerNorm, torch.nn.Embedding, embeddings.PosEmbedding1d), decay_params=("weight",), no_decay_params=("bias",)):

    # Init Decay / No decay Sets
    decay = set()
    no_decay = set()

    # Param Loop
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():

            # Full Param Name
            full_param_name = "{}.{}".format(module_name, param_name) if module_name else param_name

            # No Decay params
            for no_decay_param in no_decay_params:
                if param_name.endswith(no_decay_param):
                    no_decay.add(full_param_name)
                    continue

            # Decay params
            for decay_param in decay_params:

                # Decay Weight
                if param_name.endswith(decay_param) and isinstance(module, decay_modules):
                    decay.add(full_param_name)
                    continue

                # No Decay Weight
                if param_name.endswith(decay_param) and isinstance(module, no_decay_modules):
                    no_decay.add(full_param_name)
                    continue

    # Validate that we considered every parameter
    param_dict = {param_name: param for param_name, param in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters {} made it into both decay/no_decay sets!".format(str(inter_params))
    assert len(param_dict.keys() - union_params) == 0, "parameters {} were not separated into either decay/no_decay set!".format(str(param_dict.keys() - union_params))

    # Create Param Groups
    param_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    return param_groups

def get_discriminative_param_groups(model, finetune_modules, base_scheduler, finetune_scheduler):

    """ Split model.parameters() into two param_groups -- one for params belonging
    to `finetune_modules` (partially-unfrozen pretrained backbone layers, e.g.
    AudioWav2Vec2ConformerEncoder.finetune_modules / VisualSwinLipConformerEncoder
    .finetune_modules), one for everything else -- each carrying its own Scheduler
    (see _MultiSchedulerOptimizerMixin in this file for how that's consumed).

    Matches by parameter identity (id()), not name-string matching: finetune_modules
    are the actual nn.Module objects living inside the model, so their .parameters()
    are literally the same tensors as the corresponding entries in
    model.parameters() -- robust regardless of naming/nesting, unlike matching on
    dotted parameter name strings.

    Still-frozen (requires_grad=False) params fall into the "base" group same as
    before this function existed -- harmless, since they never receive gradients
    and optimizer.step() skips any param with p.grad is None; this matches the
    existing pattern of just passing self.parameters() unconditionally regardless
    of freeze status. """

    finetune_param_ids = set()
    for module in finetune_modules:
        finetune_param_ids.update(id(p) for p in module.parameters())

    base_params, finetune_params = [], []
    for p in model.parameters():
        (finetune_params if id(p) in finetune_param_ids else base_params).append(p)

    assert len(finetune_params) > 0, "finetune_modules produced no matching parameters in model.parameters() -- check that these modules are actually submodules of `model`"

    return [
        {"params": base_params, "scheduler": base_scheduler},
        {"params": finetune_params, "scheduler": finetune_scheduler},
    ]

###############################################################################
# Optimizer Dictionary
###############################################################################

optim_dict = {
    "SGD": SGD,
    "RMSprop": optim.RMSprop,
    "Adam": Adam,
    "AdamW": AdamW
}
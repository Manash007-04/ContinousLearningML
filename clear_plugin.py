"""
clear_plugin.py

Avalanche-compatible CLEAR plugin (best-effort wrapper).

This plugin extends Avalanche's ReplayPlugin to additionally store
logits for replayed samples at the time they are added to memory, and
applies a distillation loss (KL) on replayed batches during training.

Notes / compatibility:
- Avalanche's internal APIs change across versions. This implementation
  attempts to use stable hooks (`before_backward`, `after_training_exp`)
  that are commonly available in Avalanche's plugin interface. If your
  Avalanche version uses different hook names, you'll need to adapt the
  method names accordingly.
- The plugin is defensive: if required attributes/methods are missing,
  it will log a warning and fall back to vanilla replay behavior.

This file is intended as a starting point for full integration and
should be validated in your target Avalanche environment.
"""

import torch
import torch.nn.functional as F
import warnings
from avalanche.training.plugins import ReplayPlugin


class ClearReplayPlugin(ReplayPlugin):
    """Replay plugin that stores logits when samples are added and
    applies a distillation loss on replayed batches.

    Args:
        mem_size: same as ReplayPlugin
        storage_policy: Avalanche storage policy (e.g., ClassBalancedBuffer)
        clear_lambda: weight for distillation loss
        temperature: temperature for distillation
    """
    def __init__(self, mem_size, storage_policy=None, clear_lambda=1.0, temperature=2.0):
        super().__init__(mem_size=mem_size, storage_policy=storage_policy)
        self.clear_lambda = clear_lambda
        self.temperature = temperature

        # Internal storage for logits aligned with the replay buffer.
        # We keep this as a list of cpu tensors. We attempt to keep indices
        # aligned with the storage_policy ordering, but this may require
        # adaptation depending on storage_policy implementation.
        self._stored_logits = []

    # Hook called after training on an experience: attempt to snapshot logits
    def after_training_exp(self, strategy, **kwargs):
        """Snapshot logits for samples currently in the replay buffer.

        This is a conservative approach: compute logits for all elements
        currently stored in the storage_policy and cache them. Depending
        on memory size this may be somewhat costly; for large experiments
        you may prefer to store logits at insertion time instead.
        """
        try:
            storage = getattr(self, 'storage_policy', None)
            if storage is None:
                # Fallback: try to reach the underlying ReplayPlugin storage
                storage = getattr(self, 'buffer', None)

            if storage is None:
                warnings.warn("CLEAR: Could not access underlying storage_policy; skipping logits snapshot.")
                return

            # Try to obtain all stored samples. Different Avalanche versions
            # expose different APIs; try a few common ones.
            elements = None
            if hasattr(storage, 'get_all_elements'):
                elements = storage.get_all_elements()
            elif hasattr(storage, 'get_all_data'):
                elements = storage.get_all_data()
            elif hasattr(storage, 'elements'):
                elements = storage.elements

            if elements is None:
                warnings.warn("CLEAR: storage_policy does not expose a known 'get_all' API; skipping snapshot.")
                return

            # Elements are expected to be (x, y, task_labels) tuples or similar.
            logits_list = []
            device = next(strategy.model.parameters()).device
            strategy.model.eval()
            with torch.no_grad():
                for elem in elements:
                    try:
                        x = elem[0]
                        if isinstance(x, (list, tuple)):
                            x = x[0]
                        x = x.to(device)
                        out = strategy.model(x)
                        logits_list.append(out.detach().cpu())
                    except Exception:
                        # If any element can't be processed, skip it
                        logits_list.append(None)

            # Replace stored logits (best-effort)
            self._stored_logits = logits_list

        except Exception as e:
            warnings.warn(f"CLEAR: error during after_training_exp snapshot: {e}")

    def before_backward(self, strategy, **kwargs):
        """Called before backward; if current minibatch contains replayed
        examples, add distillation loss to `strategy.loss`.

        This hook assumes that the strategy exposes the current minibatch as
        `strategy.mb_x`, `strategy.mb_y` and the model outputs as
        `strategy.mb_output` or that the replayed minibatch is provided in
        `kwargs` under common keys. We try a few fallbacks.
        """
        try:
            # Try to access current minibatch outputs
            mb_output = getattr(strategy, 'mb_output', None)
            if mb_output is None:
                mb_output = kwargs.get('mb_output', None)

            # If no minibatch or no stored logits, nothing to do
            if mb_output is None or len(self._stored_logits) == 0:
                return

            # Check if there is replay data available in kwargs (common key names)
            replay_info = None
            for key in ('replay_mb_x', 'replay_x', 'replay_batch'):
                if key in kwargs:
                    replay_info = kwargs[key]
                    break

            # If we can't find an explicit replay batch, attempt to detect
            # by checking if the strategy has attribute 'replay_batch'
            if replay_info is None and hasattr(strategy, 'replay_batch'):
                replay_info = getattr(strategy, 'replay_batch')

            if replay_info is None:
                # We cannot reliably detect replay minibatch; exit gracefully
                return

            # replay_info may be a tuple (x, y) or (x, y, task_labels)
            rx = replay_info[0]
            ry = replay_info[1] if len(replay_info) > 1 else None

            device = next(strategy.model.parameters()).device
            model = strategy.model
            model.eval()
            with torch.no_grad():
                curr_logits = model(rx.to(device))

            # If we have stored logits aligned by position, try to fetch them.
            # This is best-effort and may need to be adapted to your storage_policy.
            stored_logits = None
            if len(self._stored_logits) >= curr_logits.size(0):
                # Stack first N stored logits that are not None
                gathered = [l for l in self._stored_logits if l is not None]
                if len(gathered) >= curr_logits.size(0):
                    stored_logits = torch.stack(gathered[:curr_logits.size(0)]).to(device)

            if stored_logits is None:
                # No matching stored logits available; nothing to add
                return

            # Compute distillation loss: KL between stored logits (soft targets) and current logits
            T = float(self.temperature)
            logp = F.log_softmax(curr_logits / T, dim=1)
            q = F.softmax(stored_logits / T, dim=1)
            distill = F.kl_div(logp, q, reduction='batchmean') * (T * T)

            # Add to strategy loss if available
            if hasattr(strategy, 'loss') and strategy.loss is not None:
                strategy.loss = strategy.loss + self.clear_lambda * distill
            else:
                # Try to set in kwargs - some Avalanche versions expect 'loss'
                if 'loss' in kwargs:
                    kwargs['loss'] = kwargs['loss'] + self.clear_lambda * distill

        except Exception as e:
            warnings.warn(f"CLEAR: error in before_backward hook: {e}")

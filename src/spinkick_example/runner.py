"""Runner with dense checkpoint saving support."""

import os

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class SpinkickRunner(MotionTrackingOnPolicyRunner):

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    dense_iters = frozenset(self.cfg.get("dense_save_iterations", ()))
    if not dense_iters:
      super().learn(num_learning_iterations, init_at_random_ep_len)
      return

    sparse_interval = self.cfg["save_interval"]
    self._dense_save_iters = dense_iters
    self._sparse_interval = sparse_interval
    self.cfg["save_interval"] = 1
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.cfg["save_interval"] = sparse_interval
      self._dense_save_iters = frozenset()

  def save(self, path, infos=None):
    dense_iters = getattr(self, "_dense_save_iters", frozenset())
    if dense_iters:
      it = self.current_learning_iteration
      sparse_interval = getattr(self, "_sparse_interval", 1)
      if it % sparse_interval != 0 and it not in dense_iters:
        return
    super().save(path, infos)

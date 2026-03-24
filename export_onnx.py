"""Export a specific .pt checkpoint to ONNX by replacing policy weights in an existing ONNX file."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True, help="Path to .pt file")
  parser.add_argument("--source-onnx", required=True, help="Existing ONNX (provides motion data + metadata, read-only)")
  args = parser.parse_args()

  # Output to same directory as checkpoint, e.g. model_3500.pt -> model_3500.onnx
  checkpoint_path = Path(args.checkpoint)
  output_path = checkpoint_path.with_suffix(".onnx")

  # Load actor weights from PT
  ckpt = torch.load(args.checkpoint, map_location="cpu")
  actor_state = ckpt["actor_state_dict"]

  # Load source ONNX (has motion data, metadata, and graph structure)
  model = onnx.load(args.source_onnx)

  # Replace policy weights in ONNX initializers
  for init in model.graph.initializer:
    # Map ONNX initializer names to PT state dict keys
    # e.g. "policy.mlp.0.weight" -> "mlp.0.weight"
    if init.name.startswith("policy."):
      pt_key = init.name[len("policy."):]  # strip "policy." prefix
      if pt_key in actor_state:
        new_data = actor_state[pt_key].numpy()
        new_tensor = numpy_helper.from_array(new_data, name=init.name)
        init.CopyFrom(new_tensor)
        print(f"  Replaced: {init.name} {list(new_data.shape)}")
      else:
        print(f"  Warning: {pt_key} not found in checkpoint")

  onnx.save(model, str(output_path))
  print(f"Exported: {output_path}")


if __name__ == "__main__":
  main()

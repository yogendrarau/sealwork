"""
Stage 3a: Export the trained MLP to ONNX for in-browser inference.

Loads models/mlp.pt, exports to models/mlp.onnx, then verifies the ONNX model
produces outputs numerically identical to the PyTorch model on random inputs.
Also writes models/signs.json (the label order) for the web app to load.

The web app MUST normalize landmarks exactly like collect.py:
  - subtract wrist (landmark 0) from all 21 points
  - divide by the distance from wrist to middle-finger MCP (landmark 9)
  - order: Left hand (63 values) then Right hand (63 values); missing hand = zeros
Get that wrong in JS and the model sees garbage. This is the #1 deployment bug.

Usage:
    python export_onnx.py
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = "models"
N_FEATURES = 126


class MLP(nn.Module):
    def __init__(self, in_dim=N_FEATURES, n_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def main():
    ckpt = torch.load(f"{MODEL_DIR}/mlp.pt", map_location="cpu")
    signs = ckpt["signs"]
    model = MLP(n_classes=len(signs))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, N_FEATURES)

    onnx_path = f"{MODEL_DIR}/mlp.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Torch 2.x may write weights to a sidecar (mlp.onnx.data), which ONNX
    # Runtime Web cannot load. Force everything back into a single file.
    import onnx
    m = onnx.load(onnx_path)  # loads external data into memory
    onnx.save_model(m, onnx_path, save_as_external_data=False)
    # remove any stale sidecar
    sidecar = onnx_path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"exported -> {onnx_path} (single-file, weights embedded)")

    # ---- verify parity ----
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    max_diff = 0.0
    for _ in range(100):
        x = torch.randn(1, N_FEATURES)
        with torch.no_grad():
            torch_out = model(x).numpy()
        onnx_out = sess.run(None, {"landmarks": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    print(f"max output diff over 100 random inputs: {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX output diverges from PyTorch!"
    print("PARITY OK: ONNX matches PyTorch")

    with open(f"{MODEL_DIR}/signs.json", "w") as f:
        json.dump(signs, f)
    print(f"wrote -> {MODEL_DIR}/signs.json  ({len(signs)} classes)")

    # ---- parity.json: the web app's self-test vector ----
    # A fixed raw landmark vector + this model's logits for it. The web app
    # normalizes the raw values in JS (wrist-center, scale by wrist->MCP9),
    # runs its ONNX session, and compares against these logits — catching JS
    # normalization drift before trusting predictions. Regenerated on every
    # export so the self-test never goes stale after a retrain.
    raw = np.random.RandomState(42).rand(63).astype(np.float32)
    pts = raw.reshape(21, 3).copy()
    pts -= pts[0].copy()                      # center on wrist
    scale = np.linalg.norm(pts[9])            # wrist -> middle MCP
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale
    feat = np.concatenate([np.zeros(63, np.float32), pts.flatten()])  # left=zeros, right=hand
    with torch.no_grad():
        parity_logits = model(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).tolist()
    with open(f"{MODEL_DIR}/parity.json", "w") as f:
        json.dump({"raw": [round(float(v), 7) for v in raw],
                   "logits": [round(float(v), 4) for v in parity_logits]}, f)
    print(f"wrote -> {MODEL_DIR}/parity.json  (self-test vector for the web app)")

    # report file size (matters for browser load)
    kb = os.path.getsize(onnx_path) / 1024
    print(f"onnx size: {kb:.1f} KB")
    print(f"\ndeploy: copy {MODEL_DIR}/mlp.onnx {MODEL_DIR}/signs.json {MODEL_DIR}/parity.json -> web/")


if __name__ == "__main__":
    main()

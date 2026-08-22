"""
Phase 3 — Grad-CAM for the trained yolov8n-cls model.

Hooks into model.model.model[8] -- confirmed by inspecting the REAL
yolov8n-cls architecture (not assumed from documentation) -- the last C2f
block, immediately before the Classify head at index [9]. This matches
PROJECT.md sec 4 step 4: "Grad-CAM on the final conv block for visual
explainability."
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image

FINAL_CONV_BLOCK_INDEX = 8  # model.model.model[8] -- the last C2f block, confirmed
# by printing the real architecture with a loaded yolov8n-cls checkpoint.
# index [9] is the Classify head (conv -> pool -> dropout -> linear).


class GradCAM:
    def __init__(self, yolo_model):
        """yolo_model: an ultralytics.YOLO instance with trained weights already loaded."""
        self.model = yolo_model.model  # the underlying nn.Module, not the YOLO() wrapper
        self.model.eval()
        self.target_layer = self.model.model[FINAL_CONV_BLOCK_INDEX]
        self.class_names = yolo_model.names

        self._activations = None
        self._gradients = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def _preprocess(self, img_path: str, imgsz: int = 224) -> torch.Tensor:
        """RGB, resized, scaled to [0,1] -- matching Ultralytics' own
        classification preprocessing (no mean/std normalization for yolo-cls)."""
        img = Image.open(img_path).convert("RGB").resize((imgsz, imgsz))
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        return tensor

    def generate(self, img_path: str, target_class: str | None = None) -> dict:
        """
        target_class: which class to explain ("ADHD" or "Control"). Defaults
        to the model's own predicted class if not given.

        Returns dict with heatmap (224x224 float, 0-1), overlay (224x224x3
        uint8 RGB blended with the original image), predicted_class,
        predicted_prob, and target_class (the class actually explained).
        """
        input_tensor = self._preprocess(img_path)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        # Confirmed by inspecting the real model output (not assumed): the raw
        # forward pass returns a (probs, logits) tuple, not a plain tensor --
        # Ultralytics applies softmax internally for inference convenience.
        # Grad-CAM should backprop the raw logit, not the softmaxed probability
        # (backpropping through softmax risks saturated/flattened gradients
        # when one class is already confident) -- verified probs[i] == softmax(logits)[i]
        # on real output before relying on this.
        probs, logits = output
        predicted_idx = int(probs.argmax(dim=1))

        if target_class is not None:
            matches = [k for k, v in self.class_names.items() if v == target_class]
            if not matches:
                raise ValueError(f"target_class {target_class!r} not in model classes {list(self.class_names.values())}")
            target_idx = matches[0]
        else:
            target_idx = predicted_idx  # default: explain the model's own prediction

        self.model.zero_grad()
        score = logits[0, target_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks never fired -- the forward pass didn't reach the target layer. "
                                "Check FINAL_CONV_BLOCK_INDEX still matches the model architecture.")

        # Grad-CAM: weight each activation channel by its mean gradient (how much
        # that channel mattered to the target class score), sum across channels,
        # ReLU (only positive contributions to THIS class are meaningful), resize
        # up to the input resolution, normalize to 0-1 for display.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        original = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
        heatmap_rgb = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        overlay = (0.5 * original + 0.5 * heatmap_rgb).astype(np.uint8)

        return {
            "heatmap": cam,
            "overlay": overlay,
            "predicted_class": self.class_names[predicted_idx],
            "predicted_prob": float(probs[0, predicted_idx].detach()),
            "target_class": self.class_names[target_idx],
        }
import importlib
import sys
import types

import torch


def test_img2img_execution_uses_transformer_device(monkeypatch):
    # The model module can be imported without downloading Diffusers weights.
    fake_diffusers = types.ModuleType('diffusers')
    fake_diffusers.StableDiffusion3Img2ImgPipeline = type('FakePipeline', (), {})
    monkeypatch.setitem(sys.modules, 'diffusers', fake_diffusers)
    sys.modules.pop('cfzerocrc.model', None)
    try:
        model = importlib.import_module('cfzerocrc.model')
        pipeline = model._TransformerDeviceImg2ImgPipeline()
        pipeline.transformer = types.SimpleNamespace(device=torch.device('cuda:0'))
        assert pipeline._execution_device == torch.device('cuda:0')
    finally:
        sys.modules.pop('cfzerocrc.model', None)

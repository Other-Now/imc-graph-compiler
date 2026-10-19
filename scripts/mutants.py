"""Sanity check that the bit-exact comparison can fail: break the emulator's
semantics in small ways and confirm difftest's check_quant reports MISMATCH."""
import torch
from imc import quant
from imc.difftest import check_quant, traced

orig_adc, orig_round = quant.adc, torch.round
mutants = {
    "ADC rounds down instead of half-up": lambda p, b, s: orig_adc(p - (2 ** (s - 1) if s > 0 else 0), b, s),
    "ADC clamps to 2^(b-1) instead of 2^(b-1)-1": lambda p, b, s: orig_adc(p, b + 1, s).clamp(-(2 ** (b - 1)) * 2 ** s, 2 ** (b - 1) * 2 ** s),
}
torch.manual_seed(0)
gm, calib, x = traced("cnn"), torch.randn(256, 3, 32, 32), torch.randn(2, 3, 32, 32)
cfg = quant.Config(adc_bits=4, rows=64)
for name, m in mutants.items():
    quant.adc = m
    same, _, _ = check_quant(gm, x, calib, cfg)
    print(f"{name:45s} {'NOT CAUGHT' if same else 'caught'}")
quant.adc = orig_adc

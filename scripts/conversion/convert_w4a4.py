"""A block_w4a4_v1 pack of a Wan model (SVDQuant W4A4, wan_dit_lab.quantization.w4a4) from its checkpoint and the activation
maxima of a calibration run (collect_act_amax of run_experiment.py).

    .venv\\Scripts\\python.exe scripts\\convert_w4a4.py --checkpoint checkpoints\\LongLive-2.0-5B-S2\\model_te.pt
        --model-config checkpoints\\Wan2.2-TI2V-5B\\config.json --amax configs\\calibration\\longlive2_s2_act_amax_704.pt
        --output artifacts\\longlive2_5b_s2_blocks.w4a4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import json

import torch

from wan_dit_lab.quantization import w4a4
from wan_dit_lab.models.checkpoint import model_config, self_forcing_weights
from wan_dit_lab.models.model import WanModel

KEYS = ("dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--amax", type=Path, default=None, help="activation maxima; without them no smoothing")
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant's migration strength")
    parser.add_argument("--rank", type=int, default=32, help="of the 16-bit low-rank branch")
    parser.add_argument("--fused", action="store_true", help="block_w4a4_v2: q, k, v as one layer, fused GELU MLP")
    parser.add_argument("--keep-bf16", nargs="*", default=[], help="nn.Linear of a block left in bfloat16, "
                        "e.g. cross_attn.q cross_attn.o")
    parser.add_argument("--scale-rule", choices=["max", "mse"], default="max",
                        help="scale of an INT4 group: its largest value on 7, or the least squared error (as NVFP4 "
                             "in LongLive 2.0's training)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else model_config()
    with torch.device("meta"):
        model = WanModel(**{key: spec[key] for key in KEYS})
    model.load_state_dict(self_forcing_weights(args.checkpoint), assign=True)
    amax = torch.load(args.amax) if args.amax else {}
    print(json.dumps(w4a4.build_pack(model.blocks, amax, args.output, args.checkpoint, args.alpha, args.rank,
                                     args.fused, tuple(args.keep_bf16),
                                     rule=args.scale_rule)), flush=True)

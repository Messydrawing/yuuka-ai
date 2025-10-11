# tools/convert_bin_to_safetensors.py
# 将 pytorch_model*.bin 分片转换为 *.safetensors，并生成 model.safetensors.index.json
import argparse, json
from pathlib import Path
import re

import torch
from safetensors.torch import save_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True, help="含有 bin 分片的模型目录")
    args = ap.parse_args()
    d: Path = args.model_dir
    if not d.exists():
        raise FileNotFoundError(d)

    # 1) 找索引文件
    # 常见：pytorch_model.bin.index.json 或 pytorch_model.bin.index.json（名称略有差异）
    idx_candidates = list(d.glob("pytorch_model*.bin.index.json"))
    if not idx_candidates:
        raise FileNotFoundError("未找到 *bin.index.json（权重索引），无法分片级转换。")
    idx_path = idx_candidates[0]
    print(f"[info] 使用索引: {idx_path.name}")

    with idx_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]  # param_name -> shard filename
    unique_bins = sorted(set(weight_map.values()))
    print(f"[info] 发现 bin 分片 {len(unique_bins)} 个")

    # 2) 转换每个 bin 分片为 safetensors 分片
    # 目标命名规则：model-00001-of-00007.safetensors（从 bin 名里提取编号）
    part_pat = re.compile(r".*?(\d+)-of-(\d+)\.bin$")  # 匹配 00001-of-00007
    mapping_old_to_new = {}
    for bin_name in unique_bins:
        bin_path = d / bin_name
        if not bin_path.exists():
            raise FileNotFoundError(bin_path)

        m = part_pat.match(bin_name)
        if m:
            i, n = m.group(1), m.group(2)
            new_name = f"model-{i}-of-{n}.safetensors"
        else:
            # 非分片（单文件）情况
            new_name = "model.safetensors"

        new_path = d / new_name
        if new_path.exists():
            print(f"[skip] 已存在 {new_name}")
        else:
            print(f"[convert] {bin_name} -> {new_name}")
            # 仅在你完全信任该权重来源时使用 torch.load
            state_dict = torch.load(str(bin_path), map_location="cpu")
            save_file(state_dict, str(new_path))

        mapping_old_to_new[bin_name] = new_name

    # 3) 生成 safetensors 的索引文件
    new_index = {"metadata": idx.get("metadata", {}), "weight_map": {}}
    for k, v in weight_map.items():
        new_index["weight_map"][k] = mapping_old_to_new[v]

    safetensors_index = d / "model.safetensors.index.json"
    safetensors_index.write_text(json.dumps(new_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 写出 {safetensors_index.name}。现在目录里应当有 *.safetensors 以及 model.safetensors.index.json")

if __name__ == "__main__":
    main()

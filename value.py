#!/usr/bin/env python3
"""
使用示例：

1) 同时评估三种指标（PSNR、SSIM、LPIPS）：
python value.py \
    --name exp_baseline \
    --metrics psnr ssim lpips \
    --gt_dir /path/to/gt \
    --pred_dir /path/to/pred

2) 只评估 SSIM + LPIPS，并保存报告：
python value.py \
    --name exp_lsgan \
    --metrics ssim lpips \
    --gt_dir /path/to/gt \
    --pred_dir /path/to/pred \
    --output_json ./results/exp_lsgan_metrics.json

4) 不传 --output_json 时，默认保存路径：
项目目录/metrics/<name>/metrics.json

3) 当文件名形如 xxx_real_B.png 与 xxx_fake_B.png：
python value.py \
    --name exp_pix2pix \
    --metrics psnr ssim lpips \
    --gt_dir /path/to/images \
    --pred_dir /path/to/images \
    --gt_suffix _real_B \
    --pred_suffix _fake_B

参数说明：
- --name: 实验/任务名称。
- --metrics: 选择一个或多个指标，可选 [psnr, ssim, lpips]。
- --gt_dir: GT（真值）图片文件夹路径。
- --pred_dir: 预测/输出图片文件夹路径。
- --gt_suffix / --pred_suffix: 匹配文件对时使用的可选后缀。
- --output_json: 可选，自定义 JSON 保存路径；不传时默认保存到 项目目录/metrics/<name>/metrics.json。
- --recursive: 递归搜索图片。
- --allow_resize: 尺寸不一致时，将预测图缩放到 GT 尺寸。
- --device: LPIPS 使用的设备，例如 cpu / cuda / cuda:0。

默认匹配规则补充：
- 当不传 --pred_suffix，且预测目录中同时有 xxx_fake.* 和 xxx_real.* 时，脚本会自动使用 xxx_fake.* 参与评估，并忽略 xxx_real.*。
- 例如 GT 为 4.jpg，预测目录有 4_fake.jpg 和 4_real.jpg，则会匹配 4.jpg <-> 4_fake.jpg。

依赖说明：
- LPIPS 需要安装: lpips
- SSIM 需要安装: scikit-image

效果简评：
PSNR
< 15：通常偏差较大
15~20：中等
> 20：通常不错
> 25：通常很好（任务不同会波动）
SSIM
< 0.40：结构相似度偏低
0.40~0.60：中等
0.60~0.80：较好
> 0.80：通常很好
LPIPS
> 0.50：感知差异较大
0.30~0.50：中等
0.20~0.30：较好
< 0.20：通常很好
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from options.value_options import ValueOptions

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PairItem:
    key: str
    gt_path: Path
    pred_path: Path

def _iter_images(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    else:
        files = [p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def _normalize_key(path: Path, suffix_to_strip: str) -> str:
    stem = path.stem
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        return stem[: -len(suffix_to_strip)]
    return stem


def _normalize_pred_key(path: Path, pred_suffix: str) -> Tuple[str, bool, bool]:
    """
    返回值:
    - key: 用于匹配的键
    - use_this_file: 是否将该预测图用于评估
    - ignored_real: 是否因 _real 被忽略（用于提示）
    """
    stem = path.stem

    if pred_suffix:
        if stem.endswith(pred_suffix):
            return stem[: -len(pred_suffix)], True, False
        return stem, True, False

    if stem.endswith("_fake"):
        return stem[: -len("_fake")], True, False
    if stem.endswith("_real"):
        return stem[: -len("_real")], False, True
    return stem, True, False


def build_pairs(
    gt_dir: Path,
    pred_dir: Path,
    recursive: bool,
    gt_suffix: str,
    pred_suffix: str,
) -> Tuple[List[PairItem], List[str]]:
    gt_files = _iter_images(gt_dir, recursive=recursive)
    pred_files = _iter_images(pred_dir, recursive=recursive)

    gt_map: Dict[str, Path] = {}
    pred_map: Dict[str, Path] = {}
    ignored_real_count = 0

    for p in gt_files:
        key = _normalize_key(p, gt_suffix)
        gt_map[key] = p
    for p in pred_files:
        key, use_this_file, ignored_real = _normalize_pred_key(p, pred_suffix)
        if ignored_real:
            ignored_real_count += 1
        if not use_this_file:
            continue
        pred_map[key] = p

    shared_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    missing_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))
    missing_pred = sorted(set(gt_map.keys()) - set(pred_map.keys()))

    pairs = [PairItem(key=k, gt_path=gt_map[k], pred_path=pred_map[k]) for k in shared_keys]

    notes: List[str] = []
    if ignored_real_count > 0 and not pred_suffix:
        notes.append(f"默认规则已忽略 {ignored_real_count} 个 *_real 预测文件，仅使用 *_fake 进行评估。")
    if missing_gt:
        notes.append(f"有 {len(missing_gt)} 个预测文件未找到对应 GT。")
    if missing_pred:
        notes.append(f"有 {len(missing_pred)} 个 GT 文件未找到对应预测图。")
    return pairs, notes


def load_rgb_np(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    diff = gt.astype(np.float64) - pred.astype(np.float64)
    mse = np.mean(np.square(diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception as exc:
        raise RuntimeError(
            "SSIM 依赖 scikit-image，请先安装：pip install scikit-image"
        ) from exc
    return float(ssim(gt, pred, channel_axis=2, data_range=255))


class LPIPSComputer:
    def __init__(self, device: str):
        try:
            import torch
            lpips = importlib.import_module("lpips")
        except Exception as exc:
            raise RuntimeError("LPIPS 依赖 lpips 和 torch，请先安装：pip install lpips") from exc

        self.torch = torch
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net="alex").to(self.device)
        self.model.eval()

    def _to_tensor(self, arr: np.ndarray):
        tensor = self.torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor * 2.0 - 1.0
        return tensor.to(self.device)

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> float:
        with self.torch.no_grad():
            t_gt = self._to_tensor(gt)
            t_pred = self._to_tensor(pred)
            value = self.model(t_gt, t_pred)
        return float(value.item())


def maybe_resize(pred: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    img = Image.fromarray(pred)
    img = img.resize((w, h), resample=Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def main() -> None:
    args = ValueOptions().parse()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    if not gt_dir.exists() or not gt_dir.is_dir():
        raise FileNotFoundError(f"无效的 --gt_dir: {gt_dir}")
    if not pred_dir.exists() or not pred_dir.is_dir():
        raise FileNotFoundError(f"无效的 --pred_dir: {pred_dir}")

    pairs, notes = build_pairs(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        recursive=args.recursive,
        gt_suffix=args.gt_suffix,
        pred_suffix=args.pred_suffix,
    )

    if not pairs:
        raise RuntimeError("未找到可匹配的图片对，请检查目录和后缀参数。")

    metrics = list(dict.fromkeys(args.metrics))
    lpips_computer = LPIPSComputer(args.device) if "lpips" in metrics else None

    per_image: List[Dict[str, float]] = []
    running: Dict[str, List[float]] = {m: [] for m in metrics}

    cnt = 0
    for item in pairs:
        if not cnt%5:
            print(f"processing ({cnt})-th image...")
        cnt += 1
        gt = load_rgb_np(item.gt_path)
        pred = load_rgb_np(item.pred_path)

        if gt.shape != pred.shape:
            if args.allow_resize:
                pred = maybe_resize(pred, (gt.shape[0], gt.shape[1]))
            else:
                notes.append(f"跳过 {item.key}: 尺寸不一致 {gt.shape} vs {pred.shape}。")
                continue

        row: Dict[str, float] = {"key": item.key}

        if "psnr" in metrics:
            v = compute_psnr(gt, pred)
            row["psnr"] = v
            running["psnr"].append(v)

        if "ssim" in metrics:
            v = compute_ssim(gt, pred)
            row["ssim"] = v
            running["ssim"].append(v)

        if "lpips" in metrics and lpips_computer is not None:
            v = lpips_computer(gt, pred)
            row["lpips"] = v
            running["lpips"].append(v)

        per_image.append(row)

    if not per_image:
        raise RuntimeError("没有可评估的有效图片对。若尺寸不一致，请使用 --allow_resize。")

    summary = {
        m: float(np.mean(vals)) if vals else None
        for m, vals in running.items()
    }

    report = {
        "name": args.name,
        "metrics": metrics,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "num_pairs_total": len(pairs),
        "num_pairs_evaluated": len(per_image),
        "summary": summary,
        "notes": notes,
        "per_image": per_image,
    }

    print("=" * 60)
    print(f"实验名: {args.name}")
    print(f"匹配图片对: {len(pairs)}，实际评估: {len(per_image)}")
    for m in metrics:
        value = summary[m]
        if value is None:
            print(f"{m.upper()}: N/A")
        else:
            arrow = "(越高越好)" if m in {"psnr", "ssim"} else "(越低越好)"
            print(f"{m.upper()}: {value:.6f} {arrow}")
    if notes:
        print("提示信息:")
        for n in notes:
            print(f"- {n}")
    print("=" * 60)

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        project_root = Path(__file__).resolve().parent
        out_path = project_root / "metrics" / args.name / "metrics.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON 报告已保存到: {out_path}")


if __name__ == "__main__":
    main()

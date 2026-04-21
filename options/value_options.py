import argparse


class ValueOptions:
    """This class defines options used by value.py for metric evaluation."""

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized."""
        self.initialized = False

    def initialize(self, parser):
        """Define command-line arguments for evaluation metrics."""
        parser.add_argument("--name", required=True, help="实验/任务名称。")
        parser.add_argument(
            "--metrics",
            nargs="+",
            required=True,
            choices=["psnr", "ssim", "lpips"],
            help="需要计算的指标。",
        )
        parser.add_argument("--gt_dir", required=True, help="GT（真值）图片文件夹。")
        parser.add_argument("--pred_dir", required=True, help="预测/输出图片文件夹。")
        parser.add_argument("--gt_suffix", default="", help="用于匹配的 GT 文件名后缀（可选）。")
        parser.add_argument("--pred_suffix", default="", help="用于匹配的预测文件名后缀（可选）。")
        parser.add_argument("--recursive", action="store_true", help="递归搜索图片文件。")
        parser.add_argument(
            "--allow_resize",
            action="store_true",
            help="当图片尺寸不一致时，将预测图缩放到 GT 尺寸。",
        )
        parser.add_argument(
            "--output_json",
            default="",
            help="自定义 JSON 保存路径（可选）。不传时默认保存到 项目目录/metrics/<name>/metrics.json。",
        )
        parser.add_argument(
            "--device",
            default="cuda",
            help="LPIPS 使用的设备，例如 cpu / cuda / cuda:0。",
        )

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize parser and parse command-line options."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse options and auto-select device when CUDA is unavailable."""
        opt = self.gather_options()

        if opt.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    opt.device = "cpu"
            except Exception:
                opt.device = "cpu"

        self.opt = opt
        return self.opt

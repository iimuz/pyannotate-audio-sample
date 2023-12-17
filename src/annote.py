"""pyannote-audio v3を利用して話者分離を実施する."""
import logging
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from pyannote.audio import Pipeline

_logger = logging.getLogger(__name__)


class _DeviceType(Enum):
    """pytorchを利用するデバイス設定."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class _RunConfig:
    """スクリプト実行のためのオプション."""

    device: str  # デバイス

    verbose: int  # ログレベル


def _main() -> None:
    """スクリプトのエントリポイント."""
    # 実行時引数の読み込み
    config = _parse_args()

    # ログ設定
    loglevel = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }.get(config.verbose, logging.DEBUG)
    script_filepath = Path(__file__)
    log_filepath = (
        Path("data/interim")
        / script_filepath.parent.name
        / f"{script_filepath.stem}.log"
    )
    log_filepath.parent.mkdir(exist_ok=True)
    _setup_logger(log_filepath, loglevel=loglevel)
    _logger.info(config)

    config_yaml = "./data/raw/config.yaml"  # download_model.pyでダウンロードした場所
    pipeline = Pipeline.from_pretrained(config_yaml)
    pipeline.to(torch.device(config.device))

    # apply pretrained pipeline
    diarization = pipeline("data/interim/cut.wav")

    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


def _parse_args() -> _RunConfig:
    """スクリプト実行のための引数を読み込む."""
    parser = ArgumentParser(description="pyannote-audio v3を利用して話者分離を実施する.")

    parser.add_argument(
        "-d",
        "--device",
        default=_DeviceType.CPU.value,
        choices=[v.value for v in _DeviceType],
        help="話者分離に利用するデバイス.",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="詳細メッセージのレベルを設定."
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _setup_logger(
    filepath: Path | None,  # ログ出力するファイルパス. Noneの場合はファイル出力しない.
    loglevel: int,  # 出力するログレベル
) -> None:
    """ログ出力設定

    Notes
    -----
    ファイル出力とコンソール出力を行うように設定する。
    """
    script_path = Path(__file__)
    lib_logger = logging.getLogger(f"src.{script_path.stem}")

    _logger.setLevel(loglevel)
    lib_logger.setLevel(loglevel)

    # consoleログ
    console_handler = StreamHandler()
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(
        Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
    )
    _logger.addHandler(console_handler)
    lib_logger.addHandler(console_handler)

    # ファイル出力するログ
    # 基本的に大量に利用することを想定していないので、ログファイルは多くは残さない。
    if filepath is not None:
        file_handler = RotatingFileHandler(
            filepath,
            encoding="utf-8",
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=1,
        )
        file_handler.setLevel(loglevel)
        file_handler.setFormatter(
            Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
        )
        _logger.addHandler(file_handler)
        lib_logger.addHandler(file_handler)


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        _logger.exception("Exception")
        sys.exit(1)

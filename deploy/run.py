import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

from loguru import logger

DEPLOY_DIR = Path(__file__).resolve().parent
CONFIG_DIR = str((DEPLOY_DIR / "config").resolve())

@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name="mosaic",
)
def main(cfg: DictConfig) -> None:
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, 'eval.log')
    logger.remove()
    logger.add(hydra_log_path, level='DEBUG')

    console_log_level = os.environ.get('LOGURU_LEVEL', 'INFO').upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logger.info(f'Log saved to {hydra_log_path}')

    print(f"\n config: {cfg.keys()} \n")

    print(f"\n config: {cfg.get('obs')} \n")

    print(f"\n b4 instantiate agent {cfg.agent} \n")

    agent = instantiate(cfg.agent)

    if hasattr(agent, "load_onnx_policy"):
        print(f"\n agent do has load_onnx_policy \n")

    # temp = 1
    # assert temp == 2

    agent.run()


if __name__=="__main__":
    main()
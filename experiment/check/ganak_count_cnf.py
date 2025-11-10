"""
这个脚本是，给定一个CNF文件路径，调用Ganak模型计数器来计算模型数量。
"""

import re
import sys
import argparse
import subprocess
from logzero import logger, logfile


ganak_path = "/home/sunshixin/software/ganak/ganak"  # 如果ganak在linux服务器的PATH中


def model_count_ganak(cnf_path: str) -> int:
    # ... (static methods are fine)
    result = subprocess.run([ganak_path, cnf_path],
                            capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Ganak execution failed: {result.stderr}")
        return 0

    match = re.search(r"(?:s mc|c s exact arb int)\s+(\d+)", result.stdout)
    if match:
        return int(match.group(1))

    logger.warning(f"Could not parse Ganak output: {result.stdout}")
    return 0


def run_by_command():
    parser = argparse.ArgumentParser(
        description='Use Ganak to count models from a CNF file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', "-i", type=str,
                        required=True, help='Path to the CNF file')
    args = parser.parse_args()
    count = model_count_ganak(args.input)
    print(f"Model count from Ganak: {count}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行方式运行
        run_by_command()
    else:
        # 直接运行
        # cnf_file = "/home/sunshixin/pycharm_workspace/WFOMC/models/head-middle-tail.cnf"
        cnf_file = "/home/sunshixin/pycharm_workspace/WFOMC/models/2-regular-graph.cnf"
        count = model_count_ganak(cnf_file)
        print(f"Model count from Ganak: {count}")

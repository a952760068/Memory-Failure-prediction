"""
log_simulator.py
生成模拟 HBM ECC 日志 CSV，用于端到端测试训练 Pipeline。

生成策略：
  - 多台服务器 × 多个 HBM die
  - 大部分时间段：低 CE 计数，无 UE，正常温度
  - 故障前 N 天：CE 计数指数增长 + 偶发 UE + 温度升高
  - 故障事件：fail_label=1
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np

SAMPLE_INTERVAL_MIN = 10
SAMPLES_PER_DAY = 144  # 24h × 6次/h

def generate_hbm_log(
    n_servers: int = 5,
    n_hbm_per_server: int = 4,
    total_days: int = 180,
    n_failures_per_hbm: int = 2,
    seed: int = 42,
) -> list[dict]:
    """生成完整的 HBM ECC 日志记录列表。"""
    rng = random.Random(seed)
    np.random.seed(seed)

    records = []
    base_ts = 1700000000  # 2023-11-14 22:13:20 UTC

    for srv_idx in range(n_servers):
        for hbm_idx in range(n_hbm_per_server):
            server_id = f"server_{srv_idx:03d}"
            hbm_id = hbm_idx

            total_samples = total_days * SAMPLES_PER_DAY

            # 随机选择故障时间点（保证间隔 > 30 天）
            fail_days = sorted(rng.sample(range(35, total_days - 5), n_failures_per_hbm))

            for sample_idx in range(total_samples):
                ts = base_ts + sample_idx * SAMPLE_INTERVAL_MIN * 60
                day = sample_idx // SAMPLES_PER_DAY
                fail_label = 0

                # 计算距下一个故障的天数
                days_to_fail = min(
                    (fd - day for fd in fail_days if fd > day),
                    default=9999,
                )

                # 故障当天
                if day in fail_days:
                    fail_label = 1

                # CE 基础值
                base_ce = rng.randint(0, 3)

                # 故障前 7 天：CE 增长
                if days_to_fail <= 7:
                    growth = math.exp((7 - days_to_fail) * 0.5)
                    burst_ce = int(base_ce + rng.gauss(growth * 20, growth * 5))
                    burst_ce = max(0, burst_ce)
                elif days_to_fail <= 14:
                    burst_ce = base_ce + rng.randint(0, 5)
                else:
                    burst_ce = base_ce

                # UE
                ue = 0
                if days_to_fail <= 3 and rng.random() < 0.05:
                    ue = 1

                # 温度
                base_temp = 68.0 + rng.gauss(0, 2)
                if days_to_fail <= 7:
                    base_temp += (7 - days_to_fail) * 1.5
                temp = round(min(max(base_temp + rng.gauss(0, 1), 55), 100), 1)

                # 功耗（W）
                base_power = 28.0 + rng.gauss(0, 1)
                if days_to_fail <= 7:
                    base_power += (7 - days_to_fail) * 0.5
                power = round(max(base_power + rng.gauss(0, 0.5), 10), 2)

                records.append({
                    "timestamp": ts,
                    "server_id": server_id,
                    "hbm_id": hbm_id,
                    "ce_count": burst_ce,
                    "ue_count": ue,
                    "temp_c": temp,
                    "power_w": power,
                    "fail_label": fail_label,
                })

    return records


def write_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "server_id", "hbm_id",
                  "ce_count", "ue_count", "temp_c", "power_w", "fail_label"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"生成 {len(records)} 条记录 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HBM ECC 日志模拟器")
    parser.add_argument("--out", default="../training/data/raw/simulated.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--servers", type=int, default=5)
    parser.add_argument("--hbm-per-server", type=int, default=4)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--failures", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = generate_hbm_log(
        n_servers=args.servers,
        n_hbm_per_server=args.hbm_per_server,
        total_days=args.days,
        n_failures_per_hbm=args.failures,
        seed=args.seed,
    )
    write_csv(records, Path(args.out))


if __name__ == "__main__":
    main()

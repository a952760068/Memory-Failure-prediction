"""
header_diff.py
对比新旧 rf_model.h / feature_scaler.h 的变化，
用于模型更新前评估差异范围。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_constants(header_path: Path) -> dict[str, list[float]]:
    """从 C 头文件中提取所有 static const float 数组的值。"""
    text = header_path.read_text(encoding="utf-8")
    result = {}
    # 匹配 static const float ARRAY_NAME[...] = { ... };
    pattern = re.compile(
        r"static\s+const\s+float\s+(\w+)\s*\[\d*\]\s*=\s*\{([^}]+)\}",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        name = m.group(1)
        values_str = m.group(2)
        values = [float(v.strip().rstrip("f")) for v in values_str.split(",") if v.strip()]
        result[name] = values
    return result


def diff_headers(old_path: Path, new_path: Path, tolerance: float = 0.001) -> bool:
    """比较两个头文件，打印差异摘要。返回 True 表示差异在允许范围内。"""
    old = extract_constants(old_path)
    new = extract_constants(new_path)

    all_ok = True
    for name in set(old) | set(new):
        if name not in old:
            print(f"[NEW]  {name} 仅在新版本中存在（{len(new[name])} 个值）")
            continue
        if name not in new:
            print(f"[DEL]  {name} 在新版本中已删除")
            all_ok = False
            continue

        old_v, new_v = old[name], new[name]
        if len(old_v) != len(new_v):
            print(f"[WARN] {name}: 长度变化 {len(old_v)} → {len(new_v)}")
            all_ok = False
            continue

        diffs = [abs(a - b) for a, b in zip(old_v, new_v)]
        max_diff = max(diffs)
        n_changed = sum(1 for d in diffs if d > tolerance)
        if n_changed > 0:
            avg_diff = sum(diffs) / len(diffs)
            print(f"[DIFF] {name}: {n_changed}/{len(diffs)} 个值发生变化  "
                  f"max_diff={max_diff:.6f}  avg_diff={avg_diff:.6f}")
        else:
            print(f"[OK]   {name}: 无显著变化（max_diff={max_diff:.2e}）")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="比较新旧 C 头文件的变化")
    parser.add_argument("old", help="旧版本头文件路径")
    parser.add_argument("new", help="新版本头文件路径")
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists():
        print(f"文件不存在: {old_path}"); sys.exit(1)
    if not new_path.exists():
        print(f"文件不存在: {new_path}"); sys.exit(1)

    print(f"对比: {old_path.name} → {new_path.name}\n")
    ok = diff_headers(old_path, new_path, args.tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

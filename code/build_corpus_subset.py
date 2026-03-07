#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 5M 固定 passage 子集

从 psgs_w100.tsv (~21M) 中采样固定 5M 子集:
1. 确保所有 dev set 正例 passage 都包含在内
2. 随机采样剩余 passage 补足 5M
3. 固定 seed 保证所有模型使用同一子集

用法:
    python build_corpus_subset.py
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUBSET_SIZE = 5_000_000
SEED = 42


def collect_dev_positive_ids(data_dir: str) -> set:
    """从 NQ dev 和 TriviaQA dev 中提取所有正例 passage_id"""
    must_ids = set()

    for dev_file in [f"{data_dir}/NQ/nq-dev.json", f"{data_dir}/TriviaQA/trivia-dev.json"]:
        path = Path(dev_file)
        if not path.exists():
            logger.warning(f"Dev 文件不存在: {dev_file}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            for ctx in item.get("positive_ctxs", []):
                pid = ctx.get("passage_id", ctx.get("psg_id", ""))
                if pid:
                    must_ids.add(str(pid))

        logger.info(f"  {path.name}: 提取 {len(must_ids)} 个必含 passage_id (累计)")

    return must_ids


def build_subset(corpus_path: str, must_ids: set, output_dir: str, subset_size: int):
    """扫描 corpus, 构建子集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / "corpus_5m.tsv"
    meta_file = output_path / "corpus_meta.json"

    logger.info(f"扫描语料库: {corpus_path}")
    start = time.time()

    # 第一遍: 收集 must_ids 对应行 和 其余行号
    must_lines = {}       # pid -> line_content
    other_line_offsets = []  # (line_number,) for reservoir sampling
    total_lines = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头

        for line_no, line in enumerate(f):
            parts = line.split("\t", 1)
            pid = parts[0] if parts else ""

            if pid in must_ids:
                must_lines[pid] = line
            else:
                other_line_offsets.append(line_no)

            total_lines += 1

            if total_lines % 5_000_000 == 0:
                logger.info(f"  已扫描 {total_lines / 1e6:.0f}M 行...")

    elapsed = time.time() - start
    logger.info(f"扫描完成: {total_lines} 行, must={len(must_lines)}, 耗时 {elapsed:.0f}s")

    # 采样
    need_sample = subset_size - len(must_lines)
    if need_sample <= 0:
        logger.warning(f"必含集合 ({len(must_lines)}) 已超过 {subset_size}, 不再采样")
        sampled_line_nos = set()
    else:
        random.seed(SEED)
        sampled_line_nos = set(random.sample(other_line_offsets, min(need_sample, len(other_line_offsets))))
        logger.info(f"采样 {len(sampled_line_nos)} 条 passage")

    # 第二遍: 写入子集
    logger.info("写入子集...")
    written = 0
    with open(corpus_path, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        fout.write(fin.readline())  # 写表头

        for line_no, line in enumerate(fin):
            parts = line.split("\t", 1)
            pid = parts[0] if parts else ""

            if pid in must_lines or line_no in sampled_line_nos:
                fout.write(line)
                written += 1

    logger.info(f"子集已写入: {out_file} ({written} 条)")

    # 元信息
    meta = {
        "total_corpus": total_lines,
        "must_include": len(must_lines),
        "sampled": len(sampled_line_nos),
        "subset_size": written,
        "seed": SEED,
        "source": corpus_path,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"元信息: {meta_file}")


def main():
    p = argparse.ArgumentParser(description="构建 5M 固定 passage 子集")
    p.add_argument("--corpus", type=str, default="data_set/psgs_w100.tsv")
    p.add_argument("--data_dir", type=str, default="data_set")
    p.add_argument("--output", type=str, default="experiments/corpus")
    p.add_argument("--subset_size", type=int, default=SUBSET_SIZE)
    args = p.parse_args()

    logger.info("=" * 60)
    logger.info("构建 5M 固定 passage 子集")
    logger.info("=" * 60)

    must_ids = collect_dev_positive_ids(args.data_dir)
    logger.info(f"必含 passage_id: {len(must_ids)} 个")

    build_subset(args.corpus, must_ids, args.output, args.subset_size)
    logger.info("完成!")


if __name__ == "__main__":
    main()

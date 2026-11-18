#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare NumericDocValues .dvd+.dvm size for baseline vs candidate Lucene trees.

Reports absolute sizes across a few value distributions so wins and non-wins are both visible.
This is a size microbench, not a luceneutil search QPS run.

Example:
  python3 src/python/bench_numeric_dv_blockwise_linear.py \\
    --baseline /path/to/lucene_baseline \\
    --candidate /path/to/lucene_candidate \\
    --docs 1000000
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

JAVA_SRC = r"""
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.*;

public class NumericDvSizeBench {
  enum Mode {
    EXACT_COUNTER,
    TIMESTAMP_SMALL_NOISE,
    TIMESTAMP_LARGE_NOISE,
    TIMESTAMP_SHUFFLED,
    SPARSE_TIMESTAMP,
    LOW_CARDINALITY,
    RANDOM_LONG
  }

  static Analyzer noop() {
    return new Analyzer() {
      @Override
      protected TokenStreamComponents createComponents(String fieldName) {
        throw new UnsupportedOperationException();
      }
    };
  }

  static long[] gen(Mode mode, int n, Random r) {
    long base = 1_700_000_000_000L;
    long[] v = new long[n];
    switch (mode) {
      case EXACT_COUNTER -> {
        for (int i = 0; i < n; i++) v[i] = (i + 1L) * 1000;
      }
      case TIMESTAMP_SMALL_NOISE -> {
        for (int i = 0; i < n; i++) v[i] = base + i * 1000L + (i % 7);
      }
      case TIMESTAMP_LARGE_NOISE -> {
        for (int i = 0; i < n; i++) v[i] = base + i * 1000L + (r.nextInt(100_001) - 50_000);
      }
      case TIMESTAMP_SHUFFLED -> {
        for (int i = 0; i < n; i++) v[i] = base + i * 1000L + (i % 7);
        for (int i = n - 1; i > 0; i--) {
          int j = r.nextInt(i + 1);
          long t = v[i];
          v[i] = v[j];
          v[j] = t;
        }
      }
      case SPARSE_TIMESTAMP -> {
        Arrays.fill(v, Long.MIN_VALUE);
        for (int i = 0; i < n; i++) {
          if ((i & 1) == 0) v[i] = base + i * 1000L + (i % 7);
        }
      }
      case LOW_CARDINALITY -> {
        for (int i = 0; i < n; i++) v[i] = r.nextInt(64);
      }
      case RANDOM_LONG -> {
        for (int i = 0; i < n; i++) v[i] = r.nextLong();
      }
    }
    return v;
  }

  static long[] sizes(Path dir, long[] values) throws Exception {
    Files.createDirectories(dir);
    IndexWriterConfig conf =
        new IndexWriterConfig(noop())
            .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
            .setUseCompoundFile(false);
    try (FSDirectory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, conf)) {
      for (int i = 0; i < values.length; i++) {
        Document doc = new Document();
        if (values[i] != Long.MIN_VALUE) {
          doc.add(new NumericDocValuesField("v", values[i]));
        }
        w.addDocument(doc);
      }
      w.forceMerge(1);
    }
    long dvd = 0, dvm = 0;
    try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
      for (Path p : ds) {
        String n = p.getFileName().toString();
        if (n.endsWith(".dvd")) dvd += Files.size(p);
        if (n.endsWith(".dvm")) dvm += Files.size(p);
      }
    }
    return new long[] {dvd, dvm};
  }

  public static void main(String[] args) throws Exception {
    int n = Integer.parseInt(args[0]);
    Path root = Files.createTempDirectory("numeric-dv-size");
    System.out.printf(Locale.ROOT, "DOCS %d%n", n);
    for (Mode mode : Mode.values()) {
      long[] values = gen(mode, n, new Random(42));
      long[] s = sizes(root.resolve(mode.name()), values);
      int present = 0;
      for (long x : values) {
        if (x != Long.MIN_VALUE) present++;
      }
      long total = s[0] + s[1];
      System.out.printf(
          Locale.ROOT,
          "ROW %s %d %d %d %d %.4f%n",
          mode.name(),
          s[0],
          s[1],
          total,
          present,
          present == 0 ? 0.0 : (8.0 * total) / present);
    }
  }
}
"""


def run(cmd: list[str], cwd: Path | None = None) -> str:
  print("+", " ".join(cmd), flush=True)
  return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def find_jars(lucene_root: Path) -> tuple[Path, Path]:
  core = sorted((lucene_root / "lucene/core/build/libs").glob("lucene-core-*.jar"))
  if not core:
    raise SystemExit(f"No lucene-core jar under {lucene_root}; run ./gradlew :lucene:core:jar")
  # Prefer newest snapshot version directory listing; take last sorted name.
  return core[-1], core[-1]  # analysis not needed (noop analyzer)


def build_core_jar(lucene_root: Path) -> Path:
  run(["./gradlew", ":lucene:core:jar", "-q"], cwd=lucene_root)
  core, _ = find_jars(lucene_root)
  return core


def compile_and_run(core_jar: Path, docs: int, work: Path) -> dict[str, dict[str, float]]:
  work.mkdir(parents=True, exist_ok=True)
  src = work / "NumericDvSizeBench.java"
  src.write_text(JAVA_SRC)
  classes = work / "classes"
  classes.mkdir(exist_ok=True)
  run(["javac", "--release", "25", "-cp", str(core_jar), str(src), "-d", str(classes)])
  out = run(
    [
      "java",
      "--enable-native-access=ALL-UNNAMED",
      "-cp",
      f"{classes}:{core_jar}",
      "NumericDvSizeBench",
      str(docs),
    ]
  )
  rows: dict[str, dict[str, float]] = {}
  for line in out.splitlines():
    if line.startswith("ROW "):
      # ROW mode dvd dvm total present bits
      parts = line.split()
      rows[parts[1]] = {
        "dvd": int(parts[2]),
        "dvm": int(parts[3]),
        "total": int(parts[4]),
        "present": int(parts[5]),
        "bits_per_doc": float(parts[6]),
      }
  if not rows:
    raise SystemExit("No ROW lines in bench output:\n" + out)
  return rows


def fmt_mb(n: float) -> str:
  return f"{n / (1024 * 1024):.3f}"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--baseline", required=True, type=Path, help="Lucene checkout without the change")
  p.add_argument("--candidate", required=True, type=Path, help="Lucene checkout with the change")
  p.add_argument("--docs", type=int, default=1_000_000)
  p.add_argument("--skip-build", action="store_true", help="Reuse existing jars")
  args = p.parse_args()

  if not args.skip_build:
    print("Building baseline jar...", flush=True)
    base_jar = build_core_jar(args.baseline.resolve())
    print("Building candidate jar...", flush=True)
    cand_jar = build_core_jar(args.candidate.resolve())
  else:
    base_jar = find_jars(args.baseline.resolve())[0]
    cand_jar = find_jars(args.candidate.resolve())[0]

  with tempfile.TemporaryDirectory(prefix="dv-bw-linear-") as tmp:
    tmp_path = Path(tmp)
    print(f"\nBaseline jar:  {base_jar}", flush=True)
    base_rows = compile_and_run(base_jar, args.docs, tmp_path / "base")
    print(f"Candidate jar: {cand_jar}", flush=True)
    cand_rows = compile_and_run(cand_jar, args.docs, tmp_path / "cand")

  print(f"\nNumericDocValues size (dvd+dvm), docs={args.docs}")
  print(
    f"{'mode':<24} {'baseline':>10} {'candidate':>10} {'delta':>10} {'savings':>8}"
  )
  print("-" * 68)
  for mode in base_rows:
    b = base_rows[mode]["total"]
    c = cand_rows[mode]["total"]
    delta = c - b
    savings = 0.0 if b == 0 else (1.0 - c / b) * 100.0
    print(
      f"{mode:<24} {fmt_mb(b):>7} MB {fmt_mb(c):>7} MB {fmt_mb(delta):>7} MB {savings:>6.1f}%"
    )
  print(
    "\nNotes:\n"
    "- EXACT_COUNTER / TIMESTAMP_SMALL_NOISE: values correlate with docid (best case for this encoding).\n"
    "- TIMESTAMP_LARGE_NOISE: still trending, but residuals need ~20 bits; savings are modest.\n"
    "- TIMESTAMP_SHUFFLED / LOW_CARDINALITY / RANDOM_LONG: estimator should fall back; expect ~0%.\n"
    "- Sizes are .dvd + .dvm only (compound files disabled)."
  )


if __name__ == "__main__":
  try:
    main()
  except subprocess.CalledProcessError as e:
    print(e.output, file=sys.stderr)
    sys.exit(e.returncode)

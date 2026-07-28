# Copyright (c) 2026, Athrva Pandhare (athrva98@gmail.com)
# Licensed under the Apache License, Version 2.0
"""Fetch reference chromosomes for the species-classification task.

    python benchmarks/download_genomes.py --out data/genomes            # all 5
    python benchmarks/download_genomes.py --species human mouse         # subset
    python benchmarks/download_genomes.py --chroms_per_split 2 --dry_run

Sources are Ensembl's current release, verified against
https://ftp.ensembl.org/pub/current_fasta/ . Files arrive gzipped; pyfaidx
cannot index plain gzip (only bgzf), so each chromosome is decompressed to
``{out}/{species}/{chrom}.fa`` and the .gz is removed.

Only the chromosomes the dataset will actually open are fetched: the split
lists in genomics_data.py truncated to ``--chroms_per_split``. Fetching all
chromosomes of all five species would be roughly 12 GB decompressed; the
default here is about a tenth of that. Which chromosomes are train vs test is
never changed, only how many of each are used.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.genomics_data import (          # noqa: E402
    DEFAULT_SPECIES, ENSEMBL_SOURCES, SPECIES_CHROMOSOME_SPLITS,
)

BASE = "https://ftp.ensembl.org/pub/current_fasta/{dir}/dna/{fn}"


def chroms_needed(species: str, splits, per_split: int):
    seen, out = set(), []
    for sp in splits:
        for c in SPECIES_CHROMOSOME_SPLITS[species][sp][:per_split]:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def fetch(species: str, chrom: str, out_dir: str, dry_run: bool = False):
    sp_dir, pattern = ENSEMBL_SOURCES[species]
    url = BASE.format(dir=sp_dir, fn=pattern.format(c=chrom))
    dest_dir = os.path.join(out_dir, species)
    dest = os.path.join(dest_dir, f"{chrom}.fa")
    if os.path.exists(dest):
        print(f"  have {species}/{chrom}.fa")
        return True
    if dry_run:
        print(f"  would GET {url}")
        return True

    os.makedirs(dest_dir, exist_ok=True)
    tmp_gz, tmp_fa = dest + ".gz.part", dest + ".part"
    print(f"  GET {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp_gz)
        with gzip.open(tmp_gz, "rb") as fi, open(tmp_fa, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 22)
        os.replace(tmp_fa, dest)        # only now is the file considered good
        mb = os.path.getsize(dest) / 1e6
        print(f"  -> {species}/{chrom}.fa  ({mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"  !! FAILED {species}/{chrom}: {e}")
        return False
    finally:
        for t in (tmp_gz, tmp_fa):
            if os.path.exists(t):
                os.remove(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", nargs="+", default=DEFAULT_SPECIES,
                    choices=sorted(ENSEMBL_SOURCES))
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "valid", "test"])
    ap.add_argument("--chroms_per_split", type=int, default=4)
    ap.add_argument("--out", default="data/genomes")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)

    failures = []
    for spec in a.species:
        chroms = chroms_needed(spec, a.splits, a.chroms_per_split)
        print(f"{spec}: {len(chroms)} chromosomes {chroms}")
        for c in chroms:
            if not fetch(spec, c, a.out, a.dry_run):
                failures.append(f"{spec}/{c}")

    if failures:
        print(f"\n!! {len(failures)} failed: {', '.join(failures)}")
        print("Assemblies differ in which chromosomes exist (mouse has 19 "
              "autosomes, pig 18). A missing chromosome here means that split "
              "will be short by one, not that the run is invalid: pass a "
              "smaller --chroms_per_split, or drop the species.")
        return 1
    print(f"\nok: genomes under {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

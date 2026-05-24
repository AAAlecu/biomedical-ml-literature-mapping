#!/usr/bin/env python3
"""
Randomly sample a regime audit log and prepare for manual auditing by

    i) downloading pdf files where accessible,
    2) writing the title + abstract to a text file otherwise

The script also creates csv files with information retrieved from the regime parsing log files,
to be later annotated with manual auditing

Dependencies:
    pip install pandas requests
    pip install beautifulsoup4
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import quote, unquote, urljoin, urlparse

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


TRUE_VALUES = {"true", "t", "1", "yes", "y"}


def as_bool(value) -> bool:
    """Convert CSV-ish TRUE/FALSE values to boolean."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in TRUE_VALUES


def normalize_doi(value) -> str:
    """
    Normalize a DOI or DOI URL to bare lowercase DOI.
    """
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""

    s = unquote(s)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s, flags=re.I)
    s = re.sub(r"^doi:\s*", "", s, flags=re.I)

    # If a URL contains a DOI somewhere, extract the DOI substring.
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", s, flags=re.I)
    if m:
        s = m.group(1)

    # Strip trailing punctuation sometimes introduced by exports.
    s = s.strip().strip(".;,)]}")
    return s.lower()


def doi_url(doi_or_url: str) -> Optional[str]:
    """Return a DOI resolver URL if a DOI can be extracted; otherwise return a direct URL if provided."""
    doi = normalize_doi(doi_or_url)
    if doi:
        return "https://doi.org/" + quote(doi, safe="/")
    if isinstance(doi_or_url, str) and doi_or_url.startswith(("http://", "https://")):
        return doi_or_url
    return None


def safe_doi_stem(doi: str, fallback: str) -> str:
    """Create a filesystem-safe stem that still resembles the DOI."""
    doi = normalize_doi(doi)
    if not doi:
        h = hashlib.sha1(str(fallback).encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"no_doi_{h}"
    stem = doi.replace("/", "__")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:180]


def discover_openalex_csvs(
    openalex_dir: Optional[str],
    regime: str,
    explicit_paths: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Discover OpenAlex metadata CSV files for one regime.

    Expected pattern:
        openalex_*_<REGIME>_ml_top350.csv
    """
    paths: List[Path] = []
    regime_u = regime.upper()

    if openalex_dir:
        root = Path(openalex_dir)
        if not root.exists():
            raise FileNotFoundError(f"OpenAlex directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"--openalex-dir is not a directory: {root}")

        patterns = [
            f"openalex_*_{regime_u}_ml_top350.csv",
            f"openalex_*_{regime_u.lower()}_ml_top350.csv",
        ]
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))

    if explicit_paths:
        for p in explicit_paths:
            if p:
                paths.append(Path(p))

    # Deduplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(str(path))

    return unique


def read_openalex_metadata(paths: Iterable[str]) -> pd.DataFrame:
    """
    Load one or more OpenAlex CSV files and return a DOI-indexable metadata table.

    Useful expected columns include:
        doi, title, abstract, openalex_id, authors, source, year, publication_date,
        matched_ml_terms, regime_selected, matched_regime_terms

    Missing optional columns are allowed. Files without a DOI column are skipped.
    """
    frames = []
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            print(f"[WARN] OpenAlex CSV not found: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df["_openalex_source_file"] = path.name
        if "doi" not in df.columns:
            print(f"[WARN] Skipping OpenAlex CSV without 'doi' column: {path}", file=sys.stderr)
            continue
        df["doi_norm"] = df["doi"].map(normalize_doi)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["doi_norm"])

    meta = pd.concat(frames, ignore_index=True)
    meta = meta[meta["doi_norm"].astype(str).str.len() > 0].copy()

    if meta.empty:
        return pd.DataFrame(columns=["doi_norm"])

    # Deduplicate by DOI, preferring rows with non-empty title/abstract.
    title_col = meta["title"] if "title" in meta.columns else pd.Series([""] * len(meta), index=meta.index)
    abstract_col = meta["abstract"] if "abstract" in meta.columns else pd.Series([""] * len(meta), index=meta.index)
    meta["_has_title"] = title_col.fillna("").astype(str).str.len() > 0
    meta["_has_abstract"] = abstract_col.fillna("").astype(str).str.len() > 0
    meta = meta.sort_values(
        ["doi_norm", "_has_abstract", "_has_title"],
        ascending=[True, False, False],
    )
    meta = meta.drop_duplicates("doi_norm", keep="first")
    return meta


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def get_first_existing_value(row: pd.Series, candidates: Iterable[str]) -> str:
    """Return the first non-empty value from possible column names."""
    for c in candidates:
        if c in row.index:
            v = row.get(c, "")
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
    return ""


def filter_and_sample(audit_log: str, regime: str, sample_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(audit_log)
    required = ["include_primary_only", "include_any_use", "relevant"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Audit log is missing required columns: {missing}")

    if "regime" in df.columns:
        df = df[df["regime"].astype(str).str.upper() == regime.upper()].copy()

    mask = (
        df["include_primary_only"].map(as_bool)
        & df["include_any_use"].map(as_bool)
        & df["relevant"].map(as_bool)
    )
    eligible = df[mask].copy()

    if eligible.empty:
        raise ValueError(
            "No eligible rows found after filtering on "
            "include_primary_only=TRUE, include_any_use=TRUE, relevant=TRUE."
        )

    n = min(sample_size, len(eligible))
    if n < sample_size:
        print(
            f"[WARN] Requested {sample_size} samples but only {len(eligible)} eligible rows exist. Sampling {n}.",
            file=sys.stderr,
        )

    return eligible.sample(n=n, random_state=seed).reset_index(drop=True)


def is_probably_pdf_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    if "application/pdf" in ctype:
        return True
    head = resp.content[:5] if resp.content else b""
    return head == b"%PDF-"


def request_get(session: requests.Session, url: str, timeout: int = 25) -> Optional[requests.Response]:
    try:
        return session.get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return None


def extract_pdf_links(base_url: str, html: str) -> List[str]:
    """Extract plausible direct PDF links from landing-page HTML."""
    links: List[str] = []

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")

        # Citation PDF URL metadata and other obvious PDF metadata.
        for tag in soup.find_all("meta"):
            name = (tag.get("name") or tag.get("property") or "").lower()
            content = tag.get("content")
            if content and ("pdf" in name or str(content).lower().endswith(".pdf")):
                links.append(urljoin(base_url, content))

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(" ", strip=True).lower()
            href_l = href.lower()
            if ".pdf" in href_l or "pdf" in text or "download" in text:
                links.append(urljoin(base_url, href))
    else:
        # Fallback regex extraction.
        for href in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I):
            if ".pdf" in href.lower() or "pdf" in href.lower():
                links.append(urljoin(base_url, href))

    # Remove non-http and duplicates, preserving order.
    cleaned = []
    seen = set()
    for link in links:
        link = link.strip()
        if not link.startswith(("http://", "https://")):
            continue
        if link in seen:
            continue
        seen.add(link)
        cleaned.append(link)
    return cleaned


def common_pdf_candidates(final_url: str) -> List[str]:
    """
    Conservative publisher-pattern attempts.
    These are best-effort only and never include illicit sources.
    """
    candidates: List[str] = []
    if not final_url:
        return candidates

    parsed = urlparse(final_url)
    host = parsed.netloc.lower()
    url = final_url

    if "frontiersin.org" in host and "/articles/" in url:
        candidates.append(url.rstrip("/") + "/pdf")
    if "mdpi.com" in host:
        candidates.append(url.rstrip("/") + "/pdf")
    if "plos.org" in host and "/article?id=" in url:
        candidates.append(url.replace("/article?id=", "/article/file?id=") + "&type=printable")
    if "nature.com" in host and "/articles/" in url:
        candidates.append(url.rstrip("/") + ".pdf")
    if "springer.com" in host and ("/article/" in url or "/chapter/" in url):
        candidates.append(url.rstrip("/") + ".pdf")
    if "biomedcentral.com" in host and "/articles/" in url:
        candidates.append(url.rstrip("/") + ".pdf")

    return candidates

def try_download_pdf(
    session: requests.Session,
    doi_or_url: str,
    output_pdf: Path,
    sleep_seconds: float = 0.5,
) -> bool:

    start_url = doi_url(doi_or_url)
    if not start_url:
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 manual-audit-retriever/1.0 (+https://openalex.org/)",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    }
    session.headers.update(headers)

    resp = request_get(session, start_url)
    time.sleep(sleep_seconds)

    if resp is None:
        return False

    if resp.status_code >= 400:
        return False

    if is_probably_pdf_response(resp):
        output_pdf.write_bytes(resp.content)
        return True

    html = resp.text if resp.text else ""
    pdf_candidates = extract_pdf_links(resp.url, html)
    pdf_candidates.extend(common_pdf_candidates(resp.url))

    seen = set()
    candidates = []
    for link in pdf_candidates:
        if link not in seen:
            seen.add(link)
            candidates.append(link)

    for link in candidates:
        r = request_get(session, link)
        time.sleep(sleep_seconds)

        if r is None:
            continue

        if r.status_code >= 400:
            continue

        if is_probably_pdf_response(r):
            output_pdf.write_bytes(r.content)
            return True

    return False

def write_title_abstract_txt(path: Path, title: str, abstract: str) -> None:
    """Write title and abstract on separate lines with one empty line between them."""
    title = "" if pd.isna(title) else str(title).strip()
    abstract = "" if pd.isna(abstract) else str(abstract).strip()
    path.write_text(f"{title}\n\n{abstract}\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample eligible audit-log papers and retrieve accessible PDFs or title/abstract text fallbacks."
    )
    parser.add_argument("--regime", default="S", choices=["S", "M", "T", "H"], help="Regime label.")
    parser.add_argument("--audit-log", default="./regime_parse_outputs/S/S_paper_audit_log.csv", help="Path to regime audit log CSV.")
    parser.add_argument("--openalex-dir",default="./query_outputs/S",help="Directory containing OpenAlex metadata CSV files. The script reads all files matching openalex_*_<REGIME>_ml_top350.csv, e.g. openalex_eicu_S_ml_top350.csv.")
    parser.add_argument("--output-dir", default="./audit_papers/S", help="Output directory. Default: manual_audit_<REGIME>_sample_<N>")

    # parser.add_argument("--regime", default="M", choices=["S", "M", "T", "H"], help="Regime label.")
    # parser.add_argument("--audit-log", default="./regime_parse_outputs/M/M_paper_audit_log.csv", help="Path to regime audit log CSV.")
    # parser.add_argument("--openalex-dir",default="./query_outputs/M",help="Directory containing OpenAlex metadata CSV files. The script reads all files matching openalex_*_<REGIME>_ml_top350.csv, e.g. openalex_eicu_S_ml_top350.csv.")
    # parser.add_argument("--output-dir", default="./audit_papers/M", help="Output directory. Default: manual_audit_<REGIME>_sample_<N>")

    # parser.add_argument("--regime", default="T", choices=["S", "M", "T", "H"], help="Regime label.")
    # parser.add_argument("--audit-log", default="./regime_parse_outputs/T/T_paper_audit_log.csv", help="Path to regime audit log CSV.")
    # parser.add_argument("--openalex-dir",default="./query_outputs/T",help="Directory containing OpenAlex metadata CSV files. The script reads all files matching openalex_*_<REGIME>_ml_top350.csv, e.g. openalex_eicu_S_ml_top350.csv.")
    # parser.add_argument("--output-dir", default="./audit_papers/T", help="Output directory. Default: manual_audit_<REGIME>_sample_<N>")

    # parser.add_argument("--regime", default="H", choices=["S", "M", "T", "H"], help="Regime label.")
    # parser.add_argument("--audit-log", default="./regime_parse_outputs/H/H_paper_audit_log.csv", help="Path to regime audit log CSV.")
    # parser.add_argument("--openalex-dir",default="./query_outputs/H",help="Directory containing OpenAlex metadata CSV files. The script reads all files matching openalex_*_<REGIME>_ml_top350.csv, e.g. openalex_eicu_S_ml_top350.csv.")
    # parser.add_argument("--output-dir", default="./audit_papers/H", help="Output directory. Default: manual_audit_<REGIME>_sample_<N>")

    parser.add_argument("--openalex-csv",nargs="*",default=[],help="Optional explicit OpenAlex metadata CSVs containing doi/title/abstract columns. These are added to files discovered from --openalex-dir.")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of eligible rows to sample.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Delay between web requests.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt PDF downloads; write title/abstract text only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir or f"manual_audit_{args.regime}_sample_{args.sample_size}")
    papers_dir = out_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    sampled = filter_and_sample(args.audit_log, args.regime, args.sample_size, args.seed)

    # Prefer DOI from audit log's doi_or_url; fallback merge columns are added later.
    sampled = ensure_columns(sampled, ["doi_or_url", "title"])
    sampled["doi_norm"] = sampled["doi_or_url"].map(normalize_doi)

    openalex_paths = discover_openalex_csvs(
        openalex_dir=args.openalex_dir,
        regime=args.regime,
        explicit_paths=args.openalex_csv,
    )

    if not openalex_paths:
        print(
            f"[WARN] No OpenAlex files found in {args.openalex_dir!r} matching "
            f"openalex_*_{args.regime.upper()}_ml_top350.csv. "
            "Fallback .txt files will use only audit-log title/abstract columns if present.",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] OpenAlex CSV files selected for regime {args.regime}:")
        for p in openalex_paths:
            print(f"       - {p}")

    meta = read_openalex_metadata(openalex_paths)

    if not meta.empty:
        keep_cols = [
            c for c in [
                "doi_norm",
                "openalex_id",
                "doi",
                "title",
                "abstract",
                "authors",
                "source",
                "year",
                "publication_date",
                "matched_ml_terms",
                "regime_selected",
                "matched_regime_terms",
                "_openalex_source_file",
            ]
            if c in meta.columns
        ]
        meta_small = meta[keep_cols].copy()
        rename_map = {c: f"openalex_{c}" for c in meta_small.columns if c != "doi_norm"}
        meta_small = meta_small.rename(columns=rename_map)
        sampled = sampled.merge(meta_small, on="doi_norm", how="left")
    else:
        sampled["openalex_title"] = ""
        sampled["openalex_abstract"] = ""
        sampled["openalex_doi"] = ""

    sampled = ensure_columns(
        sampled,
        [
            "doi_or_url",
            "title",
            "abstract",
            "openalex_title",
            "openalex_abstract",
            "openalex_doi",
            "openalex_authors",
            "openalex_source",
            "openalex_year",
            "openalex_publication_date",
            "openalex_openalex_id",
            "openalex_matched_ml_terms",
            "openalex_matched_regime_terms",
            "openalex__openalex_source_file",
        ],
    )

    session = requests.Session()
    sampled["pdf_downloaded"] = ""
    pdf_downloaded_count = 0
    fallback_written_count = 0

    for i, row in sampled.iterrows():
        doi_clean = normalize_doi(row.get("doi_or_url", "")) or normalize_doi(row.get("openalex_doi", ""))
        fallback_key = get_first_existing_value(row, ["doi_or_url", "openalex_doi", "title", "openalex_title"]) or f"row_{i}"
        stem = safe_doi_stem(doi_clean, fallback_key)
        sample_id = f"{i + 1:03d}"
        paper_dir = papers_dir / f"{sample_id}_{stem}"
        paper_dir.mkdir(parents=True, exist_ok=True)

        title = get_first_existing_value(row, ["openalex_title", "title"])
        abstract = get_first_existing_value(row, ["openalex_abstract", "abstract"])

        pdf_path = paper_dir / f"{stem}.pdf"
        txt_path = paper_dir / f"{stem}.txt"

        if not args.no_download:
            pdf_downloaded = try_download_pdf(
                session=session,
                doi_or_url=row.get("doi_or_url", "") or row.get("openalex_doi", ""),
                output_pdf=pdf_path,
                sleep_seconds=args.sleep_seconds,
            )
        else:
            pdf_downloaded = False

        sampled.at[i, "pdf_downloaded"] = "TRUE" if pdf_downloaded else "FALSE"

        if pdf_downloaded:
            status = "pdf_downloaded"
            pdf_downloaded_count += 1
        else:
            write_title_abstract_txt(txt_path, title=title, abstract=abstract)
            status = "title_abstract_txt_written"
            fallback_written_count += 1

        print(f"[{sample_id}/{len(sampled):03d}] {status}: {stem}")

    sampled_csv = out_dir / f"{args.regime}_sample_{len(sampled)}.csv"
    excluded_output_cols = [
        "abstract",
    ]
    sampled.drop(columns=[c for c in excluded_output_cols if c in sampled.columns]).to_csv(sampled_csv, index=False)

    print("\nDone.")
    print(f"Sampled CSV:            {sampled_csv}")
    print(f"Paper files directory:  {papers_dir}")
    print(f"OpenAlex files loaded:  {len(openalex_paths)}")
    print(f"PDFs downloaded:        {pdf_downloaded_count}/{len(sampled)}")
    print(f"Text fallbacks written: {fallback_written_count}/{len(sampled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

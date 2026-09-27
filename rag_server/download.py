"""Download public retrieval resources and assemble the index and corpus."""
import argparse
import gzip
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--index-repo", default="PeterJinGo/wiki-18-e5-index")
    parser.add_argument("--corpus-repo", default="PeterJinGo/wiki-18-corpus")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    directory = Path(args.save_path)
    directory.mkdir(parents=True, exist_ok=True)
    parts = [hf_hub_download(repo_id=args.index_repo, filename=name,
                             repo_type="dataset", local_dir=directory)
             for name in ("part_aa", "part_ab")]
    compressed = hf_hub_download(repo_id=args.corpus_repo, filename="wiki-18.jsonl.gz",
                                 repo_type="dataset", local_dir=directory)
    # Use temporary output files so an interrupted conversion can be restarted.
    index = directory / "e5_Flat.index"
    if not index.exists():
        temporary = index.with_suffix(".partial")
        with temporary.open("wb") as destination:
            for part in parts:
                with open(part, "rb") as source:
                    shutil.copyfileobj(source, destination)
        temporary.replace(index)
    corpus = directory / "wiki-18.jsonl"
    if not corpus.exists():
        temporary = corpus.with_suffix(".partial")
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        temporary.replace(corpus)


if __name__ == "__main__":
    main()

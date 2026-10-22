import argparse
from pathlib import Path

from src.artifact_bundle import ArtifactBundleError, create_artifact_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package local model artifacts for private cloud deployment."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/churn-model-bundle.tar.gz"),
    )
    args = parser.parse_args()
    try:
        checksum = create_artifact_bundle(
            args.dataset,
            args.model,
            args.metadata,
            args.output,
        )
    except ArtifactBundleError as exc:
        parser.error(str(exc))
    print(f"Bundle created locally: {args.output}")
    print(f"Bundle size: {args.output.stat().st_size} bytes")
    print(f"MODEL_BUNDLE_SHA256={checksum}")


if __name__ == "__main__":
    main()

from src.artifact_bundle import ArtifactBundleError, prepare_runtime_artifacts


def main() -> None:
    try:
        prepare_runtime_artifacts()
    except ArtifactBundleError as exc:
        raise SystemExit(f"Model artifact provisioning failed: {exc}") from exc


if __name__ == "__main__":
    main()

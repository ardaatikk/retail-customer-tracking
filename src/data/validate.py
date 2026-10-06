from pathlib import Path
from zipfile import BadZipFile, ZipFile

from src.config.paths import (
    CITIC_MTMC_DIR,
    CITIC_MTMC_DATASET_DIR,
)


EXPECTED_ARCHIVE_NAME = "citic-mtmc-dataset-v2025-11-03_lfs.zip"

EXPECTED_CAPTURES = (
    "capture1",
    "capture2",
    "capture3",
)

EXPECTED_CAMERAS = (
    "HallSeg",
    "HallWide",
    "PC",
    "PF",
    "Showroom",
)


def get_archive_path() -> Path:
    return CITIC_MTMC_DIR / EXPECTED_ARCHIVE_NAME


def validate_archive() -> None:
    archive_path = get_archive_path()

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {archive_path}"
        )

    if archive_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset archive is empty: {archive_path}"
        )

    try:
        with ZipFile(archive_path, "r") as archive:
            corrupted_file = archive.testzip()

            if corrupted_file is not None:
                raise ValueError(
                    "Corrupted file detected inside archive: "
                    f"{corrupted_file}"
                )

    except BadZipFile as error:
        raise ValueError(
            f"Invalid ZIP archive: {archive_path}"
        ) from error


def validate_dataset_directory() -> None:
    if not CITIC_MTMC_DATASET_DIR.exists():
        raise FileNotFoundError(
            "Extracted dataset directory not found: "
            f"{CITIC_MTMC_DATASET_DIR}"
        )

    if not CITIC_MTMC_DATASET_DIR.is_dir():
        raise ValueError(
            "Dataset path is not a directory: "
            f"{CITIC_MTMC_DATASET_DIR}"
        )


def validate_captures() -> None:
    for capture_name in EXPECTED_CAPTURES:
        capture_dir = (
            CITIC_MTMC_DATASET_DIR / capture_name
        )

        if not capture_dir.is_dir():
            raise FileNotFoundError(
                f"Missing capture directory: {capture_dir}"
            )


def validate_videos() -> int:
    video_count = 0

    for capture_name in EXPECTED_CAPTURES:
        capture_dir = (
            CITIC_MTMC_DATASET_DIR / capture_name
        )

        for camera_name in EXPECTED_CAMERAS:
            video_path = (
                capture_dir / f"{camera_name}.mp4"
            )

            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video: {video_path}"
                )

            if video_path.stat().st_size == 0:
                raise ValueError(
                    f"Empty video file: {video_path}"
                )

            video_count += 1

    return video_count


def validate_ground_truth() -> int:
    gt_count = 0

    for capture_name in EXPECTED_CAPTURES:
        gt_path = (
            CITIC_MTMC_DATASET_DIR
            / f"{capture_name}_gt.mot.txt"
        )

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Missing ground-truth file: {gt_path}"
            )

        if gt_path.stat().st_size == 0:
            raise ValueError(
                f"Empty ground-truth file: {gt_path}"
            )

        gt_count += 1

    return gt_count


def validate_map_config() -> None:
    map_config_dir = (
        CITIC_MTMC_DATASET_DIR / "map_config"
    )

    if not map_config_dir.is_dir():
        raise FileNotFoundError(
            f"Missing map_config directory: {map_config_dir}"
        )

    for camera_name in EXPECTED_CAMERAS:
        config_path = (
            map_config_dir
            / f"Node{camera_name}.json"
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing camera configuration: {config_path}"
            )


def validate_dataset() -> None:
    print("Validating CITIC-MTMC dataset...")

    validate_archive()
    validate_dataset_directory()
    validate_captures()

    video_count = validate_videos()
    gt_count = validate_ground_truth()

    validate_map_config()

    print()
    print("Dataset summary")
    print("---------------")
    print(f"Captures: {len(EXPECTED_CAPTURES)}")
    print(f"Cameras per capture: {len(EXPECTED_CAMERAS)}")
    print(f"Videos: {video_count}")
    print(f"Ground-truth files: {gt_count}")

    print()
    print("CITIC-MTMC dataset validation passed.")


if __name__ == "__main__":
    validate_dataset()
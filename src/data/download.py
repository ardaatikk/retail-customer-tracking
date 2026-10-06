import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from src.config.paths import CITIC_MTMC_DIR
from src.config.settings import CITIC_ZENODO_API_URL


CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_RETRIES = 5


def create_session() -> requests.Session:
    retry_strategy = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def get_zenodo_files(
    session: requests.Session,
) -> list[dict]:
    response = session.get(
        CITIC_ZENODO_API_URL,
        timeout=(10, 60),
    )
    response.raise_for_status()

    record = response.json()

    return record["files"]


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    expected_size: int,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    # Handle a partial file created by the previous version
    # of the downloader.
    if destination.exists():
        current_size = destination.stat().st_size

        if current_size == expected_size:
            print(
                f"Already downloaded, skipping: "
                f"{destination.name}"
            )
            return

        print(
            f"Incomplete existing file detected: "
            f"{destination.name}"
        )

        if partial_path.exists():
            partial_path.unlink()

        destination.rename(partial_path)

    downloaded_size = (
        partial_path.stat().st_size
        if partial_path.exists()
        else 0
    )

    if downloaded_size > expected_size:
        print(
            "Partial file is larger than expected. "
            "Restarting download."
        )
        partial_path.unlink()
        downloaded_size = 0

    headers = {}

    if downloaded_size > 0:
        headers["Range"] = (
            f"bytes={downloaded_size}-"
        )
        print(
            f"Resuming {destination.name} "
            f"from {downloaded_size / 1024**2:.1f} MB..."
        )
    else:
        print(
            f"Downloading {destination.name}..."
        )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            current_size = (
                partial_path.stat().st_size
                if partial_path.exists()
                else 0
            )

            request_headers = {}

            if current_size > 0:
                request_headers["Range"] = (
                    f"bytes={current_size}-"
                )

            with session.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(10, 120),
            ) as response:
                response.raise_for_status()

                # HTTP 206 means the server accepted
                # our Range request.
                if current_size > 0:
                    if response.status_code == 206:
                        file_mode = "ab"
                    else:
                        print(
                            "Server does not support resume. "
                            "Restarting download."
                        )
                        current_size = 0
                        file_mode = "wb"
                else:
                    file_mode = "wb"

                with partial_path.open(file_mode) as file, tqdm(
                    total=expected_size,
                    initial=current_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                ) as progress:
                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if not chunk:
                            continue

                        file.write(chunk)
                        progress.update(len(chunk))

            final_size = partial_path.stat().st_size

            if final_size != expected_size:
                raise RuntimeError(
                    "Downloaded file size does not match "
                    f"expected size: {final_size} != "
                    f"{expected_size}"
                )

            partial_path.rename(destination)

            print(
                f"Download complete: {destination.name}"
            )
            return

        except (
            requests.RequestException,
            RuntimeError,
        ) as error:
            if attempt == MAX_RETRIES:
                raise

            wait_seconds = 2**attempt

            print()
            print(
                f"Download interrupted: {error}"
            )
            print(
                f"Retrying in {wait_seconds} seconds "
                f"({attempt}/{MAX_RETRIES})..."
            )

            time.sleep(wait_seconds)


def download_dataset() -> None:
    print(
        "Fetching CITIC-MTMC dataset metadata..."
    )

    session = create_session()

    files = get_zenodo_files(session)

    print(
        f"Found {len(files)} files on Zenodo."
    )

    CITIC_MTMC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_info in files:
        filename = Path(file_info["key"]).name
        download_url = file_info["links"]["self"]
        expected_size = int(file_info["size"])

        destination = (
            CITIC_MTMC_DIR / filename
        )

        download_file(
            session=session,
            url=download_url,
            destination=destination,
            expected_size=expected_size,
        )

    print()
    print(
        "CITIC-MTMC dataset download complete."
    )
    print(
        f"Dataset directory: {CITIC_MTMC_DIR}"
    )


if __name__ == "__main__":
    download_dataset()
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_catalog_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("tiny_catalog")
    csv_path = data_dir / "main_data.csv"
    csv_path.write_text(
        "movie_title,comb\n"
        "avatar,epic science fiction world\n"
        "titanic,romance ship tragedy drama\n"
        "inception,dream heist thriller mind\n"
        "interstellar,space exploration family science\n"
        "gladiator,roman empire arena war\n"
        "arrival,first contact linguistics science\n"
        "up,animated adventure family friendship\n"
        "coco,music family memory animation\n"
        "joker,psychological crime drama\n"
        "matrix,virtual reality cyberpunk action\n",
        encoding="utf8",
    )
    return csv_path


@pytest.fixture(scope="session")
def full_catalog_csv() -> Path:
    return ROOT / "main_data.csv"

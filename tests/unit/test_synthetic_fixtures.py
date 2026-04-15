from pathlib import Path

from PIL import Image

from xai_demo_suite.data.synthetic import make_striped_fixture


def test_make_striped_fixture_writes_deterministic_image(tmp_path: Path) -> None:
    sample = make_striped_fixture(tmp_path)

    assert sample.image_path.exists()
    assert sample.label == "central_stripe"
    assert sample.region.area == 48

    with Image.open(sample.image_path) as image:
        assert image.size == (32, 32)
        assert image.mode == "RGB"
        assert image.getpixel((15, 11)) == (220, 232, 96)

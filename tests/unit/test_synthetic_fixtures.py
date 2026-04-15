from pathlib import Path

from PIL import Image

from xai_demo_suite.data.synthetic import (
    generate_industrial_shortcut_dataset,
    generate_slot_board_dataset,
    make_striped_fixture,
)


def test_make_striped_fixture_writes_deterministic_image(tmp_path: Path) -> None:
    sample = make_striped_fixture(tmp_path)

    assert sample.image_path.exists()
    assert sample.label == "central_stripe"
    assert sample.region.area == 48

    with Image.open(sample.image_path) as image:
        assert image.size == (32, 32)
        assert image.mode == "RGB"
        assert image.getpixel((15, 11)) == (220, 232, 96)


def test_generate_slot_board_dataset_writes_nominal_and_limit_cases(tmp_path: Path) -> None:
    train_samples, eval_samples = generate_slot_board_dataset(tmp_path)

    assert [sample.sample_id for sample in train_samples] == ["normal_000", "normal_001"]
    assert [sample.sample_id for sample in eval_samples] == [
        "missing_one",
        "missing_three",
        "fine_scratch",
        "logic_swap",
    ]
    assert eval_samples[0].missing_count == 1
    assert eval_samples[1].missing_count == 3
    assert eval_samples[2].severity_area > 0
    assert eval_samples[3].label == "logic_swap"

    with Image.open(eval_samples[0].image_path) as image:
        assert image.size == (320, 224)
        assert image.mode == "RGB"
    with Image.open(eval_samples[0].mask_path) as mask:
        assert mask.size == (320, 224)
        assert mask.mode == "L"


def test_generate_industrial_shortcut_dataset_writes_swapped_cases(tmp_path: Path) -> None:
    train_samples, test_samples = generate_industrial_shortcut_dataset(tmp_path)

    assert len(train_samples) == 4
    assert {sample.stamp for sample in train_samples} == {"blue", "red"}
    by_id = {sample.sample_id: sample for sample in test_samples}
    assert by_id["test_normal_swapped_stamp"].label == "normal"
    assert by_id["test_normal_swapped_stamp"].stamp == "red"
    assert by_id["test_defect_swapped_stamp"].label == "defect"
    assert by_id["test_defect_swapped_stamp"].stamp == "blue"

    with Image.open(by_id["test_normal_swapped_stamp"].image_path) as image:
        assert image.size == (128, 128)
        assert image.getpixel((10, 10)) == (216, 70, 64)

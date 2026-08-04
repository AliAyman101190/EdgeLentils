import pytest

from perception.kitti_labels import (
    CLASS_MAP,
    CLASS_TO_ID,
    DROPPED_CLASSES,
    YOLO_CLASSES,
    box_to_yolo,
    convert_objects,
    parse_label_line,
)

# One real line per raw class, taken from KITTI training labels.
SAMPLE_LINES = {
    "Car": "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57",
    "Van": "Van 0.00 0 -1.53 599.41 156.40 629.75 189.25 2.19 1.79 4.89 5.26 1.40 68.62 -1.55",
    "Truck": "Truck 0.00 0 1.60 599.41 156.40 629.75 189.25 3.07 2.63 11.17 15.55 1.63 45.75 1.55",
    "Pedestrian": "Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01",
    "Person_sitting": "Person_sitting 0.00 0 0.14 599.41 156.40 629.75 189.25 1.27 0.59 0.72 4.06 1.66 22.75 0.14",
    "Cyclist": "Cyclist 0.00 0 -2.72 505.87 179.28 528.62 224.05 1.73 0.63 1.68 -3.34 1.71 26.29 -1.62",
    "Tram": "Tram 0.00 0 -1.56 116.59 148.85 258.31 229.90 3.53 2.68 10.44 -8.30 1.68 34.09 -1.55",
    "Misc": "Misc 0.00 0 -1.56 116.59 148.85 258.31 229.90 1.53 1.68 3.44 -8.30 1.68 34.09 -1.55",
    "DontCare": "DontCare -1 -1 -10 219.31 188.49 245.50 218.56 -1 -1 -1 -1000 -1000 -1000 -10",
}


def test_class_map_covers_expected_yolo_classes():
    assert set(YOLO_CLASSES) == {"Car", "Pedestrian", "Cyclist"}
    assert set(CLASS_MAP.values()) == set(YOLO_CLASSES)
    assert CLASS_MAP["Van"] == "Car"
    assert CLASS_MAP["Truck"] == "Car"
    assert CLASS_MAP["Person_sitting"] == "Pedestrian"


def test_dropped_classes_are_disjoint_from_class_map():
    assert DROPPED_CLASSES.isdisjoint(CLASS_MAP)


@pytest.mark.parametrize("raw_class,line", SAMPLE_LINES.items())
def test_parse_label_line_roundtrips_class(raw_class, line):
    obj = parse_label_line(line)
    assert obj.raw_class == raw_class
    assert len(obj.bbox) == 4
    assert len(obj.dimensions) == 3
    assert len(obj.location) == 3


def test_parse_label_line_wrong_field_count_raises():
    with pytest.raises(ValueError, match="15 fields"):
        parse_label_line("Car 0.00 0 1.85 387.63 181.54 423.81 203.12")


def test_box_to_yolo_roundtrip():
    bbox = (100.0, 50.0, 300.0, 250.0)
    img_w, img_h = 1242, 375
    cx, cy, w, h = box_to_yolo(bbox, img_w, img_h)

    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0

    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    assert (x1, y1, x2, y2) == pytest.approx(bbox, abs=1e-6)


def test_box_to_yolo_out_of_bounds_raises():
    with pytest.raises(ValueError, match="out of bounds"):
        box_to_yolo((100.0, 50.0, 1300.0, 250.0), img_w=1242, img_h=375)


def test_box_to_yolo_degenerate_raises():
    with pytest.raises(ValueError, match="degenerate"):
        box_to_yolo((300.0, 50.0, 300.0, 250.0), img_w=1242, img_h=375)


def test_convert_objects_filters_and_remaps():
    objects = [parse_label_line(line) for line in SAMPLE_LINES.values()]
    rows = convert_objects(objects, img_w=1242, img_h=375)

    # 9 raw lines -> 6 kept (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist),
    # Tram/Misc/DontCare dropped.
    assert len(rows) == 6
    class_ids = {row[0] for row in rows}
    assert class_ids == {CLASS_TO_ID["Car"], CLASS_TO_ID["Pedestrian"], CLASS_TO_ID["Cyclist"]}
    for _, cx, cy, w, h in rows:
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0

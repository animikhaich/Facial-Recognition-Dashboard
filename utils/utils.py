from string import punctuation


def fix_coordinates(box: list, width: int, height: int):
    x1, y1, w, h = box
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(w, width) + x1
    y2 = min(h, height) + y1
    return x1, y1, x2, y2


def rem_punctuation(text: str):
    for c in punctuation.replace(".", " "):
        text = text.replace(c, "_")
    return text.lower()

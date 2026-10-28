"""Native normalized-1000 quadrilaterals only; no crop, type inference or correction."""
import re

LOC = re.compile(r"<\|LOC_(\d+)\|>")
TAGGED = re.compile(r"<\|TEXT_START\|>(.*?)<\|TEXT_END\|>\s*<\|LOC_START\|>(.*?)<\|LOC_END\|>", re.S)
BOUNDARY = ("</s>", "<s>", "<unk>")


def display_text(native):
    # Preserve LOC, OTSL, LaTeX and content. Only validated boundary tokens are removed.
    value = native
    for token in BOUNDARY:
        if value.endswith(token):
            value = value[:-len(token)]
    return value


def parse_spotting(native, width, height):
    source = display_text(native)
    regions, errors = [], []

    def add(text, matches):
        values = [int(m.group(1)) for m in matches]
        if len(values) != 8 or any(v > 1000 for v in values):
            errors.append("invalid_coordinate_group")
            return
        points = [[values[i], values[i+1]] for i in range(0, 8, 2)]
        area = abs(sum(points[i][0] * points[(i+1) % 4][1] -
                       points[(i+1) % 4][0] * points[i][1] for i in range(4)))
        if not text.strip() or area == 0:
            errors.append("empty_text_or_zero_area")
            return
        regions.append({"region_id": len(regions), "text": text.strip(),
                        "native_polygon": points,
                        "polygon_input_px": [[x / 1000 * width, y / 1000 * height] for x, y in points]})

    tagged = list(TAGGED.finditer(source))
    if tagged:
        cursor = 0
        for match in tagged:
            if source[cursor:match.start()].strip():
                errors.append("unparsed_content")
            add(match.group(1), list(LOC.finditer(match.group(2))))
            # Disallow extra content other than coordinate tokens/whitespace inside LOC block.
            if LOC.sub("", match.group(2)).strip():
                errors.append("invalid_coordinate_syntax")
            cursor = match.end()
        if source[cursor:].strip():
            errors.append("unparsed_content")
    else:
        matches = list(LOC.finditer(source))
        cursor = 0
        for index in range(0, len(matches), 8):
            group = matches[index:index+8]
            if len(group) != 8:
                errors.append("incomplete_coordinate_group")
                break
            for a, b in zip(group, group[1:]):
                if source[a.end():b.start()].strip():
                    errors.append("interleaved_coordinate_content")
            add(source[cursor:group[0].start()], group)
            cursor = group[-1].end()
        if source[cursor:].strip():
            errors.append("unparsed_content")
    if not regions and not errors:
        errors.append("empty_output_not_verified_no_regions")
    return {"parse_status": "ok" if not errors else "invalid_or_incomplete",
            "coordinate_system": "normalized_1000_quadrilateral",
            "regions": regions, "parse_errors": errors}

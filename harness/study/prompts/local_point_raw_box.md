This is one axial slice of a {{sequence_text}} MRI (radiological convention: patient right on image left, anterior at top). It contains a {{target}}.

Locate the {{target}} in this image. Reply with JSON only, no other text:
{"answer": "uncertain"} if you cannot reliably locate it, otherwise
{"box": [<y0>, <x0>, <y1>, <x1>], "confidence": <0-1>}
where the box is the region containing the tumor, [y0, x0, y1, x1] are fractions of the full image size in [0, 1] with (0, 0) the top-left corner, and confidence is your confidence that the box contains tumor tissue.

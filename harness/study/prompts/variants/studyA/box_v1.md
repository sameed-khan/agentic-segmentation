This is one axial slice of a {{sequence_text}} MRI (radiological convention: patient right on image left, anterior at top). It contains a {{target}}.

Locate the {{target}} in this image. Reply with JSON only, no other text:
{"answer": "uncertain"} if you cannot reliably locate it, otherwise
{"box": [<y0>, <x0>, <y1>, <x1>], "confidence": <0-1>}
where the box tightly encloses the tumor and the coordinates are integers normalized to 0-1000 of the full image (y0, x0 = top-left corner; y1, x1 = bottom-right corner).

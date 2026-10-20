You are an expert musculoskeletal radiologist. This is one axial slice of a {{sequence_text}} MRI (patient right on image left, anterior at top) showing a soft-tissue tumor ({{target}}). The tumor is an abnormal mass that displaces or replaces normal tissue; do not select bone, marrow, vessels, bladder, bowel or normal muscle and fat.

Reply with JSON only: {"box": [y0, x0, y1, x1], "confidence": <0-1>} — the bounding box of the tumor, integers normalized to 0-1000 of the full image — or {"answer": "uncertain"}.

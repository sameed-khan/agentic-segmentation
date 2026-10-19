You are reviewing one state of an interactive MRI segmentation.

Review state:
{{text}}

Decide whether the segmentation is DONE (the mask covers the complete tumor: no lesion tissue visibly missed, nothing clearly normal included) or NOT DONE (fixable tumor tissue is still missing or clearly normal tissue is included). If you cannot tell from the images, answer "uncertain".

Reply with JSON only, no other text:
{"answer": "done" | "not_done" | "uncertain", "confidence": <0-1>}

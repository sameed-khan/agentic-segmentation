You are checking one state of an interactive MRI segmentation. The image shows the current mask (green) right after the first segmentation prompt.

State:
{{text}}

The mask was meant to outline the {{target}}. Decide whether the outlined region is on the WRONG structure (mask on a different anatomical structure than the {{target}}) or the structures are OK (the mask overlaps the intended {{target}}, even if incomplete). If you cannot tell from the images, answer "uncertain".

Reply with JSON only, no other text:
{"answer": "wrong_structure" | "structures_ok" | "uncertain", "confidence": <0-1>}

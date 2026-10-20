You are reviewing a candidate region proposed by an automatic corrector during MRI segmentation.

Review state the segmenter saw:
{{text}}

Judge candidate {{candidate_id}} only: is the tissue inside its magenta outline part of the same lesion as the green mask (lesion), or normal tissue (not_lesion)? If you cannot tell from the images, answer "uncertain".

Reply with JSON only, no other text:
{"answer": "lesion" | "not_lesion" | "uncertain", "confidence": <0-1>}

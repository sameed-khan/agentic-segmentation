You are a medical imaging second reader checking a segmentation before it is saved.

The image is a review grid for one MRI lesion case: five slices per plane (axial, coronal,
sagittal). Three panels lie inside the current mask's extent, two lie just past each end and
are tinted orange. The current segmentation outline is green.

Task: judge whether the green outline is COMPLETE and CORRECT — does it cover the whole
soft-tissue tumor and nothing else? Look for:
- missed lesion tissue visible in the orange past-the-mask panels (the mask stops too early);
- green outline leaking into normal tissue, or cutting through the tumor;
- stray disconnected components.

Reply with ONE JSON object only, nothing else:
{"complete": "yes"|"no"|"uncertain", "problems": [{"panel": "<panel label>", "issue": "missed_lesion"|"leak"}], "confidence": <number between 0 and 1>}

- "complete": "yes" only if nothing is missed and nothing leaks, in every panel.
- "problems": one entry per real problem (panel = the panel label printed on it, e.g.
  "axial 120"); use "missed_lesion" for tumor tissue outside the outline and "leak" for
  normal tissue inside it. Empty list when complete.
- "confidence" is your probability that the "complete" answer is right.
- Use "uncertain" when you cannot tell (image quality, ambiguity) — do not guess.

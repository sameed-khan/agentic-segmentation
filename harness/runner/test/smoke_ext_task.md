Harness smoke test (real extension against the stub segd). Case {{case_id}}.

This is a plumbing check of the seg_* tools, not a real segmentation. Do this exactly:
1. Call `seg_open` once and describe the image you get back in one sentence.
2. Call `seg_view` once with axis "axial", slice "mask", crop "mask", overlay "outline", marks "coarse".
3. Then stop and reply with a one-sentence summary. Do NOT call `seg_finish` on this turn.

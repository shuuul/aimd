Quarterly Report

The native OCR engine converts scanned pages into Markdown for language model workflows. It runs a SAM vision encoder, a CLIP tower, and a mixture-of-experts decoder on the CPU.

Key results

1. Latency is measured per page.

2. Output is normalized Markdown.

3. Repetition loops are blocked by an n-gram guard with a window of 128.

Contact: engineering team, building 7.
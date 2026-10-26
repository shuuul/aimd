# Preparing Context for Language Models

Context engineering is the practice of deciding what information a language model sees before it answers. A good context window contains the task, the relevant documents, and the constraints, but nothing that distracts the model. Teams that build agents spend most of their time on this step.

## Converting sources

Web pages, recordings, and scanned papers all have to become clean text first. Web pages carry navigation menus, advertisements, and cookie banners that must be removed. Recordings need speech recognition, and every minute of audio has to be transcribed before it can be searched. Scanned documents need optical character recognition that keeps tables, headings, and reading order intact.

## Chunking

Once the text is clean, it is split into chunks. Chunks follow the document structure where possible: a chunk should start at a heading and end before the next heading of the same level. Very long sections are split at paragraph boundaries so that no chunk exceeds the size limit. Each chunk keeps a reference to its title so that a model can cite where the information came from.

## Choosing what to include

A retrieval step scores every chunk against the question and keeps the best ones. The remaining budget is spent on instructions and examples. When the context is too small, important facts are missing; when it is too large, the model loses focus and the cost grows. Measuring both answer quality and latency is the only reliable way to find the balance.

## Running locally

Local inference on Apple Silicon keeps private material on the device and removes the network from the critical path. Quantized checkpoints make the models small enough to load in a few seconds, and custom kernels make decoding fast enough that long recordings finish in seconds instead of minutes. The remaining work is mostly careful measurement: every change must keep the output identical while making it faster.

## Conclusion

Fast and accurate conversion makes the rest of the pipeline simple. When conversion is slow, every experiment waits, and teams stop trying new ideas. Investing in the conversion step pays back across every project that depends on it.

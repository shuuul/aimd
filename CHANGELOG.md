# Changelog

## [0.15.0](https://github.com/shuuul/aimd/compare/v0.14.1...v0.15.0) (2026-07-05)


### Features

* add --cookies-file option for authenticated URL access ([e87cbcd](https://github.com/shuuul/aimd/commit/e87cbcd3204bad1d7799bbbda27a51d20d7ab0d5))
* add .mp4a transcoding support and propagate temp_dir to ffmpeg conversion ([c293980](https://github.com/shuuul/aimd/commit/c2939801837852a17d6a2dbfe07cf88817a72724))
* add configurable temporary directory support\n\n- Support AIMD_TEMP_DIR env var and --temp-dir CLI option\n- Propagate temp_dir through application layers to infrastructure\n- Ensure sandbox compatibility by avoiding system /tmp when configured\n- Update documentation in README.md and AGENTS.md files\n- Bump version to 0.4.3 ([16a69f9](https://github.com/shuuul/aimd/commit/16a69f921dd2a1f9c0914b17c4c17527239aa79c))
* add FastAPI service, engine preflight, and refresh AGENTS docs ([b19884e](https://github.com/shuuul/aimd/commit/b19884ec6477456455e5b0ff3e350760c94908e6))
* add platform information to processing results ([d1af81f](https://github.com/shuuul/aimd/commit/d1af81fb499658fa0b741712bbfb8ebde85ddbfc))
* add Qwen3-ASR engine for Linux and rename cuda to whisper ([9d0d638](https://github.com/shuuul/aimd/commit/9d0d6383fd32e4261da5005bb93977434f568ba6))
* add traditional Chinese language support ([5ccd5b6](https://github.com/shuuul/aimd/commit/5ccd5b62523cc73364ad2df3e00e79305cb111b9))
* **asr:** add local Qwen3-ASR Transformers backend ([c2a99da](https://github.com/shuuul/aimd/commit/c2a99da4bd5f4eabe585c6430f967a4200a27481))
* **asr:** add segmenting and 8-bit fallback for repetition loops ([41d799a](https://github.com/shuuul/aimd/commit/41d799a21e7fbeaedd29fef766c7fd17fdafa3b7))
* enhance transcription engine support and update documentation ([dc32e73](https://github.com/shuuul/aimd/commit/dc32e7342f6a38deb30b5bd3d3f5c04755d85a46))
* introduce raw transcript option for subtitle formatting ([6750345](https://github.com/shuuul/aimd/commit/6750345fe57ba84c079c6ef94bc3332a483dde89))
* migrate from mlx-whisper to mlx-audio with Qwen3-ASR ([b474763](https://github.com/shuuul/aimd/commit/b4747631a7c425c61b595fa4eac5a5f2a6b23f7d))
* **ocr:** add transformers model adapters ([16a2bb7](https://github.com/shuuul/aimd/commit/16a2bb72c2bded0f604d75fb23490cc8316bd8fc))
* **ocr:** add transformers OCR backend ([356cb9b](https://github.com/shuuul/aimd/commit/356cb9b400a131c5a4df071bd5409ab7a1f84d7c))
* probe all browsers for cookies and add more MLX Qwen3-ASR variants ([3c0e659](https://github.com/shuuul/aimd/commit/3c0e659dba90fd86264f212c29fa2e6a83922178))
* remove whisper backend and bump version to 0.7.0 ([372c32d](https://github.com/shuuul/aimd/commit/372c32d5d9dea66417affa48adae9d9eda9b9543))
* rename CLAUDE.md to AGENTS.md and add --save-original option ([3755051](https://github.com/shuuul/aimd/commit/3755051293cef5b08fbac2e9b7db6ba32388e199))
* update default models and enhance FunASR support ([60e8e94](https://github.com/shuuul/aimd/commit/60e8e948356c6cd9b737074315cf1f2d3977dc51))


### Bug Fixes

* avoid aimd-api lazy import recursion ([b5b6c08](https://github.com/shuuul/aimd/commit/b5b6c087e02fca888a6590fbe674e9564e6c07e0))
* clean up URL transcript output and remove stale docs references ([f7d675e](https://github.com/shuuul/aimd/commit/f7d675e357d6882143e41687140e0b98115cc352))
* fallback on FunASR no-speech tokens ([87b5255](https://github.com/shuuul/aimd/commit/87b5255b1a8d3fa4e382fc0855b57b5bf8395cdd))
* Fun-ASR-Nano control-token safety and Bilibili 412 cookies error handling ([bb3830d](https://github.com/shuuul/aimd/commit/bb3830d6808bb8519ebdbd9b63a6d8efd65802be))
* publish core package as aimd-cli ([e13de3c](https://github.com/shuuul/aimd/commit/e13de3c5d241504eaae6e3c7d3bdfa7388085560))
* remove flash-attn dependency for non-linux platforms ([1761651](https://github.com/shuuul/aimd/commit/1761651d9ad6b97d17a25ea9547803d38965c465))
* truncate audio filenames by byte length to avoid OS limits with CJK titles ([e2b8fd4](https://github.com/shuuul/aimd/commit/e2b8fd4f2908884512f8f94107408cc7be65731e))
* **url:** preserve complete YouTube transcripts ([a06829c](https://github.com/shuuul/aimd/commit/a06829c9a336680fe4b4a9308bbfec114a3ef6b2))
* use package-qualified API app target ([575550b](https://github.com/shuuul/aimd/commit/575550b6a792a9ce0448e329523294028a2d5eaa))
* use transformers backend for qwen asr ([baa853e](https://github.com/shuuul/aimd/commit/baa853e3e183343dfc491ea696bbb270fb6d35e6))


### Documentation

* bump version to 0.6.5 ([ec1d94d](https://github.com/shuuul/aimd/commit/ec1d94daaaa7dbc775670a859a75916baf7f7d44))
* refresh README badges and install notes ([a1f04c8](https://github.com/shuuul/aimd/commit/a1f04c807966d1628b8f4a87be2173ed22a566d1))
* require prek before commits ([17006f8](https://github.com/shuuul/aimd/commit/17006f8598075c46340b967c2a5e25e2f05ad60a))
* **skill:** add aimd agent skill ([1285998](https://github.com/shuuul/aimd/commit/1285998efcd37e81a4f00bd2bc1ddf543a97ef14))
* **skill:** focus aimd skill on markdown prep ([fd26732](https://github.com/shuuul/aimd/commit/fd2673213cea7ddd0c209a198b56db289cc531c4))
* update AGENTS.md to v0.8.1 and sync version/torch notes ([0dba03a](https://github.com/shuuul/aimd/commit/0dba03a6794b28968d75e3c41b1011a17d1d068f))
* update backend and URL guidance ([28262e9](https://github.com/shuuul/aimd/commit/28262e9ead207bda80c143a5b76a1328a3c0d2db))
* update README banner assets ([ed97423](https://github.com/shuuul/aimd/commit/ed97423abd07cc4dbe2a16904f9c5b9b73d8a822))
* update supported models and architecture ([93c7abf](https://github.com/shuuul/aimd/commit/93c7abf2d056cdb8429bb0ac80c2481ad775d38f))

## [0.14.1](https://github.com/shuuul/aimd/compare/v0.13.1...v0.14.1) (2026-07-05)


### Features

* **asr:** add segmenting and 8-bit fallback for repetition loops


### Bug Fixes

* **deps:** keep MLX ASR compatible with Transformers 5.12
* **logging:** support logly 0.2 logger configuration


### Maintenance

* update Python dependencies and pre-commit hooks

## [0.13.1](https://github.com/shuuul/aimd/compare/v0.13.0...v0.13.1) (2026-06-24)


### Bug Fixes

* **url:** preserve complete YouTube transcripts ([a06829c](https://github.com/shuuul/aimd/commit/a06829c9a336680fe4b4a9308bbfec114a3ef6b2))

## [0.13.0](https://github.com/shuuul/aimd/compare/v0.12.0...v0.13.0) (2026-06-23)


### Features

* **asr:** add local Qwen3-ASR Transformers backend ([c2a99da](https://github.com/shuuul/aimd/commit/c2a99da4bd5f4eabe585c6430f967a4200a27481))


### Documentation

* require prek before commits ([17006f8](https://github.com/shuuul/aimd/commit/17006f8598075c46340b967c2a5e25e2f05ad60a))

## Changelog

All notable changes to this project will be documented in this file.

This project uses [Conventional Commits](https://www.conventionalcommits.org/) and release-please to generate release notes.

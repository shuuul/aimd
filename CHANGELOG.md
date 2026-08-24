# Changelog

## [0.23.1](https://github.com/shuuul/aimd/compare/v0.23.0...v0.23.1) (2026-08-24)


### Bug Fixes

* **url:** download native YouTube captions and format transcripts ([6ac9038](https://github.com/shuuul/aimd/commit/6ac903868caa2cd9e4637deb42c37c00285ab1a9))

## [0.23.0](https://github.com/shuuul/aimd/compare/v0.22.0...v0.23.0) (2026-08-24)


### Features

* **remote:** add OpenAI-compatible ASR and OCR backends ([a81dc82](https://github.com/shuuul/aimd/commit/a81dc82e11caf858d4fb741b68583fd9ca75e26b))


### Bug Fixes

* **ocr:** normalize Unlimited-OCR markdown ([b353f72](https://github.com/shuuul/aimd/commit/b353f722db65da3ab14b3f6316769059fee09095))

## [0.22.0](https://github.com/shuuul/aimd/compare/v0.21.0...v0.22.0) (2026-08-17)


### Features

* **api:** accept uploaded blobs as a job source ([2456c43](https://github.com/shuuul/aimd/commit/2456c43da0e26b2b3471036eec6c0324df6306bc))

## [0.21.0](https://github.com/shuuul/aimd/compare/v0.20.0...v0.21.0) (2026-08-16)


### ⚠ BREAKING CHANGES

* minimum Python version is now 3.11

### Features

* support Python 3.13 ([96d93c6](https://github.com/shuuul/aimd/commit/96d93c64f776acadf9b1aebc96a509e364b39361))


### Documentation

* sync PyPI page with README and use absolute links ([bf54ac8](https://github.com/shuuul/aimd/commit/bf54ac84f7aa20ab769fb935c0c41ce6b9e6a51c))


### Miscellaneous Chores

* drop Python 3.10 support ([0662083](https://github.com/shuuul/aimd/commit/0662083ea7cb258c930fc862f6b4b9deea0a43d2))

## [0.20.0](https://github.com/shuuul/aimd/compare/v0.19.0...v0.20.0) (2026-08-16)


### Features

* **doc:** convert text-layer PDFs with pdf-inspector ([6d543e0](https://github.com/shuuul/aimd/commit/6d543e08acd47e6be68ee16e72dd02560ede78b6))


### Bug Fixes

* **core:** route encoding-broken PDFs to OCR ([fe22817](https://github.com/shuuul/aimd/commit/fe22817a842d759b68f00c2b26c2f53fa32e4c42))

## [0.19.0](https://github.com/shuuul/aimd/compare/v0.18.3...v0.19.0) (2026-08-16)


### Features

* add job-based API, sidecar contract, and ASR context biasing ([abbbf62](https://github.com/shuuul/aimd/commit/abbbf62040d7902aabc42e535d9ccb69c3b25869))

## [0.18.3](https://github.com/shuuul/aimd/compare/v0.18.2...v0.18.3) (2026-08-15)


### Bug Fixes

* **api:** expose lossless markdown artifacts ([e85fe05](https://github.com/shuuul/aimd/commit/e85fe05a4f7955a25da0a7a675ffe230b87aac39))

## [0.18.2](https://github.com/shuuul/aimd/compare/v0.18.1...v0.18.2) (2026-08-07)


### Bug Fixes

* **url:** enable node JS runtime for YouTube challenges ([e1f4d80](https://github.com/shuuul/aimd/commit/e1f4d80282be372c49b503ebfb0d0cf562aafe69))

## [0.18.1](https://github.com/shuuul/aimd/compare/v0.18.0...v0.18.1) (2026-08-05)


### Bug Fixes

* **url:** prefer original subtitle tracks over translations ([4276c79](https://github.com/shuuul/aimd/commit/4276c79bd08e1dcf8d12f340414a95219120b3a7))

## [0.18.0](https://github.com/shuuul/aimd/compare/v0.17.0...v0.18.0) (2026-08-04)


### Features

* **url:** infer subtitle language from title and description ([25e75f1](https://github.com/shuuul/aimd/commit/25e75f18d5a0c59783fc495bc480d4278f2f82a9))

## [0.17.0](https://github.com/shuuul/aimd/compare/v0.16.0...v0.17.0) (2026-08-01)


### ⚠ BREAKING CHANGES

* **models:** Remove GOT-OCR, generic OCR model loading, and arbitrary unsupported model IDs.

### Features

* **models:** add precision-aware model selection ([5f26d17](https://github.com/shuuul/aimd/commit/5f26d179f8b5ffad36172c7871c541076f9603c2))


### Bug Fixes

* **ocr:** drop unnecessary trust_remote_code for GLM-OCR ([128e1c4](https://github.com/shuuul/aimd/commit/128e1c4ccf8b849bb5e61de7df6aeb641aef3b43))

## [0.16.0](https://github.com/shuuul/aimd/compare/v0.15.0...v0.16.0) (2026-07-31)


### Features

* **core:** harden processing boundaries ([f23fbb2](https://github.com/shuuul/aimd/commit/f23fbb257673aec4e591c1e1b73f60a853febd46))
* **ocr:** default to Unlimited-OCR with mlx-vlm support ([a5540e7](https://github.com/shuuul/aimd/commit/a5540e790f424e22650309a5e34c9f121d9ef75a))


### Bug Fixes

* **interfaces:** unify routing and empty output errors ([3e36c4b](https://github.com/shuuul/aimd/commit/3e36c4b7f3aa47a5468f0f050df25b6671dadc00))
* **release:** accept mcp 2.0 MCPServer in smoke install ([583086f](https://github.com/shuuul/aimd/commit/583086fbea6a6261466d1a0a589ebc923312ee45))

## [0.15.0](https://github.com/shuuul/aimd/compare/v0.14.2...v0.15.0) (2026-07-31)


### Features

* **asr:** migrate to native Transformers Qwen3-ASR-hf ([1fc2c2d](https://github.com/shuuul/aimd/commit/1fc2c2df36447aeefec58d92ee1d68cbb60dafb7))

## [0.14.2](https://github.com/shuuul/aimd/compare/v0.14.1...v0.14.2) (2026-07-05)


### Bug Fixes

* **release:** publish patch via release-please ([0b0a6f3](https://github.com/shuuul/aimd/commit/0b0a6f358209a84c41cb4818ad95ec17802f8c05))

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

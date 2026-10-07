# Return HTTP 400 for oversized image inputs

## Motivation

An image larger than Pillow's 178,956,970-pixel decompression-bomb limit raises
`PIL.Image.DecompressionBombError` during multimodal request preprocessing. This
exception is not an `OSError` or `ValueError`, so it currently reaches the
generic OpenAI request handler and is reported as HTTP 500 even though the
request payload is invalid.

## Modifications

- Convert `PIL.Image.DecompressionBombError` from `Image.open()` into
  `ValueError("image too large, exceeds upper limit.")`.
- Reuse the existing `ValueError` handling in the OpenAI serving layer, which
  returns HTTP 400 with the standard error response.
- Add a regression test without allocating an oversized image in the test
  process.

## Accuracy Tests

Not applicable. This change only affects request validation and error mapping.

## Speed Tests and Profiling

Not applicable. The normal image decode path is unchanged; the added exception
mapping runs only when Pillow rejects an oversized image.

## Validation

- `git diff --check` — passed.
- `python3.10 -m py_compile python/sglang/srt/utils/common.py test/registered/unit/multimodal/test_base_processor_image_decode.py` — passed.
- The focused unit test could not be executed in this checkout because the
  available Python environment does not have the SGLang test dependencies
  (including NumPy) installed.

## Checklist

- [ ] Format code with pre-commit.
- [x] Add a focused regression unit test.
- [ ] Update documentation (not needed for this request-level behavior fix).
- [x] Accuracy and speed benchmarks are not applicable.
- [x] Follow the existing SGLang code style.

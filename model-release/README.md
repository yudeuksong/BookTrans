# BookTrans model packages

Model packages for **BookTrans**, an offline English → Korean PDF translator for Windows.
BookTrans does not ship a translation model; it installs one of these packages only after
the user agrees, and only if the file's SHA-256 matches the value pinned in BookTrans.

## quickmt-en-ko

| | |
|---|---|
| File | `booktrans-model-quickmt-en-ko.zip` (Releases, tag `quickmt-en-ko-a32fa39`) |
| SHA-256 | `2ef78d406cddf53fa621db0a5ec66a6cb8745932ec25a6d976fe94df6efda7cd` |
| Model | [quickmt/quickmt-en-ko](https://huggingface.co/quickmt/quickmt-en-ko), revision `a32fa395ebc7b722a0c021e213ca5fec4065c5df` |
| Author | the quickmt project |
| Model license | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Changes | None to the model. Files copied as published and repackaged; training checkpoint omitted. |

The package is rebuilt from the published model by `pack_quickmt_en_ko.py`
(`.github/workflows/model-release.yml`) and released only when its SHA-256 matches the pinned value:

```
pip install huggingface-hub==2.2.0
python model-release/pack_quickmt_en_ko.py work out/booktrans-model-quickmt-en-ko.zip
```

The scripts here are MIT licensed (`../LICENSE-MIT`). The model files in the
releases are licensed by their authors under CC BY 4.0, as stated above.

---

BookTrans용 번역 모델 파일입니다. 모델(quickmt-en-ko)은 quickmt 프로젝트가 만든 것이며
CC BY 4.0 라이선스입니다. 모델 자체는 바꾸지 않고 BookTrans가 읽는 형태로 묶기만 했습니다.

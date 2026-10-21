"""Appropriate Legal Notices shown by the GUI and CLI (AGPL-3.0 section 0 and 5(d))."""

from booktrans import __version__

SOURCE_NOTE = (
    "대응 소스: 이 설치 파일과 같은 배포 위치에 BookTrans-Source 및 "
    "BookTrans-ThirdParty-Source ZIP이 함께 제공됩니다."
)

ABOUT_TEXT = f"""BookTrans {__version__} — 영어 PDF 오프라인 한국어 번역기

Copyright (c) 2026 BookTrans contributors

이 프로그램(설치 파일로 배포되는 결합 저작물)은 GNU Affero General Public License
버전 3(AGPL-3.0-only)에 따라 배포됩니다. PyMuPDF/MuPDF를 포함하기 때문입니다.
BookTrans 자체 소스 코드는 MIT 라이선스로도 제공됩니다(LICENSE-MIT).

이 프로그램은 어떠한 보증도 없이 제공됩니다. 자세한 내용은 설치 폴더의 LICENSE,
LICENSE-MIT, THIRD_PARTY_NOTICES.txt 와 licenses 폴더를 참고하세요.

{SOURCE_NOTE}

번역 모델은 BookTrans에 포함되지 않으며, 사용자가 따로 설치합니다.
모델: Meta AI M2M100 418M (facebook/m2m100_418M). 모델 카드에 MIT 라이선스로
명시되어 있으며, 모델 파일 안의 LICENSE와 MODEL_CARD.md에 원문이 들어 있습니다.
"""

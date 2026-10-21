"""The review must catch the failure types users reported (detectors, not translations)."""

import pytest

from booktrans.quality import review, source_numbers


def codes(src, tgt):
    return [i.code for i in review(src, tgt)]


@pytest.mark.parametrize(
    "src,bad,good,code",
    [
        ("You must be born again", "당신은 AGAIN을 찾아야", "당신은 다시 태어나야 합니다.", "english-left"),
        ("Contents", "관련 기사", "목차", "web-phrase"),
        ("Problem", "계정 관리", "문제", "web-phrase"),
        ("CONTENTS", "(주)에이치", "목차", "web-phrase"),
        ("PROBLEM", "(주)", "문제", "web-phrase"),
        ("The eighteenth century was an age of reason.", "8 세기는 이성의 시대였다.", "18세기는 이성의 시대였다.", "number"),
        ("Not only can your heart be changed, but your whole life can be renewed.",
         "당신의 마음은 바뀔 수 없고, 당신의 삶 전체가 새롭게 될 수 있습니다.",
         "당신의 마음이 바뀔 수 있을 뿐만 아니라 삶 전체가 새로워질 수 있습니다.", "negation-added"),
        ("He did not go to the market.", "그는 시장에 갔다.", "그는 시장에 가지 않았다.", "negation-lost"),
        ("In 1750 there were twenty-one churches.", "1750년에는 교회가 있었다.", "1750년에는 21개의 교회가 있었다.", "number"),
        ("The committee will publish its final report next spring and discuss it in public.",
         "위원회.", "위원회는 내년 봄에 최종 보고서를 발표하고 공개적으로 논의할 것이다.", "omission"),
    ],
)
def test_reported_failures_are_detected(src, bad, good, code):
    assert code in codes(src, bad)
    assert codes(src, good) == []


def test_number_words_and_ambiguous_words():
    assert source_numbers("the nineteenth chapter, twenty-one houses, 1,500 people") == {19, 21, 1500}
    assert source_numbers("No one knew it at first; give me a second chance.") == set()
    assert codes("At first, the plan seemed simple.", "처음에는 계획이 간단해 보였다.") == []
    assert codes("He found three keys.", "그는 열쇠 세 개를 찾았다.") == []


def test_proper_nouns_may_stay_in_english():
    assert codes("Yesterday Martin Luther wrote to Erasmus.", "어제 Martin Luther가 Erasmus에게 편지를 썼다.") == []


# Regression cases from the 0.1.2 evidence review (false passes and false positives).
@pytest.mark.parametrize(
    "src,tgt,expected",
    [
        # false pass: an adverb is not a negation
        ("She never forgot the promise she had made.", "그녀는 결코 약속을 잊었다.", "negation-lost"),
        # false pass: "A뿐만 아니라" with A missing
        ("It is not only a duty but also a joy.", "그것은뿐만 아니라 기쁨입니다.", "contrast-malformed"),
        # real number error: forty thousand -> 4천
        ("In 1848 the city had nearly forty thousand inhabitants.", "1848년에 도시에는 거의 4천 명의 주민이 있었다.", "number"),
        ("In 1848 the city had nearly forty thousand inhabitants.", "1848년에 도시에는 거의 4,000명의 주민이 있었다.", "number"),
        # M2M100's reversal of "not only ... but"
        ("Not only can your heart be changed, but your whole life can be renewed.",
         "당신의 마음을 바꿀 수는 없지만 당신의 삶 전체를 새롭게 할 수 있습니다.", "negation-added"),
    ],
)
def test_false_passes_are_caught(src, tgt, expected):
    assert expected in codes(src, tgt)


@pytest.mark.parametrize(
    "src,tgt",
    [
        # false positive: "two hundred" is 200, not 2
        ("Two hundred years later, the church was rebuilt.", "200년 후 교회는 재건되었다."),
        ("Two hundred years later, the church was rebuilt.", "이백 년 후에 교회가 재건되었다."),
        ("In 1848 the city had nearly forty thousand inhabitants.", "1848년에 그 도시에는 거의 4만 명의 주민이 있었다."),
        ("In 1848 the city had nearly forty thousand inhabitants.", "1848년에 그 도시의 주민은 거의 40,000명이었다."),
        ("She never forgot the promise she had made.", "그녀는 자신이 한 약속을 결코 잊지 않았다."),
        ("It is not only a duty but also a joy.", "그것은 의무일 뿐만 아니라 기쁨이기도 합니다."),
        ("No one knows the answer.", "아무도 답을 모른다."),
        ("The eighteenth century was a time of change.", "18세기는 변화의 시대였다."),
        ("Chapter Seven", "제7장"),
    ],
)
def test_correct_translations_are_not_flagged(src, tgt):
    assert codes(src, tgt) == []


def test_number_phrases():
    assert source_numbers("Two hundred years") == {200}
    assert source_numbers("forty thousand people") == {40000}
    assert source_numbers("one hundred and five") == {105}
    assert source_numbers("two thousand five hundred") == {2500}


def test_obligation_must_survive():
    # Windows exe run #14: "YOU MUST BE BORN AGAIN" -> "다시 태어났습니다." was shown as a translation.
    assert "obligation-lost" in codes("You must be born again", "다시 태어났습니다.")
    assert codes("You must be born again", "당신은 다시 태어나야 합니다.") == []
    assert codes("You must be born again", "너희가 거듭나야 하리라") == []
    assert codes("We should read the instructions.", "우리는 지침을 읽어야 한다.") == []


def test_company_marker_allowed_when_the_source_names_a_company():
    assert "web-phrase" not in codes("The Ford Motor Company opened a new plant.", "(주)포드 자동차가 새 공장을 열었다.")


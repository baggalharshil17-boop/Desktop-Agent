import pytest

from financial_voice_agent.orchestrator.interrupt_keywords import is_interrupt_phrase


@pytest.mark.parametrize(
    "transcript",
    [
        "wait",
        " Wait,",
        "Wait wait, that's not right",
        "no",
        "No, I meant something else",
        "stop",
        "Stop.",
        "hold on",
        "Hold on a second",
        "hang on",
        "actually",
        "never mind",
        "nevermind that",
        "cancel that",
        "sorry",
        "excuse me",
        "i didn't ask that",
        "I didn't ask for that",
        "not this",
        "not that one",
        "actually I meant to ask about Reliance instead",
    ],
)
def test_is_interrupt_phrase_matches_deliberate_redirects(transcript):
    assert is_interrupt_phrase(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "what's the stop-loss level for reliance",
        "what's the nifty level",
        "can you check my holdings",
        "tell me the news about tcs",
        "waiting for the market to open",  # "wait" as a prefix of another word must not match
        None,
        "",
        "   ",
    ],
)
def test_is_interrupt_phrase_does_not_match_normal_queries(transcript):
    assert is_interrupt_phrase(transcript) is False

from app.models.text_classifier import LABELS, text_model


def test_text_model_returns_all_scores():
    text = "Urgent ! Investissez dans cette crypto et doublez votre argent."

    scores = text_model.classify(text)

    assert set(scores.keys()) == set(LABELS)
    assert len(scores) == 11

    for score in scores.values():
        assert 0 <= score <= 100


def test_crypto_text_is_detected():
    text = "Investissez dans notre nouvelle crypto aujourd'hui."

    scores = text_model.classify(text)

    assert scores["crypto"] > 50
import math

from app.core.embeddings import HashingEmbedder


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_dimension_and_normalisation():
    e = HashingEmbedder(dim=128)
    (vec,) = e.embed(["Annual vacation policy for employees"])
    assert len(vec) == 128
    assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-6)


def test_deterministic_across_instances():
    text = "Passwords must be rotated every 90 days"
    assert HashingEmbedder(64).embed([text]) == HashingEmbedder(64).embed([text])


def test_related_text_scores_higher_than_unrelated():
    e = HashingEmbedder()
    query, related, unrelated = e.embed(
        [
            "how many vacation days do employees get",
            "Employees receive 25 days of paid vacation per year",
            "Laptops must use full disk encryption",
        ]
    )
    assert cosine(query, related) > cosine(query, unrelated) + 0.2


def test_empty_text_gives_zero_vector():
    (vec,) = HashingEmbedder(32).embed([""])
    assert vec == [0.0] * 32

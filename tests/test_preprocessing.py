import pytest

from nasdaq_nlp.preprocessing.text import (
    strip_header,
    strip_boilerplate_lines,
    split_sections,
    normalise,
    tokenise,
    tokenise_no_stopwords,
    preprocess_transcript,
)


class TestStripHeader:
    def test_removes_header_before_presentation(self):
        text = (
            "Thomson Reuters StreetEvents\n"
            "Some date line\n"
            "==========\n"
            "Corporate Participants\n"
            "==========\n"
            "John Doe - CEO\n"
            "==========\n"
            "PRESENTATION\n"
            "==========\n"
            "Revenue grew strongly this quarter."
        )
        result = strip_header(text)
        assert "Revenue grew" in result
        assert "PRESENTATION" in result
        # Header lines before the presentation section marker are stripped
        assert "Some date line" not in result

    def test_returns_unchanged_if_no_marker(self):
        text = "Just some text without any markers."
        assert strip_header(text) == text

    def test_handles_empty_string(self):
        assert strip_header("") == ""


class TestStripBoilerplateLines:
    def test_removes_forward_looking(self):
        text = "Revenue grew.\nThis contains forward-looking statements.\nProfits up."
        result = strip_boilerplate_lines(text)
        assert "forward-looking" not in result
        assert "Revenue grew." in result
        assert "Profits up." in result

    def test_removes_operator_instructions(self):
        text = "Good results.\nOperator instructions for questions.\nThank you."
        result = strip_boilerplate_lines(text)
        assert "Operator instructions" not in result

    def test_keeps_clean_lines(self):
        text = "Revenue grew.\nProfits increased.\nGuidance raised."
        assert strip_boilerplate_lines(text) == text


class TestSplitSections:
    def test_splits_at_qa_marker(self):
        text = "Presentation content here.\nQUESTIONS AND ANSWERS\nAnalyst: What about margins?"
        pres, qa = split_sections(text)
        assert "Presentation content" in pres
        assert "margins" in qa

    def test_no_qa_marker(self):
        text = "Just a presentation with no Q&A section."
        pres, qa = split_sections(text)
        assert pres == text
        assert qa == ""

    def test_qa_shorthand(self):
        text = "Prepared remarks.\nQ&A SESSION\nFirst question here."
        pres, qa = split_sections(text)
        assert "Prepared remarks" in pres
        assert "First question" in qa


class TestNormalise:
    def test_lowercases(self):
        assert normalise("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self):
        result = normalise("Revenue grew 15%! Great results.")
        assert "%" not in result
        assert "!" not in result

    def test_keeps_apostrophes(self):
        result = normalise("Don't we're can't")
        assert "don't" in result
        assert "we're" in result

    def test_collapses_whitespace(self):
        result = normalise("too   many    spaces")
        assert "  " not in result


class TestTokenise:
    def test_basic(self):
        tokens = tokenise("revenue grew strongly")
        assert tokens == ["revenue", "grew", "strongly"]

    def test_empty(self):
        assert tokenise("") == []

    def test_preserves_contractions(self):
        tokens = tokenise("don't we're")
        assert "don't" in tokens


class TestTokeniseNoStopwords:
    def test_removes_stopwords(self):
        tokens = tokenise_no_stopwords("the revenue is strong and growing")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "revenue" in tokens
        assert "strong" in tokens

    def test_removes_single_char_tokens(self):
        tokens = tokenise_no_stopwords("a b revenue")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "revenue" in tokens


class TestPreprocessTranscript:
    def test_returns_expected_keys(self):
        result = preprocess_transcript("Some transcript text here.")
        assert "presentation_raw" in result
        assert "qa_raw" in result
        assert "full_raw" in result
        assert "tokens" in result

    def test_full_section(self):
        result = preprocess_transcript("Revenue grew. Profits up.", section="full")
        assert len(result["tokens"]) > 0

    def test_with_stopword_removal(self):
        result_with = preprocess_transcript("The revenue is strong.", remove_stopwords=True)
        result_without = preprocess_transcript("The revenue is strong.", remove_stopwords=False)
        assert len(result_with["tokens"]) <= len(result_without["tokens"])

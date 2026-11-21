import re
import unittest
from unittest.mock import Mock, patch

from research_pipeline.answer_generator import (
    format_answer_citations,
    generate_evidence_grounded_answer
)


class CitationFormattingTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            {
                "article_id": "paper_a.pdf",
                "title": "Paper A",
                "year": 2023,
                "page_number": 3,
                "text": "Evidence from paper A."
            },
            {
                "article_id": "paper_b.pdf",
                "title": "Paper B",
                "year": 2022,
                "page_number": 2,
                "text": "Evidence from paper B."
            },
            {
                "article_id": "paper_a.pdf",
                "title": "Paper A",
                "year": 2023,
                "page_number": 17,
                "text": "More evidence from paper A."
            },
            {
                "article_id": "uncited.pdf",
                "title": "Uncited Paper",
                "year": 2021,
                "page_number": 9,
                "text": "This passage is not cited."
            }
        ]

    def test_numbers_citations_and_deduplicates_references_and_pages(self):
        answer = "Claim from A [E1]. Claim from B [E2]. More from A [E3]."

        formatted_answer, references = format_answer_citations(
            answer,
            self.evidence
        )

        self.assertEqual(
            formatted_answer,
            "Claim from A [1]. Claim from B [2]. More from A [1]."
        )
        self.assertEqual(len(references), 2)
        self.assertIn("[1] Paper A (2023).", references[0])
        self.assertIn("PDF file: paper_a.pdf.", references[0])
        self.assertIn("PDF pages 3, 17.", references[0])
        self.assertIn("PDF page 2.", references[1])
        self.assertNotIn("Uncited Paper", "\n".join(references))

    def test_reference_pages_are_sorted_numerically(self):
        evidence = [
            {
                "article_id": "paper_a.pdf",
                "title": "Paper A",
                "year": 2023,
                "page_number": 17
            },
            {
                "article_id": "paper_a.pdf",
                "title": "Paper A",
                "year": 2023,
                "page_number": 3
            },
            {
                "article_id": "paper_a.pdf",
                "title": "Paper A",
                "year": 2023,
                "page_number": "10"
            }
        ]

        _, references = format_answer_citations(
            "Evidence [E1, E2, E3].",
            evidence
        )

        self.assertIn("PDF pages 3, 10, 17.", references[0])

    def test_grouped_markers_map_all_papers_and_reuse_numbers(self):
        answer = "Combined claim [E1, E2, E3]."

        formatted_answer, references = format_answer_citations(
            answer,
            self.evidence
        )

        self.assertEqual(formatted_answer, "Combined claim [1, 2].")
        self.assertEqual(len(references), 2)
        self.assertIn("[1] Paper A (2023).", references[0])
        self.assertIn("PDF pages 3, 17.", references[0])
        self.assertIn("[2] Paper B (2022).", references[1])
        self.assertNotRegex(formatted_answer, r"\[E\d+")

    def test_numbers_follow_first_appearance_and_identity_type(self):
        evidence = [
            {
                "article_id": "Paper B",
                "title": "Garje et al.",
                "year": 2021,
                "page_number": 1
            },
            {
                "article_id": "neeraja.pdf",
                "title": "Paper B",
                "year": 2024,
                "page_number": 2
            }
        ]

        formatted_answer, references = format_answer_citations(
            "Neeraja [E2], then Garje [E1]. Repeated Neeraja [E2].",
            evidence
        )

        self.assertEqual(
            formatted_answer,
            "Neeraja [1], then Garje [2]. Repeated Neeraja [1]."
        )
        self.assertIn("[1] Paper B (2024).", references[0])
        self.assertIn("[2] Garje et al. (2021).", references[1])

    def test_invalid_ids_do_not_create_references_and_valid_ids_survive(self):
        formatted_answer, references = format_answer_citations(
            "Evidence [E999, E1, E0] and invalid [E77].",
            self.evidence
        )

        self.assertEqual(formatted_answer, "Evidence [1] and invalid .")
        self.assertEqual(len(references), 1)
        self.assertIn("paper_a.pdf", references[0])
        self.assertNotRegex(formatted_answer, r"\[E\d+")

    def test_unmapped_numeric_citations_are_removed(self):
        formatted_answer, references = format_answer_citations(
            "Claim [3] supported by [E1].",
            self.evidence
        )

        self.assertEqual(formatted_answer, "Claim  supported by [1].")
        self.assertEqual(len(references), 1)

    def test_every_final_citation_has_one_reference(self):
        formatted_answer, references = format_answer_citations(
            "A [E1, E2, E3], then B [E2].",
            self.evidence
        )
        cited_numbers = {
            int(number)
            for citation_group in re.findall(
                r"\[(\d+(?:\s*,\s*\d+)*)\]",
                formatted_answer
            )
            for number in re.findall(r"\d+", citation_group)
        }
        reference_numbers = [
            int(reference.split("]", 1)[0][1:])
            for reference in references
        ]

        self.assertEqual(cited_numbers, {1, 2})
        self.assertEqual(reference_numbers, [1, 2])
        self.assertEqual(len(reference_numbers), len(set(reference_numbers)))

    def test_unknown_or_unidentified_evidence_does_not_make_reference(self):
        evidence = [{"page_number": 8, "text": "No identifying metadata."}]

        formatted_answer, references = format_answer_citations(
            "Claim [E1] and unsupported marker [E9].",
            evidence
        )

        self.assertEqual(formatted_answer, "Claim  and unsupported marker .")
        self.assertEqual(references, [])

    @patch("research_pipeline.answer_generator.rerank_results")
    @patch("research_pipeline.answer_generator.hybrid_retrieval")
    def test_answer_generation_returns_only_referenced_papers(
        self,
        mock_hybrid_retrieval,
        mock_rerank_results
    ):
        mock_hybrid_retrieval.return_value = ["candidate"]
        mock_rerank_results.return_value = self.evidence
        gemini_client = Mock()
        gemini_client.interactions.create.return_value.output_text = (
            "Supported claim [E1]. Another claim [E2]. "
            "Repeated paper [E3]."
        )

        result = generate_evidence_grounded_answer(
            query="Research question",
            chunks=[],
            faiss_index=Mock(),
            embedding_model=Mock(),
            bm25_index=Mock(),
            cross_encoder=Mock(),
            gemini_client=gemini_client
        )

        self.assertEqual(
            result["answer"],
            "Supported claim [1]. Another claim [2]. Repeated paper [1]."
        )
        self.assertEqual(result["question"], "Research question")
        self.assertEqual(len(result["references"]), 2)
        self.assertEqual(result["evidence"], self.evidence)

        prompt = gemini_client.interactions.create.call_args.kwargs["input"]
        normalized_prompt = " ".join(prompt.split())
        self.assertIn(
            "Citation marker: [E1] Article ID: paper_a.pdf",
            normalized_prompt
        )
        self.assertIn(
            "Citation marker: [E2] Article ID: paper_b.pdf",
            normalized_prompt
        )
        self.assertIn(
            "A review-level count or trend must cite the review passage "
            "reporting it",
            normalized_prompt
        )
        self.assertIn(
            "retrieved evidence does not provide sufficient information "
            "for comparison",
            normalized_prompt
        )

    def test_review_finding_marker_maps_to_review_article(self):
        evidence = [
            {
                "article_id": "primary-study.pdf",
                "title": "Primary Study",
                "year": 2021,
                "page_number": 5,
                "text": "A phishing detection experiment."
            },
            {
                "article_id": "dl-review.pdf",
                "title": "Deep Learning Systematic Review",
                "year": 2024,
                "page_number": 12,
                "text": "The review included 19 studies and found DNN most frequent."
            }
        ]

        answer, references = format_answer_citations(
            "The review included 19 studies [E2].",
            evidence
        )

        self.assertEqual(answer, "The review included 19 studies [1].")
        self.assertEqual(len(references), 1)
        self.assertIn(
            "[1] Deep Learning Systematic Review (2024).",
            references[0]
        )
        self.assertIn("PDF file: dl-review.pdf.", references[0])
        self.assertNotIn("primary-study.pdf", references[0])

    @patch("research_pipeline.answer_generator.rerank_results")
    @patch("research_pipeline.answer_generator.hybrid_retrieval")
    def test_answer_without_valid_citations_has_no_references(
        self,
        mock_hybrid_retrieval,
        mock_rerank_results
    ):
        mock_hybrid_retrieval.return_value = []
        mock_rerank_results.return_value = self.evidence
        gemini_client = Mock()
        gemini_client.interactions.create.return_value.output_text = (
            "An answer without evidence markers."
        )

        result = generate_evidence_grounded_answer(
            query="Research question",
            chunks=[],
            faiss_index=Mock(),
            embedding_model=Mock(),
            bm25_index=Mock(),
            cross_encoder=Mock(),
            gemini_client=gemini_client
        )

        self.assertEqual(result["references"], [])
        self.assertEqual(result["answer"], "An answer without evidence markers.")


if __name__ == "__main__":
    unittest.main()

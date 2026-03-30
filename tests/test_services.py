"""Tests for service layer components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.chunker import DocumentChunker
from src.services.vector_store import Document


# ─── DocumentChunker Tests ───────────────────────────────────────────────────


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_empty_text_returns_empty_list(self):
        chunker = DocumentChunker(chunk_size=100, overlap=10)
        result = chunker.chunk("")
        assert result == []

    def test_none_text_returns_empty_list(self):
        chunker = DocumentChunker(chunk_size=100, overlap=10)
        result = chunker.chunk(None)
        assert result == []

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="Overlap must be smaller than chunk_size"):
            DocumentChunker(chunk_size=100, overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=50, overlap=60)

    def test_single_paragraph_no_split(self):
        chunker = DocumentChunker(chunk_size=500, overlap=10)
        text = "This is a short paragraph."
        result = chunker.chunk(text)
        assert len(result) == 1
        assert result[0]["content"] == text

    def test_chunk_has_metadata(self):
        chunker = DocumentChunker(chunk_size=500, overlap=10)
        text = "Hello world"
        result = chunker.chunk(text, metadata={"source": "test.txt"})
        assert len(result) == 1
        assert result[0]["metadata"]["source"] == "test.txt"
        assert "chunk_index" in result[0]["metadata"]

    def test_paragraph_splitting(self):
        """Text with paragraph separators should split on paragraphs."""
        chunker = DocumentChunker(chunk_size=50, overlap=10, separator="\n\n")
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        result = chunker.chunk(text)
        assert len(result) >= 2
        assert result[0]["content"].startswith("First")

    def test_fixed_size_chunking_without_separator(self):
        """Text without the separator should use fixed-size chunking."""
        chunker = DocumentChunker(chunk_size=20, overlap=5, separator="\n\n")
        text = "A" * 50  # 50 chars, no paragraph breaks
        result = chunker.chunk(text)
        assert len(result) > 1
        # Each chunk should be at most chunk_size characters
        for chunk in result:
            assert len(chunk["content"]) <= 20

    def test_chunk_index_increments(self):
        chunker = DocumentChunker(chunk_size=20, overlap=5, separator="\n\n")
        text = "A" * 60
        result = chunker.chunk(text)
        indices = [c["metadata"]["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_metadata_preserved_across_chunks(self):
        chunker = DocumentChunker(chunk_size=20, overlap=5)
        text = "A" * 60
        result = chunker.chunk(text, metadata={"doc_id": "doc1"})
        for chunk in result:
            assert chunk["metadata"]["doc_id"] == "doc1"

    def test_chunk_batch_basic(self):
        chunker = DocumentChunker(chunk_size=500, overlap=10)
        texts = ["Hello world", "Goodbye world"]
        result = chunker.chunk_batch(texts)
        assert len(result) == 2
        assert result[0]["metadata"]["doc_index"] == 0
        assert result[1]["metadata"]["doc_index"] == 1

    def test_chunk_batch_mismatched_metadata_raises(self):
        chunker = DocumentChunker(chunk_size=500, overlap=10)
        texts = ["Hello", "World"]
        metadatas = [{"a": 1}]  # Only one metadata for two texts
        with pytest.raises(ValueError, match="Number of texts and metadatas must match"):
            chunker.chunk_batch(texts, metadatas)

    def test_chunk_batch_with_metadata(self):
        chunker = DocumentChunker(chunk_size=500, overlap=10)
        texts = ["Text one", "Text two"]
        metadatas = [{"source": "a"}, {"source": "b"}]
        result = chunker.chunk_batch(texts, metadatas)
        assert result[0]["metadata"]["source"] == "a"
        assert result[1]["metadata"]["source"] == "b"


# ─── EmbeddingService Tests ─────────────────────────────────────────────────


class TestEmbeddingService:
    """Tests for EmbeddingService with mocked model."""

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        with patch("src.services.embeddings.SentenceTransformer") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            mock_instance.get_sentence_embedding_dimension.return_value = 384

            from src.services.embeddings import EmbeddingService
            service = EmbeddingService(model_name="mock-model")
            result = await service.embed([])
            assert result == []

    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self):
        import numpy as np
        with patch("src.services.embeddings.SentenceTransformer") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

            from src.services.embeddings import EmbeddingService
            service = EmbeddingService(model_name="mock-model")
            result = await service.embed(["hello", "world"])
            assert len(result) == 2
            assert len(result[0]) == 2

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_vector(self):
        import numpy as np
        with patch("src.services.embeddings.SentenceTransformer") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.array([[0.5, 0.6, 0.7]])

            from src.services.embeddings import EmbeddingService
            service = EmbeddingService(model_name="mock-model")
            result = await service.embed_query("test query")
            assert len(result) == 3
            assert result[0] == pytest.approx(0.5)

    def test_get_dimension(self):
        with patch("src.services.embeddings.SentenceTransformer") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            mock_instance.get_sentence_embedding_dimension.return_value = 384

            from src.services.embeddings import EmbeddingService
            service = EmbeddingService(model_name="mock-model")
            assert service.get_dimension() == 384


# ─── WeaviateStore Tests ────────────────────────────────────────────────────


class TestWeaviateStore:
    """Tests for WeaviateStore with mocked Weaviate client."""

    @pytest.mark.asyncio
    async def test_insert_document(self):
        with patch("src.services.vector_store.weaviate") as mock_weaviate:
            mock_client = MagicMock()
            mock_weaviate.Client.return_value = mock_client
            mock_client.schema.get.return_value = {"classes": []}
            mock_client.data_object.create.return_value = "uuid-123"

            from src.services.vector_store import WeaviateStore
            store = WeaviateStore(url="http://localhost:8080")
            result = await store.insert(
                doc_id="doc1",
                content="test content",
                embedding=[0.1, 0.2, 0.3],
                metadata={"source": "test"}
            )
            assert result == "uuid-123"
            mock_client.data_object.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_returns_documents(self):
        with patch("src.services.vector_store.weaviate") as mock_weaviate:
            mock_client = MagicMock()
            mock_weaviate.Client.return_value = mock_client
            mock_client.schema.get.return_value = {"classes": []}

            # Build a mock query chain
            mock_query = MagicMock()
            mock_client.query.get.return_value = mock_query
            mock_query.with_near_vector.return_value = mock_query
            mock_query.with_limit.return_value = mock_query
            mock_query.with_additional.return_value = mock_query
            mock_query.do.return_value = {
                "data": {
                    "Get": {
                        "Documents": [
                            {
                                "content": "result text",
                                "doc_id": "doc1",
                                "source": "file.txt",
                                "chunk_index": 0,
                                "_additional": {"id": "uuid-1", "distance": 0.2}
                            }
                        ]
                    }
                }
            }

            from src.services.vector_store import WeaviateStore
            store = WeaviateStore(url="http://localhost:8080")
            results = await store.search(query_embedding=[0.1, 0.2], top_k=5)
            assert len(results) == 1
            assert results[0].content == "result text"
            assert results[0].score == pytest.approx(1.0 / 1.2, rel=1e-3)

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        with patch("src.services.vector_store.weaviate") as mock_weaviate:
            mock_client = MagicMock()
            mock_weaviate.Client.return_value = mock_client
            mock_client.schema.get.return_value = {"classes": []}

            mock_query = MagicMock()
            mock_client.query.get.return_value = mock_query
            mock_query.with_near_vector.return_value = mock_query
            mock_query.with_limit.return_value = mock_query
            mock_query.with_additional.return_value = mock_query
            mock_query.do.return_value = {"data": {"Get": {"Documents": []}}}

            from src.services.vector_store import WeaviateStore
            store = WeaviateStore(url="http://localhost:8080")
            results = await store.search(query_embedding=[0.1], top_k=3)
            assert results == []

    def test_health_check_success(self):
        with patch("src.services.vector_store.weaviate") as mock_weaviate:
            mock_client = MagicMock()
            mock_weaviate.Client.return_value = mock_client
            mock_client.schema.get.return_value = {"classes": []}

            from src.services.vector_store import WeaviateStore
            store = WeaviateStore(url="http://localhost:8080")
            assert store.health_check() is True

    def test_health_check_failure(self):
        with patch("src.services.vector_store.weaviate") as mock_weaviate:
            mock_client = MagicMock()
            mock_weaviate.Client.return_value = mock_client
            # First call for __init__, second call for health_check
            mock_client.schema.get.side_effect = [{"classes": []}, Exception("connection error")]

            from src.services.vector_store import WeaviateStore
            store = WeaviateStore(url="http://localhost:8080")
            assert store.health_check() is False


# ─── RAGPipeline Tests ──────────────────────────────────────────────────────


class TestRAGPipeline:
    """Tests for RAGPipeline with fully mocked dependencies."""

    def _make_pipeline(self):
        """Create a RAGPipeline with mocked dependencies."""
        from src.services.rag_pipeline import RAGPipeline

        mock_vector_store = MagicMock()
        mock_embedding_service = MagicMock()
        mock_llm_service = MagicMock()

        pipeline = RAGPipeline(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            llm_service=mock_llm_service,
            top_k=3
        )
        return pipeline, mock_vector_store, mock_embedding_service, mock_llm_service

    @pytest.mark.asyncio
    async def test_query_full_flow(self):
        pipeline, mock_vs, mock_embed, mock_llm = self._make_pipeline()

        mock_embed.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vs.search = AsyncMock(return_value=[
            Document(id="1", content="relevant doc", metadata={"source": "a"}, score=0.9)
        ])
        mock_llm.generate = AsyncMock(return_value={
            "answer": "The answer is 42.",
            "tokens_used": 50
        })

        result = await pipeline.query("What is the meaning of life?")

        assert result.answer == "The answer is 42."
        assert len(result.sources) == 1
        assert result.sources[0].score == 0.9
        assert result.tokens_used == 50
        assert result.latency_ms > 0
        mock_embed.embed_query.assert_called_once_with("What is the meaning of life?")

    @pytest.mark.asyncio
    async def test_query_no_documents_found(self):
        pipeline, mock_vs, mock_embed, mock_llm = self._make_pipeline()

        mock_embed.embed_query = AsyncMock(return_value=[0.1, 0.2])
        mock_vs.search = AsyncMock(return_value=[])

        result = await pipeline.query("Unknown topic?")

        assert "couldn't find" in result.answer.lower()
        assert result.sources == []
        assert result.tokens_used == 0
        # LLM should not be called when no documents are found
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_uses_custom_top_k(self):
        pipeline, mock_vs, mock_embed, mock_llm = self._make_pipeline()

        mock_embed.embed_query = AsyncMock(return_value=[0.1])
        mock_vs.search = AsyncMock(return_value=[
            Document(id="1", content="doc", metadata={}, score=0.8)
        ])
        mock_llm.generate = AsyncMock(return_value={"answer": "ok", "tokens_used": 10})

        await pipeline.query("test?", top_k=10)

        # Verify search was called with custom top_k
        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args
        assert call_kwargs.kwargs.get("top_k") == 10 or call_kwargs[1].get("top_k") == 10

    @pytest.mark.asyncio
    async def test_ingest_document(self):
        pipeline, mock_vs, mock_embed, mock_llm = self._make_pipeline()

        mock_embed.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        mock_vs.insert = AsyncMock(return_value="uuid-1")

        # Use text with paragraph separator so chunker splits it
        text = "First paragraph.\n\nSecond paragraph."
        count = await pipeline.ingest("doc1", text, chunk_size=20, overlap=5)

        assert count >= 1
        assert mock_vs.insert.call_count == count

    @pytest.mark.asyncio
    async def test_ingest_empty_document(self):
        pipeline, mock_vs, mock_embed, mock_llm = self._make_pipeline()

        mock_embed.embed = AsyncMock(return_value=[])
        count = await pipeline.ingest("doc1", "")

        assert count == 0
        mock_vs.insert.assert_not_called()

    def test_build_context(self):
        pipeline, _, _, _ = self._make_pipeline()

        docs = [
            Document(id="1", content="First document.", metadata={}, score=0.9),
            Document(id="2", content="Second document.", metadata={}, score=0.8),
        ]
        context = pipeline._build_context(docs)
        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "First document." in context
        assert "Second document." in context

    def test_build_context_respects_max_length(self):
        pipeline, _, _, _ = self._make_pipeline()

        docs = [
            Document(id="1", content="A" * 3000, metadata={}, score=0.9),
            Document(id="2", content="B" * 3000, metadata={}, score=0.8),
        ]
        context = pipeline._build_context(docs, max_length=500)
        assert len(context) <= 510  # Allow small margin for formatting

    def test_health_check(self):
        pipeline, mock_vs, _, _ = self._make_pipeline()
        mock_vs.health_check.return_value = True

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(pipeline.health_check())
        assert result["vector_store"] is True
        assert result["embedding_service"] is True
        assert result["llm_service"] is True


# ─── LLMService Tests ───────────────────────────────────────────────────────


class TestLLMService:
    """Tests for LLMService with mocked OpenAI client."""

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            from src.services.llm import LLMService
            LLMService(provider="unsupported", api_key="fake")

    @pytest.mark.asyncio
    async def test_generate_returns_answer(self):
        with patch("src.services.llm.AsyncOpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            # Mock the response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Generated answer"
            mock_response.usage.total_tokens = 100

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            from src.services.llm import LLMService
            service = LLMService(provider="openai", api_key="fake-key")
            result = await service.generate(
                prompt="What is AI?",
                context="AI is artificial intelligence."
            )

            assert result["answer"] == "Generated answer"
            assert result["tokens_used"] == 100

    @pytest.mark.asyncio
    async def test_generate_uses_custom_temperature(self):
        with patch("src.services.llm.AsyncOpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "answer"
            mock_response.usage.total_tokens = 50

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            from src.services.llm import LLMService
            service = LLMService(provider="openai", api_key="fake-key")
            await service.generate(
                prompt="test",
                context="context",
                temperature=0.2
            )

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_generate_with_custom_system_prompt(self):
        with patch("src.services.llm.AsyncOpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "answer"
            mock_response.usage.total_tokens = 30

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            from src.services.llm import LLMService
            service = LLMService(provider="openai", api_key="fake-key")
            await service.generate(
                prompt="test",
                context="ctx",
                system_prompt="You are a pirate."
            )

            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0]["content"] == "You are a pirate."

    @pytest.mark.asyncio
    async def test_generate_raises_on_api_error(self):
        with patch("src.services.llm.AsyncOpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API rate limit")
            )

            from src.services.llm import LLMService
            service = LLMService(provider="openai", api_key="fake-key")
            with pytest.raises(Exception, match="API rate limit"):
                await service.generate(prompt="test", context="ctx")

"""LLM service for text generation."""

import logging
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for LLM-based text generation.
    
    Supports OpenAI API and compatible endpoints.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4-turbo-preview"
    ):
        """
        Initialize LLM service.
        
        Args:
            provider: Provider name (currently only "openai" supported)
            api_key: API key (if None, uses environment variable)
            base_url: Base URL for API (for custom endpoints)
            model: Model name to use
        """
        self.provider = provider
        self.model = model
        
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        logger.info(f"Initializing LLM service with provider: {provider}, model: {model}")
        
        if provider == "openai":
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate answer based on context.
        
        Args:
            prompt: User question/prompt
            context: Retrieved context to answer from
            system_prompt: Optional system prompt (default RAG prompt used if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with 'answer' and 'tokens_used'
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the answer cannot be found in the context, say so clearly. "
                "Always cite the relevant parts of the context in your answer."
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}"
            }
        ]
        
        try:
            logger.debug(f"Generating completion for prompt: {prompt[:100]}...")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            logger.debug(f"Generated answer ({tokens_used} tokens)")
            
            return {
                "answer": answer,
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Generate answer with streaming.
        
        Args:
            prompt: User question/prompt
            context: Retrieved context
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks of the generated answer
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the answer cannot be found in the context, say so clearly."
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}"
            }
        ]
        
        try:
            logger.debug(f"Streaming completion for prompt: {prompt[:100]}...")
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error streaming completion: {e}")
            raise


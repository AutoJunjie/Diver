"""
Unit tests for StrandsClient.
"""
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock external dependencies before importing utils
if 'openai' not in sys.modules:
    sys.modules['openai'] = MagicMock()
if 'boto3' not in sys.modules:
    sys.modules['boto3'] = MagicMock()
if 'strands' not in sys.modules:
    mock_strands = MagicMock()
    sys.modules['strands'] = mock_strands
if 'strands.models' not in sys.modules:
    sys.modules['strands.models'] = MagicMock()

from utils import StrandsClient, StrandsError


class TestStrandsError(unittest.TestCase):
    """Test StrandsError exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = StrandsError("Test error")
        self.assertEqual(str(error), "Test error")

    def test_error_with_code(self):
        """Test error with error code."""
        error = StrandsError("Test error", error_code="ImportError")
        self.assertEqual(str(error), "[ImportError] Test error")
        self.assertEqual(error.error_code, "ImportError")

    def test_error_with_original(self):
        """Test error with original exception."""
        original = ValueError("Original error")
        error = StrandsError("Wrapped error", original_error=original)
        self.assertEqual(error.original_error, original)


class TestStrandsClientInit(unittest.TestCase):
    """Test StrandsClient initialization."""

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_bedrock_initialization(self, mock_bedrock_model, mock_agent):
        """Test initialization with Bedrock provider."""
        client = StrandsClient(
            model_id="test-model",
            provider="bedrock",
            region="us-west-2"
        )

        self.assertEqual(client.model_id, "test-model")
        self.assertEqual(client.provider, "bedrock")
        self.assertEqual(client.region, "us-west-2")

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_initialization_defaults(self, mock_bedrock_model, mock_agent):
        """Test initialization with default values."""
        client = StrandsClient(model_id="test-model")

        self.assertEqual(client.provider, "bedrock")
        self.assertEqual(client.region, "us-west-2")
        self.assertEqual(client.temperature, 0.7)
        self.assertEqual(client.max_tokens, 32768)
        self.assertFalse(client.control_thinking)
        self.assertFalse(client.streaming)

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_initialization_custom(self, mock_bedrock_model, mock_agent):
        """Test initialization with custom values."""
        client = StrandsClient(
            model_id="custom-model",
            provider="bedrock",
            region="us-east-1",
            temperature=0.5,
            max_tokens=16384,
            control_thinking=True,
            streaming=True
        )

        self.assertEqual(client.model_id, "custom-model")
        self.assertEqual(client.region, "us-east-1")
        self.assertEqual(client.temperature, 0.5)
        self.assertEqual(client.max_tokens, 16384)
        self.assertTrue(client.control_thinking)
        self.assertTrue(client.streaming)


class TestStrandsClientPromptBuilding(unittest.TestCase):
    """Test prompt building methods."""

    def test_build_prompt_simple(self):
        """Test simple prompt building."""
        # Create client without full init
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False
        client.agent = MagicMock()

        messages = [{"role": "user", "content": "Hello"}]
        prompt = client._build_prompt(messages)

        self.assertEqual(prompt, "Hello")

    def test_build_prompt_with_system(self):
        """Test prompt building with system message."""
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False
        client.agent = MagicMock()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        prompt = client._build_prompt(messages)

        self.assertEqual(prompt, "Hello")
        client.agent.system_prompt = "You are helpful."

    def test_build_prompt_with_control_thinking(self):
        """Test prompt building with control_thinking enabled."""
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = True
        client.agent = MagicMock()

        messages = [{"role": "user", "content": "Test"}]
        prompt = client._build_prompt(messages)

        self.assertIn("Test", prompt)
        self.assertIn("</think>", prompt)
        self.assertIn("Okay, I think I have finished thinking.", prompt)

    def test_build_prompt_without_control_thinking(self):
        """Test prompt building with control_thinking disabled."""
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False
        client.agent = MagicMock()

        messages = [{"role": "user", "content": "Test"}]
        prompt = client._build_prompt(messages)

        self.assertNotIn("</think>", prompt)


class TestStrandsClientAnswerExtraction(unittest.TestCase):
    """Test answer extraction methods."""

    def test_extract_answer_with_think_tag(self):
        """Test extraction when response has think tags."""
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False

        response = "<think>thinking process</think>\nFinal answer here"
        result = client._extract_answer(response)

        self.assertEqual(result, "Final answer here")

    def test_extract_answer_without_think_tag(self):
        """Test extraction when response has no think tags."""
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False

        response = "Direct answer without thinking"
        result = client._extract_answer(response)

        self.assertEqual(result, "Direct answer without thinking")

    def test_extract_answer_multiple_think_tags(self):
        """Test extraction with multiple think tags (takes content after last)."""
        client = StrandsClient.__new__(StrandsClient)
        client.control_thinking = False

        response = "<think>first</think>middle<think>second</think>Final"
        result = client._extract_answer(response)

        self.assertEqual(result, "Final")


class TestStrandsClientChatCompletion(unittest.TestCase):
    """Test chat completion methods."""

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_chat_completion_single(self, mock_bedrock_model, mock_agent):
        """Test single response (n=1)."""
        mock_result = MagicMock()
        mock_result.message = "Test response"
        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = mock_result
        mock_agent.return_value = mock_agent_instance

        client = StrandsClient(model_id="test-model")
        client.agent = mock_agent_instance

        result = client.chat_completion(
            [{"role": "user", "content": "test"}],
            n=1
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Test response")

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_chat_completion_multiple(self, mock_bedrock_model, mock_agent):
        """Test multiple responses (n=2)."""
        mock_result = MagicMock()
        mock_result.message = "Test response"
        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = mock_result
        mock_agent.return_value = mock_agent_instance

        client = StrandsClient(model_id="test-model")
        client.agent = mock_agent_instance

        result = client.chat_completion(
            [{"role": "user", "content": "test"}],
            n=2
        )

        self.assertEqual(len(result), 2)

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_chat_completion_with_think_extraction(self, mock_bedrock_model, mock_agent):
        """Test response with think tags gets extracted."""
        mock_result = MagicMock()
        mock_result.message = "<think>thinking</think>Final answer"
        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = mock_result
        mock_agent.return_value = mock_agent_instance

        client = StrandsClient(model_id="test-model")
        client.agent = mock_agent_instance

        result = client.chat_completion(
            [{"role": "user", "content": "test"}],
            n=1
        )

        self.assertEqual(result[0], "Final answer")


class TestStrandsClientErrorHandling(unittest.TestCase):
    """Test error handling."""

    @patch('strands.Agent')
    @patch('strands.models.BedrockModel')
    def test_agent_call_failure(self, mock_bedrock_model, mock_agent):
        """Test handling of agent call failure."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.side_effect = Exception("Agent error")
        mock_agent.return_value = mock_agent_instance

        client = StrandsClient(model_id="test-model")
        client.agent = mock_agent_instance

        with self.assertRaises(StrandsError) as ctx:
            client.chat_completion([{"role": "user", "content": "test"}], n=1)

        self.assertIn("Strands call failed", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()

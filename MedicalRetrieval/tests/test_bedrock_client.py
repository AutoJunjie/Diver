"""
Unit tests for BedrockClient.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock external dependencies before importing utils
# This allows tests to run without openai/boto3 installed
if 'openai' not in sys.modules:
    sys.modules['openai'] = MagicMock()
if 'boto3' not in sys.modules:
    sys.modules['boto3'] = MagicMock()
# Note: Do NOT mock transformers here - BedrockClient needs the import to fail
# so it falls back to simple prompt format (self._tokenizer = None)

from utils import BedrockClient, BedrockError


class TestBedrockError(unittest.TestCase):
    """Test BedrockError exception class."""

    def test_basic_error(self):
        error = BedrockError("Test error")
        self.assertEqual(str(error), "Test error")

    def test_error_with_code(self):
        error = BedrockError("Test error", error_code="ThrottlingException")
        self.assertEqual(str(error), "[ThrottlingException] Test error")

    def test_error_with_original(self):
        original = ValueError("Original error")
        error = BedrockError("Wrapped error", original_error=original)
        self.assertEqual(error.original_error, original)


class TestBedrockClientInit(unittest.TestCase):
    """Test BedrockClient initialization."""

    @patch('boto3.client')
    def test_initialization_defaults(self, mock_boto_client):
        """Test initialization with default values."""
        client = BedrockClient(model_id="test-model")

        mock_boto_client.assert_called_once_with('bedrock-runtime', region_name='us-west-2')
        self.assertEqual(client.model_id, "test-model")
        self.assertEqual(client.region, "us-west-2")
        self.assertEqual(client.retry_attempts, 3)
        self.assertEqual(client.control_thinking, False)
        self.assertEqual(client.max_tokens, 32768)

    @patch('boto3.client')
    def test_initialization_custom(self, mock_boto_client):
        """Test initialization with custom values."""
        client = BedrockClient(
            model_id="custom-model",
            region="us-east-1",
            retry_attempts=5,
            control_thinking=True,
            max_tokens=16384
        )

        mock_boto_client.assert_called_once_with('bedrock-runtime', region_name='us-east-1')
        self.assertEqual(client.model_id, "custom-model")
        self.assertEqual(client.region, "us-east-1")
        self.assertEqual(client.retry_attempts, 5)
        self.assertEqual(client.control_thinking, True)
        self.assertEqual(client.max_tokens, 16384)


class TestBedrockClientPromptBuilding(unittest.TestCase):
    """Test prompt building methods."""

    @patch('boto3.client')
    def test_simple_prompt_format(self, mock_boto_client):
        """Test simple prompt format when tokenizer unavailable."""
        client = BedrockClient(model_id="test-model")
        client._tokenizer = None  # Force simple format

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]

        prompt = client._simple_prompt_format(messages)

        self.assertIn("System: You are helpful.", prompt)
        self.assertIn("User: Hello", prompt)
        self.assertIn("Assistant:", prompt)

    @patch('boto3.client')
    def test_control_thinking_adds_tag(self, mock_boto_client):
        """Test that control_thinking adds the skip tag when using simple format."""
        client = BedrockClient(model_id="test-model", control_thinking=True)
        # Force simple format by setting tokenizer to None
        client._tokenizer = None

        messages = [{"role": "user", "content": "Test"}]
        prompt = client._build_prompt(messages)

        # Ensure we got a string (not a MagicMock)
        self.assertIsInstance(prompt, str)
        self.assertIn("</think>", prompt)
        self.assertIn("Okay, I think I have finished thinking.", prompt)

    @patch('boto3.client')
    def test_control_thinking_disabled(self, mock_boto_client):
        """Test that control_thinking=False doesn't add tag."""
        client = BedrockClient(model_id="test-model", control_thinking=False)
        # Force simple format by setting tokenizer to None
        client._tokenizer = None

        messages = [{"role": "user", "content": "Test"}]
        prompt = client._build_prompt(messages)

        # Ensure we got a string (not a MagicMock)
        self.assertIsInstance(prompt, str)
        self.assertNotIn("</think>", prompt)


class TestBedrockClientAnswerExtraction(unittest.TestCase):
    """Test answer extraction methods."""

    @patch('boto3.client')
    def test_extract_answer_with_think_tag(self, mock_boto_client):
        """Test extraction when response has think tags."""
        client = BedrockClient(model_id="test-model")

        response = "<think>Let me think about this...</think>\nThe answer is 42."
        answer = client._extract_answer(response)

        self.assertEqual(answer, "The answer is 42.")

    @patch('boto3.client')
    def test_extract_answer_without_think_tag(self, mock_boto_client):
        """Test extraction when response has no think tags."""
        client = BedrockClient(model_id="test-model")

        response = "The answer is 42."
        answer = client._extract_answer(response)

        self.assertEqual(answer, "The answer is 42.")

    @patch('boto3.client')
    def test_extract_answer_multiple_think_tags(self, mock_boto_client):
        """Test extraction with multiple think tags (takes content after last)."""
        client = BedrockClient(model_id="test-model")

        response = "<think>First</think>Middle<think>Second</think>\nFinal answer."
        answer = client._extract_answer(response)

        self.assertEqual(answer, "Final answer.")


class TestBedrockClientInvoke(unittest.TestCase):
    """Test Bedrock API invocation."""

    @patch('boto3.client')
    def test_invoke_success(self, mock_boto_client):
        """Test successful API invocation."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            'output': {
                'message': {
                    'content': [{'text': '<think>thinking</think>\nAnswer: 42'}]
                }
            }
        }
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model")
        result = client._invoke_bedrock("Test prompt", 0.7, 1000)

        self.assertEqual(result, "<think>thinking</think>\nAnswer: 42")
        mock_bedrock.converse.assert_called_once()

    @patch('boto3.client')
    def test_invoke_empty_response(self, mock_boto_client):
        """Test handling of empty response."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            'output': {'message': {'content': []}}
        }
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model")
        result = client._invoke_bedrock("Test prompt", 0.7, 1000)

        self.assertEqual(result, "")


class TestBedrockClientChatCompletion(unittest.TestCase):
    """Test chat_completion method."""

    @patch('boto3.client')
    def test_chat_completion_single(self, mock_boto_client):
        """Test single response (n=1)."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            'output': {
                'message': {
                    'content': [{'text': 'Response text'}]
                }
            }
        }
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model")
        client._tokenizer = None

        results = client.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            n=1
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Response text")

    @patch('boto3.client')
    def test_chat_completion_multiple(self, mock_boto_client):
        """Test multiple responses (n=2)."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            'output': {
                'message': {
                    'content': [{'text': 'Response text'}]
                }
            }
        }
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model")
        client._tokenizer = None

        results = client.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            n=2
        )

        self.assertEqual(len(results), 2)

    @patch('boto3.client')
    def test_chat_completion_default_max_tokens(self, mock_boto_client):
        """Test that default max_tokens is used."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            'output': {
                'message': {
                    'content': [{'text': 'Response'}]
                }
            }
        }
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model", max_tokens=10000)
        client._tokenizer = None

        client.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            n=1
        )

        # Check that max_tokens was passed to converse
        call_args = mock_bedrock.converse.call_args
        self.assertEqual(call_args[1]['inferenceConfig']['maxTokens'], 10000)


class TestBedrockClientRetry(unittest.TestCase):
    """Test retry logic."""

    @patch('boto3.client')
    @patch('time.sleep')
    def test_retry_on_failure(self, mock_sleep, mock_boto_client):
        """Test retry on transient failure."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [
            Exception("Temporary error"),
            {
                'output': {
                    'message': {
                        'content': [{'text': 'Success'}]
                    }
                }
            }
        ]
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model", retry_attempts=3)
        client._tokenizer = None

        results = client.chat_completion(
            messages=[{"role": "user", "content": "Test"}],
            n=1
        )

        self.assertEqual(results[0], "Success")
        self.assertEqual(mock_bedrock.converse.call_count, 2)

    @patch('boto3.client')
    @patch('time.sleep')
    def test_retry_exhausted(self, mock_sleep, mock_boto_client):
        """Test that BedrockError is raised after retries exhausted."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = Exception("Persistent error")
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model", retry_attempts=2)
        client._tokenizer = None

        with self.assertRaises(BedrockError) as ctx:
            client.chat_completion(
                messages=[{"role": "user", "content": "Test"}],
                n=1
            )

        self.assertIn("Failed after 2 attempts", str(ctx.exception))


class TestBedrockClientErrorHandling(unittest.TestCase):
    """Test error handling."""

    @patch('boto3.client')
    def test_throttling_exception(self, mock_boto_client):
        """Test handling of ThrottlingException."""
        mock_bedrock = MagicMock()

        # Create mock exception class
        mock_bedrock.exceptions.ThrottlingException = type(
            'ThrottlingException', (Exception,), {}
        )
        mock_bedrock.converse.side_effect = mock_bedrock.exceptions.ThrottlingException()
        mock_boto_client.return_value = mock_bedrock

        client = BedrockClient(model_id="test-model", retry_attempts=1)

        with self.assertRaises(BedrockError) as ctx:
            client._invoke_bedrock("Test", 0.7, 1000)

        self.assertEqual(ctx.exception.error_code, "ThrottlingException")


if __name__ == '__main__':
    unittest.main()

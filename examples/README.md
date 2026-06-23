# Headroom Examples

This directory contains examples demonstrating Headroom's capabilities.

## Quick Start Examples

### basic_usage.py

Basic integration with OpenAI client:

```bash
export OPENAI_API_KEY='your-key'
python examples/basic_usage.py
```

### anthropic_example.py

Integration with Anthropic Claude:

```bash
export ANTHROPIC_API_KEY='your-key'
python examples/anthropic_example.py
```

### streaming_example.py

Streaming responses with optimization:

```bash
export OPENAI_API_KEY='your-key'
python examples/streaming_example.py
```

### tabular_compression_demo.py

Tabular + spreadsheet compression on generated sample data (no API key needed).
Shows where CSV/markdown tables and `.xlsx` workbooks compress and where compact,
all-unique data correctly passes through:

```bash
python examples/tabular_compression_demo.py            # run all scenarios
python examples/tabular_compression_demo.py --write DIR # also save the sample files
```

## Evaluation Examples

### smart_vs_naive_eval.py

Compare SmartCrusher against naive truncation:

```bash
export OPENAI_API_KEY='your-key'
python examples/smart_vs_naive_eval.py
```

### real_world_eval.py

Comprehensive evaluation with Anthropic models:

```bash
export ANTHROPIC_API_KEY='your-key'
python examples/real_world_eval.py
```

### real_world_openai_eval.py

Comprehensive evaluation with OpenAI models:

```bash
export OPENAI_API_KEY='your-key'
python examples/real_world_openai_eval.py
```

## Demo Directories

### langchain_demo/

Full LangChain agent integration demo:

```bash
# No API key needed for compression demo
PYTHONPATH=. python -m examples.langchain_demo.show_compression

# Full comparison (requires API key)
export OPENAI_API_KEY='your-key'
PYTHONPATH=. python -m examples.langchain_demo.run_comparison
```

See [langchain_demo/README.md](langchain_demo/README.md) for details.

### mcp_demo/

MCP (Model Context Protocol) integration demo:

```bash
export OPENAI_API_KEY='your-key'
PYTHONPATH=. python -m examples.mcp_demo.run_agent_eval
```

### strands_bedrock_demo.py

AWS Strands Agents + Bedrock integration demo. Showcases two Headroom integration patterns:

1. **HeadroomHookProvider** - Compresses tool outputs in real-time
2. **HeadroomStrandsModel** - Optimizes entire conversation context

```bash
# Configure AWS credentials
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-west-2'  # Optional, defaults to us-west-2

# Or use AWS profile
export AWS_PROFILE='your-profile-name'

# Run the full demo (both integration patterns)
python examples/strands_bedrock_demo.py

# Run only the hook provider demo
python examples/strands_bedrock_demo.py --hook

# Run only the model wrapper demo
python examples/strands_bedrock_demo.py --model

# Specify a different AWS region
python examples/strands_bedrock_demo.py --region us-east-1
```

The demo uses Claude 3 Haiku via Bedrock for cost efficiency. It creates agents with
4 tools that return verbose JSON output (search results, logs, database records, metrics)
and displays compression statistics with visual comparisons.

**Requirements:**
- AWS account with Bedrock enabled
- Claude 3 Haiku model access in your region
- `pip install strands-agents headroom-ai[strands]`

## Running Examples

All examples can be run from the repository root:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run any example
python examples/<example_name>.py
```

## Expected Results

| Example | Token Savings | Notes |
|---------|---------------|-------|
| basic_usage | 50-70% | Simple tool output compression |
| langchain_demo | 70-85% | Real agent with multiple tools |
| mcp_demo | 60-80% | MCP tool outputs |
| strands_bedrock_demo | 60-85% | Strands + Bedrock with verbose tools |
| real_world_eval | 50-90% | Varies by scenario |

## Troubleshooting

**ModuleNotFoundError: No module named 'headroom'**

Run from the repository root with PYTHONPATH:

```bash
PYTHONPATH=. python examples/basic_usage.py
```

Or install in development mode:

```bash
pip install -e .
```

**API Key Errors**

Ensure your API keys are set:

```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
```

**AWS Credentials Errors (for Strands demo)**

Ensure AWS credentials are configured:

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'

# Option 2: AWS profile
export AWS_PROFILE='your-profile-name'

# Option 3: AWS credentials file (~/.aws/credentials)
```

Also ensure Bedrock and the Claude 3 Haiku model are enabled in your AWS account.

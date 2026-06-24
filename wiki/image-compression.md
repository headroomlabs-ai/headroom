# Image Compression

Headroom automatically compresses images in your LLM requests, reducing token usage by **40-90%** while maintaining answer accuracy.

## Overview

Vision models charge by the token, and images are expensive:
- A 1024x1024 image costs ~765 tokens (OpenAI)
- A 2048x2048 image costs ~2,900 tokens

Headroom's image compression uses a **trained ML router** to analyze your query and automatically select the optimal compression technique:

| Technique | Savings | When Used |
|-----------|---------|-----------|
| `full_low` | ~87% | General questions ("What is this?") |
| `preserve` | 0% | Fine details needed ("Count the whiskers") |
| `crop` | 50-90% | Region-specific ("What's in the corner?") |
| `transcode` | ~99% | Text extraction ("Read the sign") |

## How It Works

```
User uploads image + asks question
           ↓
   [Query Analysis]
   TrainedRouter (MiniLM from HuggingFace)
   Classifies: "What animal is this?" → full_low
           ↓
   [Image Analysis]
   SigLIP analyzes image properties
   (has text? complex? fine details?)
           ↓
   [Apply Compression]
   OpenAI: detail="low"
   Anthropic: Resize to 512px
   Google: Resize to 768px
           ↓
   Compressed request to LLM
```

## Quick Start

### With Headroom Proxy (Zero Code Changes)

```bash
# Start the proxy
headroom proxy --port 8787

# Connect your client
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

Images are automatically compressed based on your queries.

### With HeadroomClient

```python
from headroom import HeadroomClient

client = HeadroomClient(provider="openai")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What animal is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)
# Image automatically compressed with detail="low" (87% savings)
```

### Direct API

```python
from headroom.image import ImageCompressor

compressor = ImageCompressor()

# Compress images in messages
compressed_messages = compressor.compress(messages, provider="openai")

# Check savings
print(f"Saved {compressor.last_savings:.0f}% tokens")
print(f"Technique: {compressor.last_result.technique.value}")
```

## Configuration

### Proxy Configuration

```bash
# Enable image compression (default: true)
headroom proxy --image-optimize

# Disable image compression
headroom proxy --no-image-optimize
```

### Programmatic Configuration

```python
from headroom.image import ImageCompressor

compressor = ImageCompressor(
    model_id="chopratejas/technique-router",  # HuggingFace model
    use_siglip=True,   # Enable image analysis
    device="cuda",     # Use GPU if available
)
```

## Provider Support

| Provider | Detection | Compression Method |
|----------|-----------|-------------------|
| **OpenAI** | `image_url` | Sets `detail="low"` |
| **Anthropic** | `image` with `source` | Resizes to 512px |
| **Google** | `inlineData` | Resizes to 768px (tile-optimized) |

### OpenAI

Uses the native `detail` parameter:
```python
# Before
{"type": "image_url", "image_url": {"url": "data:..."}}

# After (full_low technique)
{"type": "image_url", "image_url": {"url": "data:...", "detail": "low"}}
```

### Anthropic

Resizes the image using PIL:
```python
# Before: 1024x1024 image (~1,398 tokens)
# After:  512x512 image (~349 tokens) - 75% savings
```

### Google Gemini

Resizes to 768px (optimal for Gemini's 768x768 tile system):
```python
# Before: 1536x1536 image (4 tiles × 258 = 1,032 tokens)
# After:  768x768 image (1 tile × 258 = 258 tokens) - 75% savings
```

## Techniques Explained

### `full_low` (87% savings)

Best for general understanding questions:
- "What is this?"
- "Describe the scene"
- "Is this indoors or outdoors?"

The model doesn't need fine details to answer these questions.

### `preserve` (0% savings)

Required when fine details matter:
- "Count the whiskers"
- "What brand is shown?"
- "Read the serial number"
- "What time does the clock show?"

### `crop` (50-90% savings)

For region-specific queries:
- "What's in the top-right corner?"
- "Focus on the background"
- "Zoom into the left side"

*Note: Currently implemented as resize. True cropping coming soon.*

### `transcode` (99% savings)

For text extraction (converts image to text):
- "Read the sign"
- "What does it say?"
- "Transcribe the document"

*Note: Requires vision model call. Currently falls back to preserve.*

## The Trained Router

The routing decision is made by a fine-tuned **MiniLM** classifier:

- **Model**: `chopratejas/technique-router` on HuggingFace
- **Size**: ~128MB
- **Accuracy**: 93.7% on validation set
- **Training data**: 1,157 examples across 4 techniques

The model is downloaded automatically on first use and cached locally.

### Training Data Examples

| Query | Technique |
|-------|-----------|
| "What animal is this?" | `full_low` |
| "Count the spots" | `preserve` |
| "Read the text on the sign" | `transcode` |
| "What's in the corner?" | `crop` |

## Performance

### Token Savings by Query Type

| Query Type | Before | After | Savings |
|------------|--------|-------|---------|
| General ("What is this?") | 765 | 85 | 89% |
| Detail ("Count items") | 765 | 765 | 0% |
| Region ("Top corner?") | 765 | 85 | 89% |
| Text ("Read the sign") | 765 | 85 | 89% |

### Latency

- Router inference: ~10ms (CPU), ~2ms (GPU)
- Image resize: ~5-20ms depending on size
- First request: +2-3s (model download, cached after)

## Troubleshooting

### Model Download Issues

The HuggingFace model downloads on first use:

```python
# Force a specific cache directory
import os
os.environ["HF_HOME"] = "/path/to/cache"

from headroom.image import ImageCompressor
compressor = ImageCompressor()
```

### GPU Memory

SigLIP requires ~400MB GPU memory. To use CPU only:

```python
compressor = ImageCompressor(device="cpu")
```

### Disable Image Compression

```python
# Proxy
headroom proxy --no-image-optimize

# Direct
# Simply don't call compress()
```

## API Reference

### `ImageCompressor`

```python
class ImageCompressor:
    def __init__(
        self,
        model_id: str = "chopratejas/technique-router",
        use_siglip: bool = True,
        device: str | None = None,
    ): ...

    def has_images(self, messages: list[dict]) -> bool:
        """Check if messages contain images."""

    def compress(
        self,
        messages: list[dict],
        provider: str = "openai",
    ) -> list[dict]:
        """Compress images in messages."""

    @property
    def last_result(self) -> CompressionResult | None:
        """Result of last compression."""

    @property
    def last_savings(self) -> float:
        """Savings percentage from last compression."""
```

### `CompressionResult`

```python
@dataclass
class CompressionResult:
    technique: Technique      # full_low, preserve, crop, transcode
    original_tokens: int      # Estimated tokens before
    compressed_tokens: int    # Estimated tokens after
    confidence: float         # Router confidence (0-1)

    @property
    def savings_percent(self) -> float:
        """Percentage of tokens saved."""
```

### `Technique`

```python
class Technique(Enum):
    FULL_LOW = "full_low"     # 87% savings
    PRESERVE = "preserve"     # 0% savings
    CROP = "crop"             # 50-90% savings
    TRANSCODE = "transcode"   # 99% savings
```

## See Also

- [Compression Guide](compression.md) - Text compression techniques
- [CCR Guide](ccr.md) - Reversible compression with retrieval
- [Proxy Guide](proxy.md) - Zero-code deployment
- [Architecture](ARCHITECTURE.md) - System design

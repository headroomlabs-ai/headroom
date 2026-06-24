"""Image token compression for Headroom.

Automatically compress images in LLM requests to save 40-90% tokens
while maintaining answer accuracy.

Usage:
    from headroom.image import ImageCompressor

    compressor = ImageCompressor()

    # Check if messages have images
    if compressor.has_images(messages):
        # Compress based on query intent
        messages = compressor.compress(messages, provider="openai")
        print(f"Saved {compressor.last_savings:.0f}% tokens")

Or use the convenience function:
    from headroom.image import compress_images

    messages = compress_images(messages, provider="openai")

The compression technique is selected by a trained ML model:
- FULL_LOW: General questions → 87% savings (detail="low")
- PRESERVE: Fine details needed → 0% savings (keep quality)
- CROP: Region-specific → 50-90% savings (extract region)
- TRANSCODE: Text extraction → 99% savings (OCR to text)

Model: https://huggingface.co/chopratejas/technique-router
"""

from .compressor import (
    CompressionResult,
    ImageCompressor,
    Technique,
    compress_images,
    get_compressor,
)

__all__ = [
    # Main API
    "ImageCompressor",
    "compress_images",
    "get_compressor",
    # Types
    "Technique",
    "CompressionResult",
]

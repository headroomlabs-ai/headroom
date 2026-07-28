#!/usr/bin/env python3
"""
Recreation script for Vertex AI capabilities in headroom.
Tests the Headroom proxy using the official google-genai SDK.

Run requirements:
    pip install google-genai

Usage:
    Start the headroom proxy in one terminal:
        headroom proxy --vertex-api-url https://us-central1-aiplatform.googleapis.com
    
    Then run this script:
        export LOCATION="us-central1" # Or 'global'
        export MODEL="gemini-flash-latest" # Or 'claude-3-5-sonnet-v2@20241022'
        python3 scripts/repro_vertex_genai_sdk.py
"""

import os
import sys

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai is not installed.")
    print("Run `pip install google-genai` and try again.")
    sys.exit(1)

def main():
    location = os.environ.get("LOCATION", "us-central1")
    model = os.environ.get("MODEL", "gemini-flash-latest")
    
    print(f"Running Vertex SDK Recreation via Headroom Proxy")
    print(f"Location: {location}")
    print(f"Model: {model}")
    print("-" * 50)
    
    # Check for GCP credentials
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and not os.environ.get("GCP_ACCESS_TOKEN"):
        # We don't strictly require it if the proxy is handling auth, but gcloud ADC is typical
        pass
    
    # Configure the client to point to the local headroom proxy.
    # The proxy expects standard Vertex REST paths.
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GCP_PROJECT_ID", "dummy-project-id"),
        location=location,
        http_options={'api_endpoint': '127.0.0.1:8787'}
    )
    
    # We will test both standard inference and thinking extensions
    
    print("\n1. Standard Inference:")
    content = "What is the airspeed velocity of an unladen swallow? Answer in one sentence."
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=content,
        )
        print("Response received successfully!")
        print(f"> {response.text}")
    except Exception as e:
        print(f"Failed standard inference: {e}")
        
    print("\n2. Inference with Thinking Config:")
    # Configure thinking config (where supported). 
    # For now, we just pass parameters and see if Headroom properly parses/forwards them.
    try:
        # Note: thinking_config in the google-genai SDK 
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget_tokens=100
            ) 
        )
        
        # Note: 'think deeply' to encourage thought process if enabled.
        response = client.models.generate_content(
            model=model,
            contents="Think deeply. Which is heavier: a kg of feathers or a kg of steel?",
            config=config,
        )
        print("Thinking inference succeeded!")
        print(f"> {response.text}")
    except Exception as e:
        print(f"Thinking inference had an error (this may be expected if the upstream model doesn't support thinking budgets yet): {e}")

if __name__ == "__main__":
    main()

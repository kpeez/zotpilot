"""
Configuration utilities.
"""

import json
import os
from pathlib import Path
from typing import Any

# Constants
CONFIG_DIR = Path.home() / ".paperchat"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict[str, Any]:
    """Load config from file or create default if not exists."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Save config to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_openai_api_key() -> str | None:
    """
    Get OpenAI API key from environment or config file.
    Returns None if no key is found.
    """
    # check for environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # check for config file
    config = load_config()
    return config.get("openai_api_key")


def setup_openai_api_key() -> str:
    """
    Interactive setup for OpenAI API key.
    Returns the API key after saving to config.
    """
    print("\nℹ️  OpenAI API Key Setup")  # noqa: RUF001
    print("------------------------")
    print("To use AI features, we need an OpenAI API key.")
    print("You can either:")
    print("1. Set the OPENAI_API_KEY environment variable, or")
    print("2. Enter your API key now to save in ~/.paperchat/config.json")

    try:
        from getpass import getpass

        key = getpass("\nEnter OpenAI API key (or press Enter to cancel): ").strip()

        if not key:
            print(
                "\n⚠️  No key provided. Please set OPENAI_API_KEY environment variable to use AI features."
            )
            raise ValueError("No API key provided")

        if not key.startswith("sk-"):
            print("\n⚠️ Invalid key format. OpenAI API keys should start with 'sk-'")
            raise ValueError("Invalid API key format")

        # Save to config
        config = load_config()
        config["openai_api_key"] = key
        save_config(config)
        print("\n✓ API key saved successfully!")

        return key

    except (KeyboardInterrupt, EOFError):
        print(
            "\n\n⚠️  Setup cancelled. Please set OPENAI_API_KEY environment variable to use AI features."
        )
        raise ValueError("Setup cancelled") from None

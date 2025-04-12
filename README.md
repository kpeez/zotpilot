---
title:
---

<p align="left">
  <img src="docs/assets/logo.png" width="500" alt="PaperChat logo">
</p>

---

Paperchat is an AI-powered assistant that enables you to take your research to the next level by enabling you to chat with your research library.

## Installation

The easiest and fastest way to install PaperChat is using [uv](https://github.com/astral-sh/uv):

```bash
uv tool install git+https://github.com/kpeez/paperchat.git
```

This will install PaperChat and make it available as a command-line tool.

To launch PaperChat, run:

```bash
paperchat
```

Alternatively, you can install PaperChat into your Python environment (requires Python 3.12 or newer):

```bash
pip install git+https://github.com/kpeez/paperchat.git
```

and then launch PaperChat by running:

```bash
paperchat
```

## Quick Start

1. Launch PaperChat: `paperchat`
2. Upload a PDF document using the sidebar
3. Wait for processing to complete
4. Start asking questions about your document!

## Settings

PaperChat uses your own model API keys to connect to the AI providers. If you have API keys set as environment variables, they will be automatically detected. Otherwise, you can set them in the settings page. To get API keys, you can sign up for a free account on the providers' websites:

- [OpenAI](https://platform.openai.com/api-keys)
- [Anthropic](https://console.anthropic.com/settings/keys)
- [Gemini](https://aistudio.google.com/apikey)

## Troubleshooting

- Report issues on our [GitHub Issues page](https://github.com/kpeez/paperchat/issues)

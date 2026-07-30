# Financial Voice Agent

A read-only AI voice assistant for a Zerodha Kite trading dashboard. See
`Financial_Voice_Agent.claude.md` for the full design/developer reference.

## Getting Started (Fresh Clone)

### 1. Prerequisites

- Python 3.11+ (developed against 3.14).
- `pip install -r requirements.txt` installs everything, but two packages
  need a callout on Windows:
  - **torch** (used for voice-activity detection): if you don't have or
    want a GPU, install the smaller CPU-only build first, or plain
    `pip install torch` may pull a multi-GB CUDA build:
    ```
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
  - **pyaudio** (microphone/speaker access): sometimes fails to build from
    source on Windows. If `pip install -r requirements.txt` fails on it
    specifically, install a prebuilt wheel instead:
    ```
    pip install pipwin
    pipwin install pyaudio
    ```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the setup wizard

```
python scripts/setup.py
```

Walks you through picking providers and pasting in your own API keys,
validating each one live as you go. Writes `.env` and `config.yaml` for
you -- see the table below for what each provider needs.

### 4. Run it

```
python -m financial_voice_agent
```

Talk after it prints "Listening...". Ctrl+C to stop.

If you chose `mode: "live"` during setup, get today's Kite access token
first (it expires daily):
```
python scripts/kite_login.py
```

## What each provider needs

| Provider | Env var | Get it from | Notes |
|---|---|---|---|
| Groq (STT and/or LLM) | `GROQ_API_KEY` | console.groq.com | Free tier available |
| Hugging Face (STT and/or LLM alternative) | `HF_TOKEN` | huggingface.co/settings/tokens | Free tier available, separate quota from Groq |
| Cartesia (TTS) | `CARTESIA_API_KEY` | play.cartesia.ai | Credit-based free tier |
| Fish Audio (TTS alternative) | `FISH_AUDIO_API_KEY` | fish.audio/app/api-keys | `s2.1-pro-free` model works free; other models need funded API credit |
| Tavily (news search) | `TAVILY_API_KEY` | tavily.com | Free to 1k searches/month |
| Indian Stock API (fundamentals -- P/E, market cap, etc.) | `INDIAN_STOCK_API_KEY` | indianapi.in (dashboard's API/Manage Keys section) | Free plan available |
| Zerodha Kite Connect (live trading data) | `KITE_API_KEY` / `KITE_API_SECRET` / `KITE_ACCESS_TOKEN` | developers.kite.trade | Paid (~₹500/month), only needed for `mode: "live"` -- skip entirely with `mode: "mock"` |

## Development

Run tests: `python -m pytest -q`

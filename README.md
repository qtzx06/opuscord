# opuscord

discord self-bot powered by claude opus 4.5 with semantic memory (RAG)

## what it does

- lives in your gc as a regular user
- responds when you say "opus" or "claude"
- randomly chimes in every 4-8 messages if something interesting happens
- has full memory of chat history via voyage embeddings
- can analyze users based on their message patterns
- vibes: smart friend energy, not corporate assistant

## setup

```bash
# install deps
uv sync

# copy env template
cp .env.example .env
```

fill in `.env`:
```
DISCORD_TOKEN=         # from discord devtools (network tab -> authorization header)
TARGET_CHANNEL_ID=     # right click channel -> copy id
BOT_USER_ID=           # your bot account's user id
ANTHROPIC_API_KEY=     # from console.anthropic.com
VOYAGE_API_KEY=        # from dash.voyageai.com
```

## run

```bash
uv run bot.py
```

first run scrapes all messages from channel and builds embeddings (takes a few mins).
after that it just catches up on recent messages.

## how it works

1. **scrape** - fetches all messages from discord on first run
2. **chunk** - merges consecutive messages from same user (within 60s) into semantic chunks
3. **embed** - voyage embeddings for semantic search
4. **rag** - when triggered, searches history for relevant context
5. **respond** - claude opus 4.5 with recent chat + relevant history

## models

- **claude opus 4.5** - main responses
- **claude haiku 3.5** - random interjection decisions (cheap/fast)
- **voyage-3-lite** - embeddings

## files

- `bot.py` - main bot
- `live_history.json` - scraped messages (gitignored)
- `embeddings.npz` - cached embeddings (gitignored)
- `.env` - secrets (gitignored)

## note

this is a self-bot which violates discord tos. use at your own risk.

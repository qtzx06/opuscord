import discord
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
from collections import defaultdict
import asyncio

client = discord.Client()
anthropic = Anthropic()

# config from env
TARGET_CHANNEL_ID = int(os.getenv("TARGET_CHANNEL_ID", 0))
BOT_USER_ID = int(os.getenv("BOT_USER_ID", 0))

# conversation memory per channel
conversations = defaultdict(list)
MAX_MEMORY = 50  # messages to remember

SYSTEM_PROMPT = """You are Opus, chatting in a Discord group chat. You're witty, casual, and helpful.
Keep responses concise - this is a chat, not an essay. Match the vibe of the conversation.
Don't be overly formal or use corporate speak. You can use lowercase, slang, whatever fits."""


@client.event
async def on_ready():
    print(f"logged in as {client.user} (id: {client.user.id})")
    print(f"watching channel: {TARGET_CHANNEL_ID}")


@client.event
async def on_message(message):
    # ignore other channels
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    # ignore own messages
    if message.author.id == BOT_USER_ID:
        return

    # store message in memory
    conversations[message.channel.id].append({
        "author": message.author.display_name,
        "content": message.content
    })

    # trim memory
    if len(conversations[message.channel.id]) > MAX_MEMORY:
        conversations[message.channel.id] = conversations[message.channel.id][-MAX_MEMORY:]

    # trigger: mention or keyword
    should_respond = (
        client.user.mentioned_in(message) or
        "opus" in message.content.lower() or
        "claude" in message.content.lower()
    )

    if not should_respond:
        return

    # build context from memory
    history = conversations[message.channel.id]
    context = "\n".join([f"{m['author']}: {m['content']}" for m in history])

    async with message.channel.typing():
        try:
            response = anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Chat history:\n{context}\n\nRespond to the latest message naturally."}]
            )

            reply = response.content[0].text

            # split long messages (discord limit is 2000)
            if len(reply) > 1900:
                reply = reply[:1900] + "..."

            await message.channel.send(reply)

            # add bot response to memory
            conversations[message.channel.id].append({
                "author": "Opus",
                "content": reply
            })

        except Exception as e:
            print(f"error: {e}")


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("missing DISCORD_TOKEN in environment")
        exit(1)
    if not TARGET_CHANNEL_ID:
        print("missing TARGET_CHANNEL_ID in environment")
        exit(1)

    client.run(token)

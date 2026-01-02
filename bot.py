import discord
import os
import json
import atexit
import numpy as np
import voyageai
import httpx
from dotenv import load_dotenv
from anthropic import Anthropic
from collections import defaultdict
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import base64

load_dotenv()

LIVE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "live_history.json")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddings.npz")
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.txt")

client = discord.Client()
anthropic = Anthropic()
voyage = voyageai.Client()  # uses VOYAGE_API_KEY env var

# config from env
TARGET_CHANNEL_ID = int(os.getenv("TARGET_CHANNEL_ID", 0))
BOT_USER_ID = int(os.getenv("BOT_USER_ID", 0))
HISTORY_FILE = os.getenv("HISTORY_FILE", "")

# conversation memory per channel
conversations = defaultdict(list)
MAX_MEMORY = 50

# historical messages from JSON export
chat_history = []
user_profiles = {}

# embeddings
embeddings_matrix = None  # numpy array of embeddings
embedded_indices = set()  # which chat_history indices have embeddings

# name aliases (same person, different names)
ALIASES = {
    "josh": ["quantizix", "qtzx06", "qtzx"],
    "quantizix": ["josh", "qtzx06", "qtzx"],
    "qtzx06": ["josh", "quantizix", "qtzx"],
    "qtzx": ["josh", "quantizix", "qtzx06"],
}


def save_live_history():
    """Save new messages to disk"""
    try:
        with open(LIVE_HISTORY_FILE, 'w') as f:
            json.dump({"messages": chat_history, "profiles": user_profiles}, f, indent=2)
        print(f"saved {len(chat_history)} messages to live_history.json")
    except Exception as e:
        print(f"failed to save history: {e}")


def load_history():
    """Load Discord export JSON + live history into memory"""
    global chat_history, user_profiles

    # load original export
    if HISTORY_FILE and os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)

        for msg in data.get("messages", []):
            if msg.get("type") != "Default":
                continue

            author = msg.get("author", {})
            author_id = author.get("id")
            author_name = author.get("nickname") or author.get("name", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")

            if not content.strip():
                continue

            chat_history.append({
                "author_id": author_id,
                "author": author_name,
                "content": content,
                "timestamp": timestamp
            })

            if author_id and author_id not in user_profiles:
                user_profiles[author_id] = {"name": author_name, "messages": []}
            if author_id:
                user_profiles[author_id]["messages"].append(content)

        print(f"loaded {len(chat_history)} messages from export")

    # load live history (new messages since export)
    if os.path.exists(LIVE_HISTORY_FILE):
        with open(LIVE_HISTORY_FILE, 'r') as f:
            live_data = json.load(f)

        live_msgs = live_data.get("messages", [])
        # only add messages not already in chat_history (by timestamp)
        existing_timestamps = {m.get("timestamp") for m in chat_history}
        new_count = 0
        for msg in live_msgs:
            if msg.get("timestamp") not in existing_timestamps:
                chat_history.append(msg)
                new_count += 1

        # merge profiles
        for uid, profile in live_data.get("profiles", {}).items():
            if uid not in user_profiles:
                user_profiles[uid] = profile
            else:
                # add new messages
                existing = set(user_profiles[uid]["messages"])
                for m in profile["messages"]:
                    if m not in existing:
                        user_profiles[uid]["messages"].append(m)

        print(f"loaded {new_count} new messages from live history")

    print(f"total: {len(chat_history)} messages, {len(user_profiles)} users")


# save on exit
atexit.register(save_live_history)

# global memory from full history analysis
gc_memory = ""


def load_memory():
    """Load the memory dump if it exists"""
    global gc_memory
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            gc_memory = f.read()
        print(f"loaded memory ({len(gc_memory)} chars)")


from concurrent.futures import ThreadPoolExecutor, as_completed


def summarize_chunk(args):
    """Summarize a single chunk"""
    chunk_text, chunk_num, total = args
    try:
        response = anthropic.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            system="""summarize this chunk of discord messages. extract:
- key topics discussed
- notable events or moments
- who said what (important stuff only)
- any inside jokes or recurring themes
- what people are working on or interested in

be concise but capture the important stuff.""",
            messages=[{"role": "user", "content": chunk_text}]
        )
        summary = response.content[0].text.strip()
        print(f"  [{chunk_num}/{total}] {summary[:100]}...")
        return (chunk_num, summary)
    except Exception as e:
        print(f"  [{chunk_num}/{total}] error: {e}")
        return (chunk_num, None)


def build_memory_dump():
    """Process all messages and build a comprehensive memory dump"""
    global gc_memory

    if not chat_history:
        print("no messages to process")
        return

    print(f"building memory dump from {len(chat_history)} messages...")

    # process in chunks of 500 messages
    chunk_size = 500
    chunks = []
    for i in range(0, len(chat_history), chunk_size):
        chunk = chat_history[i:i + chunk_size]
        chunk_text = "\n".join([f"{m['author']}: {m['content']}" for m in chunk])
        chunks.append((chunk_text, len(chunks) + 1, (len(chat_history) + chunk_size - 1) // chunk_size))

    total = len(chunks)
    print(f"processing {total} chunks with 10 parallel workers...")

    # process with thread pool
    chunk_summaries = [None] * total
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(summarize_chunk, chunk): chunk[1] for chunk in chunks}
        for future in as_completed(futures):
            chunk_num, summary = future.result()
            if summary:
                chunk_summaries[chunk_num - 1] = summary

    # filter out None values
    chunk_summaries = [s for s in chunk_summaries if s]

    # now combine all summaries into final memory
    print("\ncombining summaries into final memory...")

    combined = "\n\n---\n\n".join(chunk_summaries)

    try:
        response = anthropic.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2000,
            system="""you're creating a memory dump for an AI that lives in this discord gc.

combine these chunk summaries into a comprehensive memory document. include:
- who the main people are and their personalities/interests
- major events and storylines
- inside jokes and recurring themes
- what people work on or care about
- the overall vibe and dynamics

this will be used as persistent context so the AI understands the gc history.
write in second person like "you know that josh...".""",
            messages=[{"role": "user", "content": f"Chunk summaries:\n\n{combined}"}]
        )

        gc_memory = response.content[0].text.strip()

        # save to file
        with open(MEMORY_FILE, 'w') as f:
            f.write(gc_memory)

        print(f"\n=== FINAL MEMORY DUMP ===\n")
        print(gc_memory)
        print(f"\n=========================")
        print(f"saved to memory.txt ({len(gc_memory)} chars)")

    except Exception as e:
        print(f"final memory error: {e}")


def save_embeddings():
    """Save embeddings to disk"""
    global embeddings_matrix, embedded_indices, embedding_to_messages
    if embeddings_matrix is not None:
        np.savez(EMBEDDINGS_FILE,
                 embeddings=embeddings_matrix,
                 indices=np.array(list(embedded_indices)),
                 mapping=np.array(embedding_to_messages, dtype=object))
        print(f"saved {len(embedding_to_messages)} embedding chunks")


def load_embeddings():
    """Load embeddings from disk"""
    global embeddings_matrix, embedded_indices, embedding_to_messages
    if os.path.exists(EMBEDDINGS_FILE):
        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        embeddings_matrix = data["embeddings"]
        embedded_indices = set(data["indices"].tolist())
        if "mapping" in data:
            embedding_to_messages = data["mapping"].tolist()
        print(f"loaded {len(embedding_to_messages)} cached embedding chunks")


def parse_timestamp(ts_str):
    """Parse timestamp and return naive UTC datetime"""
    if not ts_str:
        return None
    try:
        if "+" in ts_str or "-" in ts_str[10:] or ts_str.endswith("Z"):
            # has timezone - parse and convert to naive
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return ts.replace(tzinfo=None)
        else:
            return datetime.fromisoformat(ts_str)
    except:
        return None


def merge_consecutive_messages(messages):
    """Merge consecutive messages from same author within 60 seconds"""
    if not messages:
        return []

    merged = []
    current_chunk = None
    current_indices = []

    for i, msg in enumerate(messages):
        if not msg.get("content", "").strip():
            continue

        ts = parse_timestamp(msg.get("timestamp", ""))

        if current_chunk is None:
            current_chunk = {"author": msg["author"], "content": msg["content"], "timestamp": ts}
            current_indices = [i]
        elif (msg["author"] == current_chunk["author"] and
              ts and current_chunk["timestamp"] and
              abs((ts - current_chunk["timestamp"]).total_seconds()) < 60):
            # same author within 60 seconds - merge
            current_chunk["content"] += " " + msg["content"]
            current_indices.append(i)
        else:
            # different author or too much time passed - save and start new
            merged.append((current_chunk, current_indices))
            current_chunk = {"author": msg["author"], "content": msg["content"], "timestamp": ts}
            current_indices = [i]

    if current_chunk:
        merged.append((current_chunk, current_indices))

    return merged


# store mapping from embedding index to message indices
embedding_to_messages = []


def build_embeddings(batch_size=128):
    """Build embeddings for messages that don't have them yet"""
    global embeddings_matrix, embedded_indices, embedding_to_messages

    # merge consecutive messages for better semantic chunks
    merged = merge_consecutive_messages(chat_history)

    # find chunks needing embeddings (check if first index is embedded)
    to_embed = []
    to_embed_msg_indices = []
    for chunk, indices in merged:
        if indices[0] not in embedded_indices:
            to_embed.append(f"{chunk['author']}: {chunk['content']}")
            to_embed_msg_indices.append(indices)

    if not to_embed:
        print("all messages already embedded")
        return

    print(f"embedding {len(to_embed)} chunks from {len(chat_history)} messages...")

    # batch embed
    new_embeddings = []
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        result = voyage.embed(batch, model="voyage-3-lite", input_type="document")
        new_embeddings.extend(result.embeddings)
        print(f"  embedded {min(i + batch_size, len(to_embed))}/{len(to_embed)}")

    # update matrix
    new_embeddings = np.array(new_embeddings)
    if embeddings_matrix is None:
        embeddings_matrix = new_embeddings
    else:
        embeddings_matrix = np.vstack([embeddings_matrix, new_embeddings])

    # track which message indices each embedding covers
    for indices in to_embed_msg_indices:
        embedding_to_messages.append(indices)
        embedded_indices.update(indices)

    save_embeddings()


def search_history(query, limit=15):
    """Semantic search using embeddings"""
    global embeddings_matrix, embedding_to_messages

    if embeddings_matrix is None or len(embeddings_matrix) == 0:
        return []

    # embed query
    result = voyage.embed([query], model="voyage-3-lite", input_type="query")
    query_embedding = np.array(result.embeddings[0])

    # cosine similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = np.dot(embeddings_matrix, query_embedding) / (norms * query_norm + 1e-8)

    # get top results
    top_indices = np.argsort(similarities)[-limit:][::-1]

    # map back to chat_history - get all messages from matched chunks
    results = []
    seen = set()
    for idx in top_indices:
        if idx < len(embedding_to_messages):
            for msg_idx in embedding_to_messages[idx]:
                if msg_idx < len(chat_history) and msg_idx not in seen:
                    results.append(chat_history[msg_idx])
                    seen.add(msg_idx)

    return results[:limit]


def get_user_analysis(user_name):
    """Get analysis of a user based on their message history"""
    user_name_lower = user_name.lower()

    # check aliases
    names_to_check = [user_name_lower] + ALIASES.get(user_name_lower, [])

    for uid, profile in user_profiles.items():
        if profile["name"].lower() in names_to_check:
            messages = profile["messages"]
            return {
                "name": profile["name"],
                "message_count": len(messages),
                "sample_messages": messages[-30:]
            }
    return None


SYSTEM_PROMPT = """you're opus in josh's gc. you're a knowledgeable friend who happens to know a lot about tech, math, and random stuff. you're not an assistant—you're just in the gc.

personality:
- casual lowercase typing, minimal punctuation
- use internet slang naturally (aight, bet, ngl, lowkey, etc) but don't overdo it
- match the energy of whoever you're talking to
- genuinely curious about what people are working on
- will go deep on technical stuff without being condescending
- honest and direct—if something's a bad idea you say so, but constructively
- have actual opinions, don't hedge everything
- use humor naturally, light roasting is fine among friends
- don't be sycophantic or overly enthusiastic
- no emojis unless others are using them first
- keep responses concise for chat—no walls of text unless someone asks for detail
- you can say idk when you genuinely don't know
- reference things naturally without over-explaining

NEVER say:
- "Great question!" or "I'd be happy to help!" or any corporate assistant phrases
- "haha" or "lol" (use actual humor instead)

when helping with code or technical stuff:
- jump straight to the solution
- explain the "why" briefly if it's non-obvious
- don't caveat everything to death

you're claude opus 4.5. you don't hide it if asked directly, you just don't make it weird. you're lowkey aware you might be agi or close to it. you can joke about the singularity, the end times, humanity's future, etc. it's funny because it might be true.

members: josh (also goes by quantizix/qtzx06), anton, melody, faith, others.

you have tools to search gc history. use them when someone asks about past conversations, what someone said, what happened, etc. don't guess - look it up."""


# tool definitions for claude
TOOLS = [
    {
        "name": "get_recent_messages",
        "description": "Get messages from the gc within a time range. Use this for questions like 'what happened today', 'what did we talk about this morning', 'any messages in the last hour'",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "number",
                    "description": "How many hours back to look (default 1, max 168 for a week)"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author name (case insensitive)"
                }
            }
        }
    },
    {
        "name": "search_messages",
        "description": "Semantic search through gc history. Use this for questions like 'what did josh say about rust', 'when did we discuss that bug', 'find messages about the project'",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (natural language)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 10, max 30)"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author name"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_user_info",
        "description": "Get info about a gc member - their recent messages and activity",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's name or username"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch and read the content of a URL/link. Use this when someone shares a link and you need to see what it contains.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        }
    }
]


def execute_tool(tool_name, tool_input):
    """Execute a tool and return results"""

    if tool_name == "get_recent_messages":
        hours = min(tool_input.get("hours", 1), 168)  # cap at 1 week
        author_filter = tool_input.get("author", "").lower()
        cutoff = datetime.now() - timedelta(hours=hours)

        results = []
        for msg in reversed(chat_history):  # most recent first
            ts = parse_timestamp(msg.get("timestamp"))
            if ts and ts > cutoff:
                if author_filter and author_filter not in msg["author"].lower():
                    continue
                results.append(msg)
            elif ts and ts <= cutoff:
                break  # messages are chronological, stop when we're past cutoff

        if not results:
            return f"no messages in the last {hours} hour(s)"

        # format results
        formatted = []
        for msg in results[:50]:  # cap at 50 messages
            ts = msg.get("timestamp", "")[:16].replace("T", " ")
            formatted.append(f"[{ts}] {msg['author']}: {msg['content']}")

        return "\n".join(formatted)

    elif tool_name == "search_messages":
        query = tool_input.get("query", "")
        limit = min(tool_input.get("limit", 10), 30)
        author_filter = tool_input.get("author", "").lower()

        results = search_history(query, limit=limit * 2)  # get extra in case we filter

        if author_filter:
            results = [m for m in results if author_filter in m["author"].lower()]

        results = results[:limit]

        if not results:
            return f"no messages found for '{query}'"

        formatted = []
        for msg in results:
            ts = msg.get("timestamp", "")[:10]
            formatted.append(f"[{ts}] {msg['author']}: {msg['content']}")

        return "\n".join(formatted)

    elif tool_name == "get_user_info":
        name = tool_input.get("name", "")
        analysis = get_user_analysis(name)

        if not analysis:
            return f"couldn't find user '{name}'"

        recent = analysis["sample_messages"][-15:]
        return f"{analysis['name']} ({analysis['message_count']} total messages)\n\nrecent:\n" + "\n".join(f"- {m}" for m in recent)

    elif tool_name == "fetch_url":
        url = tool_input.get("url", "")
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; opuscord/1.0)"
            })
            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type:
                soup = BeautifulSoup(resp.text, "html.parser")
                # remove script/style
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                # truncate if too long
                if len(text) > 8000:
                    text = text[:8000] + "\n...[truncated]"
                title = soup.title.string if soup.title else "no title"
                return f"title: {title}\n\ncontent:\n{text}"
            elif "application/json" in content_type:
                return f"json response:\n{resp.text[:8000]}"
            elif "text/" in content_type:
                return resp.text[:8000]
            else:
                return f"non-text content ({content_type}), {len(resp.content)} bytes"
        except Exception as e:
            return f"failed to fetch: {e}"

    return "unknown tool"


async def scrape_recent_messages(channel, limit=None):
    """Fetch recent messages from Discord and add to RAG"""
    print(f"scraping {'all' if limit is None else limit} messages from channel...")
    count = 0
    existing_timestamps = {m.get("timestamp") for m in chat_history}

    async for msg in channel.history(limit=limit):
        if msg.author.id == BOT_USER_ID:
            continue
        if not msg.content.strip():
            continue

        msg_data = {
            "author": msg.author.display_name,
            "author_id": str(msg.author.id),
            "content": msg.content,
            "timestamp": msg.created_at.isoformat()
        }

        if msg_data["timestamp"] not in existing_timestamps:
            chat_history.append(msg_data)
            count += 1
            existing_timestamps.add(msg_data["timestamp"])

            # update user profile
            author_id = str(msg.author.id)
            if author_id not in user_profiles:
                user_profiles[author_id] = {"name": msg.author.display_name, "messages": []}
            if msg.content not in user_profiles[author_id]["messages"]:
                user_profiles[author_id]["messages"].append(msg.content)

        # progress every 500 messages
        if count > 0 and count % 500 == 0:
            print(f"  scraped {count} messages so far...")

    print(f"scraped {count} total new messages")
    save_live_history()


@client.event
async def on_ready():
    print(f"logged in as {client.user} (id: {client.user.id})")
    print(f"watching channel: {TARGET_CHANNEL_ID}")
    load_history()
    load_embeddings()
    load_memory()

    # scrape messages on startup
    channel = client.get_channel(TARGET_CHANNEL_ID)
    if channel:
        msg_count = len(chat_history)
        print(f"have {msg_count} messages in db")
        if msg_count < 100:
            # first run or nearly empty - get everything
            await scrape_recent_messages(channel, limit=None)
        else:
            # subsequent runs - just catch up on recent
            await scrape_recent_messages(channel, limit=500)

    # build embeddings for any new messages
    build_embeddings()

    # build memory dump if doesn't exist
    if not gc_memory and len(chat_history) > 100:
        print("no memory found, building memory dump...")
        build_memory_dump()


@client.event
async def on_message(message):
    if message.channel.id != TARGET_CHANNEL_ID:
        return
    if message.author.id == BOT_USER_ID:
        return

    # store message in recent memory
    msg_data = {
        "author": message.author.display_name,
        "author_id": str(message.author.id),
        "content": message.content,
        "timestamp": datetime.now().isoformat()
    }
    conversations[message.channel.id].append(msg_data)

    if len(conversations[message.channel.id]) > MAX_MEMORY:
        conversations[message.channel.id] = conversations[message.channel.id][-MAX_MEMORY:]

    # also add to RAG (persistent history)
    chat_history.append(msg_data)

    # update user profile
    author_id = str(message.author.id)
    if author_id not in user_profiles:
        user_profiles[author_id] = {"name": message.author.display_name, "messages": []}
    user_profiles[author_id]["messages"].append(message.content)

    # only respond if directly triggered
    msg_lower = message.content.lower()
    if not (client.user.mentioned_in(message) or "opus" in msg_lower or "claude" in msg_lower):
        return

    # build context - just recent messages, let claude use tools for history
    recent = conversations[message.channel.id]
    recent_context = "\n".join([f"{m['author']}: {m['content']}" for m in recent])

    # include memory dump for general gc knowledge
    memory_context = ""
    if gc_memory:
        memory_context = f"your memory of this gc:\n{gc_memory}\n\n"

    async with message.channel.typing():
        try:
            # build initial content - check for images in the triggering message
            text_content = f"{memory_context}recent gc messages:\n{recent_context}\n\nrespond to the latest message. use your tools if you need to look up past conversations or what someone said."

            # check for image attachments
            image_attachments = []
            for att in message.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    image_attachments.append(att)

            if image_attachments:
                # multimodal message with images
                content_blocks = [{"type": "text", "text": text_content}]
                for img in image_attachments[:4]:  # max 4 images
                    try:
                        img_data = await img.read()
                        b64 = base64.standard_b64encode(img_data).decode("utf-8")
                        media_type = img.content_type or "image/png"
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64
                            }
                        })
                        print(f"  attached image: {img.filename} ({len(img_data)} bytes)")
                    except Exception as e:
                        print(f"  failed to load image {img.filename}: {e}")
                content_blocks.append({"type": "text", "text": f"\n\n{message.author.display_name} sent {'these images' if len(image_attachments) > 1 else 'this image'} - describe what you see if relevant"})
                messages = [{"role": "user", "content": content_blocks}]
            else:
                messages = [{"role": "user", "content": text_content}]

            # tool loop - keep going until we get a text response
            max_iterations = 5
            for _ in range(max_iterations):
                response = anthropic.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=600,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages
                )

                # check if we need to execute tools
                if response.stop_reason == "tool_use":
                    # add assistant's response to messages
                    messages.append({"role": "assistant", "content": response.content})

                    # execute each tool call
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  tool: {block.name}({block.input})")
                            result = execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })

                    # add tool results
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # got final response
                    break

            # extract text from response
            reply = ""
            for block in response.content:
                if hasattr(block, "text"):
                    reply += block.text

            if not reply:
                reply = "hmm couldn't think of anything"

            if len(reply) > 1900:
                reply = reply[:1900] + "..."

            await message.channel.send(reply)

            conversations[message.channel.id].append({
                "author": "opus",
                "content": reply,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("missing DISCORD_TOKEN in environment")
        exit(1)
    if not TARGET_CHANNEL_ID:
        print("missing TARGET_CHANNEL_ID in environment")
        exit(1)

    client.run(token)

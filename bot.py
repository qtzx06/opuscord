import discord
import os
import json
import atexit
import re
import asyncio
import numpy as np
import voyageai
import httpx
from dotenv import load_dotenv
from anthropic import Anthropic
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import base64
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from functools import partial

load_dotenv()

LIVE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "live_history.json")
EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddings.npz")
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.txt")
SESSION_FILE = os.path.join(os.path.dirname(__file__), "session.json")

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

# acronym expansions for fuzzy matching
ACRONYMS = {
    "loml": ["love of my life", "loml"],
    "goat": ["greatest of all time", "goat"],
    "imo": ["in my opinion", "imo"],
    "imho": ["in my humble opinion", "imho"],
    "tbh": ["to be honest", "tbh"],
    "ngl": ["not gonna lie", "ngl"],
    "idk": ["i don't know", "idk"],
    "rn": ["right now", "rn"],
    "fr": ["for real", "fr"],
    "ong": ["on god", "ong"],
    "istg": ["i swear to god", "istg"],
    "wyd": ["what you doing", "wyd"],
    "hbu": ["how about you", "hbu"],
    "omw": ["on my way", "omw"],
    "lmk": ["let me know", "lmk"],
    "brb": ["be right back", "brb"],
    "gtg": ["got to go", "gtg"],
    "iirc": ["if i recall correctly", "iirc"],
    "afaik": ["as far as i know", "afaik"],
    "fwiw": ["for what it's worth", "fwiw"],
    "tfw": ["that feeling when", "tfw"],
    "mfw": ["my face when", "mfw"],
    "smh": ["shaking my head", "smh"],
    "icl": ["i can't lie", "icl"],
    "lowkey": ["lowkey", "low key"],
    "highkey": ["highkey", "high key"],
    "puh": ["puh", "pussy"],
    "fuh": ["fuh", "fuck"],
}

# BM25 index for keyword search
bm25_index = None
bm25_corpus = []  # tokenized messages
bm25_msg_indices = []  # maps bm25 index to chat_history index


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

# session tracking
last_session_end = None
session_context = ""  # summary of what happened since last session


def load_memory():
    """Load the memory dump if it exists"""
    global gc_memory
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            gc_memory = f.read()
        print(f"loaded memory ({len(gc_memory)} chars)")


def save_session():
    """Save session end time on exit"""
    try:
        with open(SESSION_FILE, 'w') as f:
            json.dump({
                "last_end": datetime.now().isoformat(),
                "message_count": len(chat_history)
            }, f)
        print("saved session state")
    except Exception as e:
        print(f"failed to save session: {e}")


def load_session():
    """Load last session end time"""
    global last_session_end
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                data = json.load(f)
                last_session_end = datetime.fromisoformat(data.get("last_end", ""))
                print(f"last session ended: {last_session_end}")
        except Exception as e:
            print(f"couldn't load session: {e}")


def build_session_context():
    """Summarize what happened since last session"""
    global session_context, last_session_end

    if not last_session_end or not chat_history:
        session_context = ""
        return

    # find messages since last session
    missed_messages = []
    for msg in chat_history:
        ts = parse_timestamp(msg.get("timestamp"))
        if ts and ts > last_session_end:
            missed_messages.append(msg)

    if not missed_messages:
        session_context = ""
        print("no messages missed since last session")
        return

    print(f"building session context for {len(missed_messages)} missed messages...")

    # if only a few messages, just include them directly
    if len(missed_messages) <= 30:
        lines = []
        for msg in missed_messages:
            ts = msg.get("timestamp", "")[:16].replace("T", " ")
            lines.append(f"[{ts}] {msg['author']}: {msg['content'][:200]}")
        session_context = "messages since you were last online:\n" + "\n".join(lines)
        print(f"session context: {len(missed_messages)} messages included directly")
        return

    # for more messages, summarize with claude
    chunk_text = "\n".join([f"{m['author']}: {m['content']}" for m in missed_messages[-200:]])  # last 200

    try:
        response = anthropic.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            system="""summarize what happened in this discord gc since the AI was last online. be concise but capture:
- major topics discussed
- any drama or notable moments
- questions that might have been directed at "opus" or "claude"
- anything the AI should know about before jumping back in

write in second person like "while you were offline...".""",
            messages=[{"role": "user", "content": chunk_text}]
        )

        session_context = response.content[0].text.strip()
        print(f"session context built ({len(session_context)} chars)")

    except Exception as e:
        print(f"failed to build session context: {e}")
        # fallback to just recent messages
        lines = []
        for msg in missed_messages[-20:]:
            lines.append(f"{msg['author']}: {msg['content'][:100]}")
        session_context = "recent messages while you were offline:\n" + "\n".join(lines)


# save session on exit
atexit.register(save_session)


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


def tokenize(text):
    """Simple tokenizer for BM25"""
    return re.findall(r'\b\w+\b', text.lower())


def expand_query(query):
    """Expand acronyms in query for better matching"""
    query_lower = query.lower()
    expanded_terms = [query_lower]

    for acronym, expansions in ACRONYMS.items():
        if acronym in query_lower:
            for exp in expansions:
                expanded_terms.append(query_lower.replace(acronym, exp))

    return expanded_terms


def build_bm25_index():
    """Build BM25 index for keyword search"""
    global bm25_index, bm25_corpus, bm25_msg_indices

    if not chat_history:
        return

    print(f"building BM25 index for {len(chat_history)} messages...")

    bm25_corpus = []
    bm25_msg_indices = []

    for i, msg in enumerate(chat_history):
        content = msg.get("content", "")
        if content.strip():
            # tokenize with author for context
            text = f"{msg['author']} {content}"
            tokens = tokenize(text)
            bm25_corpus.append(tokens)
            bm25_msg_indices.append(i)

    if bm25_corpus:
        bm25_index = BM25Okapi(bm25_corpus)
        print(f"BM25 index built with {len(bm25_corpus)} documents")


def keyword_search(query, limit=20, author_filter=None):
    """Exact keyword search using BM25"""
    global bm25_index, bm25_corpus, bm25_msg_indices

    if bm25_index is None:
        build_bm25_index()

    if bm25_index is None:
        return []

    # expand query for acronyms
    expanded = expand_query(query)

    # search with all expanded queries and combine results
    all_scores = np.zeros(len(bm25_corpus))
    for q in expanded:
        tokens = tokenize(q)
        scores = bm25_index.get_scores(tokens)
        all_scores = np.maximum(all_scores, scores)  # take max score

    # get top results
    top_indices = np.argsort(all_scores)[-limit * 2:][::-1]

    results = []
    for idx in top_indices:
        if all_scores[idx] > 0:
            msg_idx = bm25_msg_indices[idx]
            msg = chat_history[msg_idx]

            if author_filter:
                names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
                if msg["author"].lower() not in names_to_check:
                    continue

            results.append((msg, all_scores[idx]))

    return results[:limit]


def fuzzy_search(query, limit=20, author_filter=None, threshold=70):
    """Fuzzy matching search for typos and variations"""
    if not chat_history:
        return []

    # expand query for acronyms
    expanded = expand_query(query)

    results = []
    for i, msg in enumerate(chat_history):
        content = msg.get("content", "").lower()
        if not content:
            continue

        if author_filter:
            names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
            if msg["author"].lower() not in names_to_check:
                continue

        # check fuzzy match against all expanded queries
        best_score = 0
        for q in expanded:
            # partial ratio handles substring matching well
            score = fuzz.partial_ratio(q.lower(), content)
            best_score = max(best_score, score)

        if best_score >= threshold:
            results.append((msg, best_score))

    # sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def regex_search(pattern, limit=50, author_filter=None):
    """Regex pattern search through messages"""
    if not chat_history:
        return []

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return []

    results = []
    for msg in chat_history:
        content = msg.get("content", "")
        if not content:
            continue

        if author_filter:
            names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
            if msg["author"].lower() not in names_to_check:
                continue

        if regex.search(content):
            results.append(msg)

    return results[:limit]


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


def search_history(query, limit=15, mode="hybrid"):
    """
    Search using multiple strategies:
    - hybrid: combines semantic + BM25 + fuzzy (default)
    - semantic: voyage embeddings only
    - keyword: BM25 only
    - fuzzy: fuzzy matching only
    """
    global embeddings_matrix, embedding_to_messages

    if mode == "keyword":
        results = keyword_search(query, limit=limit)
        return [r[0] for r in results]

    if mode == "fuzzy":
        results = fuzzy_search(query, limit=limit)
        return [r[0] for r in results]

    # semantic search
    semantic_results = []
    if embeddings_matrix is not None and len(embeddings_matrix) > 0:
        result = voyage.embed([query], model="voyage-3-lite", input_type="query")
        query_embedding = np.array(result.embeddings[0])

        norms = np.linalg.norm(embeddings_matrix, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = np.dot(embeddings_matrix, query_embedding) / (norms * query_norm + 1e-8)

        top_indices = np.argsort(similarities)[-limit * 2:][::-1]

        for idx in top_indices:
            if idx < len(embedding_to_messages):
                for msg_idx in embedding_to_messages[idx]:
                    if msg_idx < len(chat_history):
                        score = float(similarities[idx])
                        semantic_results.append((chat_history[msg_idx], score, msg_idx))

    if mode == "semantic":
        seen = set()
        results = []
        for msg, score, idx in semantic_results:
            if idx not in seen:
                results.append(msg)
                seen.add(idx)
        return results[:limit]

    # hybrid mode: combine all search methods
    # get BM25 results
    bm25_results = keyword_search(query, limit=limit * 2)

    # get fuzzy results (for acronyms and typos)
    fuzzy_results = fuzzy_search(query, limit=limit * 2, threshold=80)

    # combine scores by message index
    combined_scores = {}

    # weight: semantic 0.4, bm25 0.4, fuzzy 0.2
    for msg, score, idx in semantic_results:
        if idx not in combined_scores:
            combined_scores[idx] = {"msg": msg, "semantic": 0, "bm25": 0, "fuzzy": 0}
        combined_scores[idx]["semantic"] = max(combined_scores[idx]["semantic"], score)

    # normalize bm25 scores
    if bm25_results:
        max_bm25 = max(r[1] for r in bm25_results)
        for msg, score in bm25_results:
            idx = chat_history.index(msg) if msg in chat_history else None
            if idx is not None:
                if idx not in combined_scores:
                    combined_scores[idx] = {"msg": msg, "semantic": 0, "bm25": 0, "fuzzy": 0}
                combined_scores[idx]["bm25"] = score / max_bm25 if max_bm25 > 0 else 0

    # normalize fuzzy scores (already 0-100)
    for msg, score in fuzzy_results:
        idx = chat_history.index(msg) if msg in chat_history else None
        if idx is not None:
            if idx not in combined_scores:
                combined_scores[idx] = {"msg": msg, "semantic": 0, "bm25": 0, "fuzzy": 0}
            combined_scores[idx]["fuzzy"] = score / 100.0

    # compute final scores
    final_results = []
    for idx, data in combined_scores.items():
        final_score = (0.4 * data["semantic"] +
                      0.4 * data["bm25"] +
                      0.2 * data["fuzzy"])
        if final_score > 0:
            final_results.append((data["msg"], final_score))

    # sort by final score
    final_results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in final_results[:limit]]


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
        "description": "Hybrid search through gc history combining semantic, keyword, and fuzzy matching. Use this for natural language queries like 'what did josh say about rust'. For acronyms like 'loml', 'ngl', etc it will auto-expand and fuzzy match.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (natural language or keywords)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 15, max 50)"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author name"
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "keyword", "fuzzy"],
                    "description": "Search mode: hybrid (default, best for most queries), semantic (meaning-based), keyword (exact BM25), fuzzy (typo-tolerant)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "keyword_search",
        "description": "Exact keyword/phrase search using BM25. Best for finding specific words, phrases, acronyms like 'loml', 'fr', 'ngl'. Automatically expands common acronyms.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Exact keyword or phrase to search for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20, max 50)"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "regex_search",
        "description": "Search using regex patterns. Use for complex pattern matching like finding URLs, mentions, specific formats.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern (case insensitive)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 30, max 100)"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "count_matches",
        "description": "Count how many times a word/phrase appears, optionally by author. Use for questions like 'how many times did josh say loml', 'who says fr the most'",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Word or phrase to count"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: only count for this author"
                },
                "by_author": {
                    "type": "boolean",
                    "description": "If true, return counts broken down by author"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_by_date",
        "description": "Search messages within a specific date range",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD format)"
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD format)"
                },
                "query": {
                    "type": "string",
                    "description": "Optional: filter by keyword"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 50)"
                }
            },
            "required": ["start_date", "end_date"]
        }
    },
    {
        "name": "get_message_stats",
        "description": "Get statistics about gc activity - message counts, top authors, active times, word frequencies",
        "input_schema": {
            "type": "object",
            "properties": {
                "stat_type": {
                    "type": "string",
                    "enum": ["overview", "top_authors", "word_freq", "activity_by_hour", "activity_by_day"],
                    "description": "Type of stats to get"
                },
                "author": {
                    "type": "string",
                    "description": "Optional: get stats for specific author"
                },
                "days": {
                    "type": "integer",
                    "description": "Optional: limit to last N days"
                }
            },
            "required": ["stat_type"]
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
    },
    {
        "name": "view_image",
        "description": "View and describe an image using Claude vision. If no URL given, automatically finds the most recent image posted. Reads text, explains memes/screenshots, describes what's happening.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Image URL (optional - if not provided, uses most recent image in chat)"
                },
                "question": {
                    "type": "string",
                    "description": "Specific question about the image"
                },
                "detail": {
                    "type": "string",
                    "enum": ["brief", "normal", "detailed"],
                    "description": "How much detail: brief (1 sentence), normal (default), detailed (thorough analysis)"
                }
            }
        }
    },
    {
        "name": "get_attachments",
        "description": "Find messages with attachments (images, files). Use to find what images someone posted, recent screenshots, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Optional: filter by author"
                },
                "content_type": {
                    "type": "string",
                    "description": "Optional: filter by type (image, video, audio, file)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20)"
                },
                "hours": {
                    "type": "number",
                    "description": "Optional: only look back N hours"
                }
            }
        }
    },
    {
        "name": "get_reactions",
        "description": "Get reactions on messages. Find what messages got the most reactions, who reacted to what, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Optional: messages by this author"
                },
                "emoji": {
                    "type": "string",
                    "description": "Optional: filter by specific emoji"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20)"
                }
            }
        }
    },
    {
        "name": "get_voice_channel",
        "description": "Check who's currently in voice channels. See who's in vc, how many people, which channels are active.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel_name": {
                    "type": "string",
                    "description": "Optional: specific voice channel name to check"
                }
            }
        }
    },
    {
        "name": "get_reply_thread",
        "description": "Get the reply chain/thread context for a message. Useful for understanding conversations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The message ID to get thread context for"
                },
                "query": {
                    "type": "string",
                    "description": "Or search for a message to get its thread"
                }
            }
        }
    }
]


def _sync_execute_tool(tool_name, tool_input):
    """Synchronous tool execution (runs in thread pool)"""

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
        limit = min(tool_input.get("limit", 15), 50)
        author_filter = tool_input.get("author", "").lower()
        mode = tool_input.get("mode", "hybrid")

        results = search_history(query, limit=limit * 2, mode=mode)

        if author_filter:
            names_to_check = [author_filter] + ALIASES.get(author_filter, [])
            results = [m for m in results if m["author"].lower() in names_to_check]

        results = results[:limit]

        if not results:
            return f"no messages found for '{query}'"

        formatted = []
        for msg in results:
            ts = msg.get("timestamp", "")[:10]
            # include reply context if available
            reply_info = ""
            if msg.get("reply_to"):
                reply_info = f" (replying to {msg['reply_to']['author']})"
            # include attachment info
            attach_info = ""
            if msg.get("attachments"):
                attach_info = f" [+{len(msg['attachments'])} attachment(s)]"
            formatted.append(f"[{ts}] {msg['author']}{reply_info}: {msg['content']}{attach_info}")

        return "\n".join(formatted)

    elif tool_name == "keyword_search":
        query = tool_input.get("query", "")
        limit = min(tool_input.get("limit", 20), 50)
        author_filter = tool_input.get("author")

        results = keyword_search(query, limit=limit, author_filter=author_filter)

        if not results:
            return f"no exact matches for '{query}'"

        formatted = []
        for msg, score in results:
            ts = msg.get("timestamp", "")[:10]
            formatted.append(f"[{ts}] {msg['author']}: {msg['content']}")

        return "\n".join(formatted)

    elif tool_name == "regex_search":
        pattern = tool_input.get("pattern", "")
        limit = min(tool_input.get("limit", 30), 100)
        author_filter = tool_input.get("author")

        results = regex_search(pattern, limit=limit, author_filter=author_filter)

        if not results:
            return f"no matches for pattern '{pattern}'"

        formatted = []
        for msg in results:
            ts = msg.get("timestamp", "")[:10]
            formatted.append(f"[{ts}] {msg['author']}: {msg['content']}")

        return "\n".join(formatted)

    elif tool_name == "count_matches":
        query = tool_input.get("query", "").lower()
        author_filter = tool_input.get("author")
        by_author = tool_input.get("by_author", False)

        # expand acronyms
        expanded = expand_query(query)

        if by_author:
            counts = Counter()
            for msg in chat_history:
                content = msg.get("content", "").lower()
                for q in expanded:
                    if q in content:
                        counts[msg["author"]] += content.count(q)
                        break

            if not counts:
                return f"no matches for '{query}'"

            sorted_counts = counts.most_common(20)
            lines = [f"'{query}' usage by author:"]
            for author, count in sorted_counts:
                lines.append(f"  {author}: {count}")
            lines.append(f"\ntotal: {sum(counts.values())}")
            return "\n".join(lines)
        else:
            total = 0
            for msg in chat_history:
                if author_filter:
                    names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
                    if msg["author"].lower() not in names_to_check:
                        continue

                content = msg.get("content", "").lower()
                for q in expanded:
                    if q in content:
                        total += content.count(q)
                        break

            author_str = f" by {author_filter}" if author_filter else ""
            return f"'{query}'{author_str}: {total} matches"

    elif tool_name == "search_by_date":
        start = tool_input.get("start_date", "")
        end = tool_input.get("end_date", "")
        query = tool_input.get("query", "").lower()
        author_filter = tool_input.get("author")
        limit = min(tool_input.get("limit", 50), 100)

        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end + "T23:59:59")
        except:
            return "invalid date format, use YYYY-MM-DD"

        results = []
        for msg in chat_history:
            ts = parse_timestamp(msg.get("timestamp"))
            if not ts or ts < start_dt or ts > end_dt:
                continue

            if author_filter:
                names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
                if msg["author"].lower() not in names_to_check:
                    continue

            if query and query not in msg.get("content", "").lower():
                continue

            results.append(msg)

        if not results:
            return f"no messages between {start} and {end}"

        formatted = []
        for msg in results[:limit]:
            ts = msg.get("timestamp", "")[:16].replace("T", " ")
            formatted.append(f"[{ts}] {msg['author']}: {msg['content']}")

        total = len(results)
        shown = min(total, limit)
        return f"found {total} messages ({shown} shown):\n\n" + "\n".join(formatted)

    elif tool_name == "get_message_stats":
        stat_type = tool_input.get("stat_type", "overview")
        author_filter = tool_input.get("author")
        days = tool_input.get("days")

        # filter by days if specified
        msgs = chat_history
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            msgs = [m for m in msgs if parse_timestamp(m.get("timestamp")) and parse_timestamp(m.get("timestamp")) > cutoff]

        if author_filter:
            names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
            msgs = [m for m in msgs if m["author"].lower() in names_to_check]

        if not msgs:
            return "no messages found"

        if stat_type == "overview":
            authors = Counter(m["author"] for m in msgs)
            total = len(msgs)
            unique_authors = len(authors)
            top_3 = authors.most_common(3)
            return f"total messages: {total}\nunique authors: {unique_authors}\ntop contributors: {', '.join(f'{a} ({c})' for a,c in top_3)}"

        elif stat_type == "top_authors":
            authors = Counter(m["author"] for m in msgs)
            lines = ["message counts by author:"]
            for author, count in authors.most_common(15):
                lines.append(f"  {author}: {count}")
            return "\n".join(lines)

        elif stat_type == "word_freq":
            words = Counter()
            for msg in msgs:
                tokens = tokenize(msg.get("content", ""))
                words.update(t for t in tokens if len(t) > 2)

            lines = ["most common words:"]
            for word, count in words.most_common(30):
                lines.append(f"  {word}: {count}")
            return "\n".join(lines)

        elif stat_type == "activity_by_hour":
            hours = Counter()
            for msg in msgs:
                ts = parse_timestamp(msg.get("timestamp"))
                if ts:
                    hours[ts.hour] += 1

            lines = ["messages by hour:"]
            for hour in range(24):
                count = hours.get(hour, 0)
                bar = "█" * (count // 10) if count else ""
                lines.append(f"  {hour:02d}:00 - {count:4d} {bar}")
            return "\n".join(lines)

        elif stat_type == "activity_by_day":
            days_count = Counter()
            for msg in msgs:
                ts = parse_timestamp(msg.get("timestamp"))
                if ts:
                    days_count[ts.strftime("%A")] += 1

            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            lines = ["messages by day:"]
            for day in day_order:
                count = days_count.get(day, 0)
                bar = "█" * (count // 20) if count else ""
                lines.append(f"  {day:9s} - {count:4d} {bar}")
            return "\n".join(lines)

        return "unknown stat type"

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

    elif tool_name == "view_image":
        url = tool_input.get("url", "")
        question = tool_input.get("question", "")
        detail_level = tool_input.get("detail", "normal")  # brief, normal, detailed

        # if no URL provided, try to find the most recent image
        if not url:
            for msg in reversed(chat_history):
                if msg.get("attachments"):
                    for att in msg["attachments"]:
                        if att.get("content_type", "").startswith("image/"):
                            url = att.get("url")
                            break
                if url:
                    break

        if not url:
            return "no image URL provided and no recent images found"

        try:
            # fetch the image
            resp = httpx.get(url, follow_redirects=True, timeout=20, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })

            if resp.status_code != 200:
                return f"failed to fetch image: HTTP {resp.status_code}"

            content_type = resp.headers.get("content-type", "image/png")

            # handle content type with charset
            if ";" in content_type:
                content_type = content_type.split(";")[0].strip()

            if not content_type.startswith("image/"):
                return f"not an image: {content_type}"

            # check file size (skip if > 20MB)
            if len(resp.content) > 20 * 1024 * 1024:
                return "image too large (>20MB)"

            # encode as base64
            img_b64 = base64.standard_b64encode(resp.content).decode("utf-8")

            # build the prompt based on detail level and question
            if question:
                prompt = question
            elif detail_level == "brief":
                prompt = "describe this image in one sentence. be direct."
            elif detail_level == "detailed":
                prompt = """analyze this image thoroughly:
1. what's shown (main subject, setting, context)
2. any text visible (read it out)
3. notable details or anything interesting
4. if it's a screenshot: what app/site, what's happening
5. if it's a meme: explain the joke
be comprehensive but don't be verbose."""
            else:
                prompt = """describe what you see in this image. if there's text, read it. if it's a meme or screenshot, explain what's going on. be concise but capture the important stuff."""

            # use claude to describe the image
            vision_response = anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800 if detail_level == "detailed" else 400,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": img_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            return vision_response.content[0].text

        except httpx.TimeoutException:
            return "image fetch timed out"
        except Exception as e:
            return f"failed to view image: {e}"

    elif tool_name == "get_attachments":
        author_filter = tool_input.get("author")
        content_type_filter = tool_input.get("content_type", "").lower()
        limit = min(tool_input.get("limit", 20), 50)
        hours = tool_input.get("hours")

        cutoff = None
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)

        results = []
        for msg in reversed(chat_history):
            if not msg.get("attachments"):
                continue

            if cutoff:
                ts = parse_timestamp(msg.get("timestamp"))
                if ts and ts < cutoff:
                    continue

            if author_filter:
                names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
                if msg["author"].lower() not in names_to_check:
                    continue

            for att in msg["attachments"]:
                att_type = att.get("content_type", "")
                if content_type_filter:
                    if content_type_filter == "image" and not att_type.startswith("image/"):
                        continue
                    elif content_type_filter == "video" and not att_type.startswith("video/"):
                        continue
                    elif content_type_filter == "audio" and not att_type.startswith("audio/"):
                        continue

                results.append({
                    "author": msg["author"],
                    "timestamp": msg.get("timestamp", "")[:16],
                    "content": msg.get("content", "")[:100],
                    "filename": att.get("filename"),
                    "url": att.get("url"),
                    "type": att_type
                })

            if len(results) >= limit:
                break

        if not results:
            return "no attachments found"

        formatted = []
        for r in results:
            formatted.append(f"[{r['timestamp']}] {r['author']}: {r['content']}\n  -> {r['filename']} ({r['type']})\n  -> {r['url']}")

        return "\n\n".join(formatted)

    elif tool_name == "get_reactions":
        author_filter = tool_input.get("author")
        emoji_filter = tool_input.get("emoji")
        limit = min(tool_input.get("limit", 20), 50)

        results = []
        for msg in reversed(chat_history):
            if not msg.get("reactions"):
                continue

            if author_filter:
                names_to_check = [author_filter.lower()] + ALIASES.get(author_filter.lower(), [])
                if msg["author"].lower() not in names_to_check:
                    continue

            reactions = msg["reactions"]
            if emoji_filter:
                reactions = [r for r in reactions if emoji_filter in r["emoji"]]

            if reactions:
                total_reactions = sum(r["count"] for r in reactions)
                results.append({
                    "author": msg["author"],
                    "content": msg.get("content", "")[:100],
                    "reactions": reactions,
                    "total": total_reactions,
                    "timestamp": msg.get("timestamp", "")[:10]
                })

            if len(results) >= limit:
                break

        if not results:
            return "no reactions found"

        # sort by total reactions
        results.sort(key=lambda x: x["total"], reverse=True)

        formatted = []
        for r in results:
            emoji_str = " ".join(f"{rx['emoji']}x{rx['count']}" for rx in r["reactions"])
            formatted.append(f"[{r['timestamp']}] {r['author']}: {r['content']}\n  reactions: {emoji_str}")

        return "\n\n".join(formatted)

    elif tool_name == "get_voice_channel":
        channel_name_filter = tool_input.get("channel_name", "").lower()

        # get the guild from the target channel
        channel = client.get_channel(TARGET_CHANNEL_ID)
        if not channel or not channel.guild:
            return "couldn't access guild"

        guild = channel.guild
        voice_channels = []

        for vc in guild.voice_channels:
            if channel_name_filter and channel_name_filter not in vc.name.lower():
                continue

            members = []
            for member in vc.members:
                status = ""
                if member.voice:
                    if member.voice.self_mute:
                        status += " (muted)"
                    if member.voice.self_deaf:
                        status += " (deafened)"
                    if member.voice.self_stream:
                        status += " (streaming)"
                    if member.voice.self_video:
                        status += " (video)"
                members.append(f"{member.display_name}{status}")

            if members:
                voice_channels.append({
                    "name": vc.name,
                    "members": members,
                    "count": len(members)
                })

        if not voice_channels:
            return "no one in voice channels" if not channel_name_filter else f"no one in '{channel_name_filter}'"

        formatted = []
        for vc in voice_channels:
            member_list = ", ".join(vc["members"])
            formatted.append(f"🔊 {vc['name']} ({vc['count']} people): {member_list}")

        return "\n".join(formatted)

    elif tool_name == "get_reply_thread":
        message_id = tool_input.get("message_id")
        query = tool_input.get("query")

        # find the message
        target_msg = None
        if message_id:
            for msg in chat_history:
                if msg.get("message_id") == message_id:
                    target_msg = msg
                    break
        elif query:
            # search for it
            results = search_history(query, limit=1, mode="hybrid")
            if results:
                target_msg = results[0]

        if not target_msg:
            return "message not found"

        # build thread by following reply chain
        thread = [target_msg]
        current = target_msg

        # follow replies backwards
        while current.get("reply_to"):
            reply_id = current["reply_to"].get("message_id")
            if not reply_id:
                break

            found = None
            for msg in chat_history:
                if msg.get("message_id") == reply_id:
                    found = msg
                    break

            if not found:
                # add partial info from reply_to
                thread.insert(0, {
                    "author": current["reply_to"].get("author", "unknown"),
                    "content": current["reply_to"].get("content", "[message not in history]"),
                    "timestamp": ""
                })
                break
            else:
                thread.insert(0, found)
                current = found

        formatted = []
        for i, msg in enumerate(thread):
            prefix = "└─" if i == len(thread) - 1 else "├─"
            ts = msg.get("timestamp", "")[:16].replace("T", " ")
            formatted.append(f"{prefix} [{ts}] {msg['author']}: {msg['content'][:200]}")

        return "reply thread:\n" + "\n".join(formatted)

    return "unknown tool"


async def execute_tool(tool_name, tool_input):
    """Async wrapper - runs blocking tool execution in thread pool to avoid blocking event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_sync_execute_tool, tool_name, tool_input))


async def scrape_recent_messages(channel, limit=None):
    """Fetch recent messages from Discord and add to RAG with full metadata"""
    print(f"scraping {'all' if limit is None else limit} messages from channel...")
    count = 0
    existing_timestamps = {m.get("timestamp") for m in chat_history}

    async for msg in channel.history(limit=limit):
        if msg.author.id == BOT_USER_ID:
            continue

        # skip empty messages unless they have attachments
        if not msg.content.strip() and not msg.attachments:
            continue

        # capture attachments
        attachments = []
        for att in msg.attachments:
            attachments.append({
                "filename": att.filename,
                "url": att.url,
                "content_type": att.content_type,
                "size": att.size
            })

        # capture reply context
        reply_to = None
        if msg.reference and msg.reference.resolved:
            ref = msg.reference.resolved
            reply_to = {
                "author": ref.author.display_name if hasattr(ref, 'author') else "unknown",
                "content": ref.content[:200] if hasattr(ref, 'content') else "",
                "message_id": str(ref.id) if hasattr(ref, 'id') else None
            }

        # capture reactions
        reactions = []
        for reaction in msg.reactions:
            reactions.append({
                "emoji": str(reaction.emoji),
                "count": reaction.count
            })

        msg_data = {
            "author": msg.author.display_name,
            "author_id": str(msg.author.id),
            "content": msg.content or f"[{len(attachments)} attachment(s)]",
            "timestamp": msg.created_at.isoformat(),
            "message_id": str(msg.id),
            "attachments": attachments if attachments else None,
            "reply_to": reply_to,
            "reactions": reactions if reactions else None
        }

        if msg_data["timestamp"] not in existing_timestamps:
            chat_history.append(msg_data)
            count += 1
            existing_timestamps.add(msg_data["timestamp"])

            # update user profile
            author_id = str(msg.author.id)
            if author_id not in user_profiles:
                user_profiles[author_id] = {"name": msg.author.display_name, "messages": []}
            if msg.content and msg.content not in user_profiles[author_id]["messages"]:
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

    # load session state first to know when we were last online
    load_session()

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

    # run heavy indexing operations in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    # build embeddings for any new messages
    await loop.run_in_executor(None, build_embeddings)

    # build BM25 index for keyword search
    await loop.run_in_executor(None, build_bm25_index)

    # build session context (what happened since last online)
    await loop.run_in_executor(None, build_session_context)

    # build memory dump if doesn't exist
    if not gc_memory and len(chat_history) > 100:
        print("no memory found, building memory dump...")
        await loop.run_in_executor(None, build_memory_dump)

    print("opus is ready")


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

    # include session context (what happened since last online)
    session_ctx = ""
    if session_context:
        session_ctx = f"context from while you were offline:\n{session_context}\n\n"

    async with message.channel.typing():
        try:
            # current time for context
            now = datetime.now()
            time_context = f"current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n\n"

            # build initial content - check for images in the triggering message
            text_content = f"{time_context}{memory_context}{session_ctx}recent gc messages:\n{recent_context}\n\nrespond to the latest message. use your tools if you need to look up past conversations or what someone said."

            # check for image attachments
            image_attachments = []
            for att in message.attachments:
                if att.content_type and att.content_type.startswith("image/"):
                    image_attachments.append(att)

            if image_attachments:
                # multimodal message with images
                content_blocks = [{"type": "text", "text": text_content}]
                loaded_images = 0
                for img in image_attachments[:4]:  # max 4 images
                    try:
                        img_data = await img.read()
                        # skip if too large (>10MB)
                        if len(img_data) > 10 * 1024 * 1024:
                            print(f"  skipping large image: {img.filename} ({len(img_data)} bytes)")
                            continue
                        b64 = base64.standard_b64encode(img_data).decode("utf-8")
                        media_type = img.content_type or "image/png"
                        # handle content type with charset
                        if ";" in media_type:
                            media_type = media_type.split(";")[0].strip()
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64
                            }
                        })
                        loaded_images += 1
                        print(f"  attached image: {img.filename} ({len(img_data)} bytes)")
                    except Exception as e:
                        print(f"  failed to load image {img.filename}: {e}")

                if loaded_images > 0:
                    # add context about the image(s)
                    img_prompt = f"\n\n[{message.author.display_name} attached {'these ' + str(loaded_images) + ' images' if loaded_images > 1 else 'this image'} to their message. you can see {'them' if loaded_images > 1 else 'it'} above. if they're asking about the image or it's relevant, describe/react to what you see. read any text in the image. if it's a meme, get the joke.]"
                    content_blocks.append({"type": "text", "text": img_prompt})
                    messages = [{"role": "user", "content": content_blocks}]
                else:
                    # all images failed to load
                    messages = [{"role": "user", "content": text_content}]
            else:
                messages = [{"role": "user", "content": text_content}]

            # tool loop - keep going until we get a text response
            max_iterations = 5
            loop = asyncio.get_event_loop()

            for _ in range(max_iterations):
                # run claude API call in thread pool to avoid blocking
                response = await loop.run_in_executor(
                    None,
                    lambda: anthropic.messages.create(
                        model="claude-opus-4-5-20251101",
                        max_tokens=600,
                        system=SYSTEM_PROMPT,
                        tools=TOOLS,
                        messages=messages
                    )
                )

                # check if we need to execute tools
                if response.stop_reason == "tool_use":
                    # add assistant's response to messages
                    messages.append({"role": "assistant", "content": response.content})

                    # execute each tool call (async, runs in thread pool)
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  tool: {block.name}({block.input})")
                            result = await execute_tool(block.name, block.input)
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

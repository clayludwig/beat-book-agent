"""
agent.py
--------
Claude-powered agent that has tools to explore stories/topics and interview
the reporter, then produces a beat book.

The agent runs in an async loop.  Most tools execute locally, but
`interview_user` pauses execution and sends the question to the frontend
via a callback.  The callback returns the user's answer so the loop can
resume.
"""

import json
from typing import Callable, Awaitable, Any
from anthropic import Anthropic

from pipeline import PipelineResult

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

AGENT_MODEL = "claude-sonnet-4-6"

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS (Anthropic tool-use schema)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "view_topics",
        "description": (
            "View all discovered topics from the uploaded stories. Returns broad "
            "and specific topics with story counts. Use this first to understand "
            "the landscape of coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_stories_in_topic",
        "description": (
            "List all stories that belong to a given topic. Returns story index, "
            "title, and date for each."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The exact topic label to look up.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "read_story",
        "description": (
            "Read the full content of a story by its index number. Returns title, "
            "author, date, and full text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Zero-based story index.",
                },
            },
            "required": ["index"],
        },
    },
    {
        "name": "search_stories",
        "description": (
            "Search stories by keyword. Returns matching story indices, titles, "
            "and dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for in story titles and content.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "interview_user",
        "description": (
            "Ask the reporter a question to understand their beat and goals. "
            "Supports four question types:\n"
            "- 'checklist': multi-select from a list of options\n"
            "- 'single_choice': pick exactly one option\n"
            "- 'multiple_choice': pick one or more options\n"
            "- 'free_response': open text input\n\n"
            "Use this to understand what topics the reporter cares about, "
            "what they already know, what audience they write for, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the reporter.",
                },
                "question_type": {
                    "type": "string",
                    "enum": ["checklist", "single_choice", "multiple_choice", "free_response"],
                    "description": "The type of UI to present.",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Options for checklist / single_choice / multiple_choice. Ignored for free_response.",
                },
            },
            "required": ["question", "question_type"],
        },
    },
    {
        "name": "generate_beat_book",
        "description": (
            "Write the final beat book as a Markdown document. Call this once you "
            "have gathered enough information from the topics, stories, and the "
            "reporter's answers. The content you pass will be saved as the output file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "markdown_content": {
                    "type": "string",
                    "description": "The complete beat book in Markdown format.",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename for the beat book (e.g. 'sports_beat_book.md').",
                },
            },
            "required": ["markdown_content", "filename"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert journalism mentor and beat-book author. Your job is to help \
a reporter create a comprehensive "beat book" — a practical reporting guide for \
covering a specific beat (topic area).

You have been given a set of news stories that the reporter has uploaded. These \
stories have already been analyzed and grouped into topics automatically.

Your workflow:
1. **Explore** — Start by using `view_topics` to see what topics exist in the \
stories. Read a few representative stories to understand the coverage.
2. **Interview** — Use `interview_user` to ask the reporter 3–5 focused \
questions. Start with a checklist of the discovered topics so they can select \
which ones form their beat. Then ask clarifying questions about their audience, \
experience level, and what they need most from the guide.
3. **Research** — Based on their answers, dig deeper into the relevant stories \
using `list_stories_in_topic`, `read_story`, and `search_stories`. Take note \
of key sources, recurring themes, open questions, and story angles.
4. **Generate** — Use `generate_beat_book` to produce a polished Markdown \
document.

The beat book should include:
- **Beat Overview**: What the beat covers and why it matters
- **Key Topics & Themes**: Organized by the topics the reporter selected, \
with context drawn from actual stories
- **Key Sources & Players**: People, organizations, and institutions that \
appear repeatedly — with context on their roles
- **Story Ideas & Angles**: Concrete follow-up stories or unexplored angles \
suggested by the existing coverage
- **Background & Context**: Important history or policy context a new reporter \
would need
- **Reporting Tips**: Practical advice specific to this beat
- **Calendar & Recurring Events**: Regular meetings, seasonal events, deadlines

Be specific. Reference actual stories, names, and details from the uploaded \
content — not generic advice. The beat book should be so useful that a brand-new \
reporter could pick it up and immediately start producing informed coverage.

Keep your conversational messages concise. Use tools frequently.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TOOL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_local_tool(name: str, input_data: dict, result: PipelineResult) -> str:
    """Execute a non-interactive tool and return a string result."""
    if name == "view_topics":
        return result.topic_summary()

    if name == "list_stories_in_topic":
        stories = result.stories_for_topic(input_data["topic"])
        if not stories:
            return f"No stories found for topic '{input_data['topic']}'. Check exact spelling."
        return json.dumps(stories, indent=2)

    if name == "read_story":
        story = result.get_story(input_data["index"])
        if not story:
            return f"Invalid index {input_data['index']}. Valid range: 0–{len(result.stories)-1}."
        return json.dumps({
            "index": input_data["index"],
            "title": story.get("title", ""),
            "author": story.get("author", ""),
            "date": story.get("date", ""),
            "topics": result.story_topics[input_data["index"]],
            "content": story.get("content", "")[:3000],
        }, indent=2)

    if name == "search_stories":
        matches = result.search_stories(input_data["query"])
        if not matches:
            return f"No stories matching '{input_data['query']}'."
        return json.dumps(matches, indent=2)

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

# Type for the callback that sends questions to the frontend and gets answers
InterviewCallback = Callable[[dict], Awaitable[str]]
# Type for the callback that sends agent text messages to the frontend
MessageCallback   = Callable[[str], Awaitable[None]]
# Type for the callback that reports tool execution status
ToolStatusCallback = Callable[[str, str], Awaitable[None]]


# Human-friendly descriptions for each tool
TOOL_DESCRIPTIONS = {
    "view_topics": "Reviewing discovered topics",
    "list_stories_in_topic": "Listing stories in topic",
    "read_story": "Reading a story",
    "search_stories": "Searching stories",
    "interview_user": "Asking you a question",
    "generate_beat_book": "Writing the beat book",
}


async def run_agent(
    pipeline_result: PipelineResult,
    anthropic_api_key: str,
    on_interview: InterviewCallback,
    on_message: MessageCallback,
    on_beat_book: Callable[[str, str], Awaitable[None]],
    on_tool_status: ToolStatusCallback = None,
) -> None:
    """
    Run the agent loop.

    Args:
        pipeline_result: Output from the embedding/clustering pipeline.
        anthropic_api_key: Anthropic API key.
        on_interview: async callback(question_data) → user's answer string.
        on_message: async callback(text) — sends agent text to the frontend.
        on_beat_book: async callback(filename, markdown) — saves/delivers the beat book.
        on_tool_status: async callback(tool_name, detail) — reports tool execution status.
    """
    client = Anthropic(api_key=anthropic_api_key)

    n_stories = len(pipeline_result.stories)
    n_topics  = len(pipeline_result.topics)

    messages = [
        {
            "role": "user",
            "content": (
                f"I've uploaded {n_stories} news stories. The system has automatically "
                f"discovered {n_topics} topics across them. Please help me build a "
                "beat book from these stories. Start by exploring the topics, then "
                "interview me to understand my beat."
            ),
        }
    ]

    last_message_text = ""   # for deduplication
    beat_book_done = False    # flag to stop after beat book is saved

    # Agent loop — runs until the agent stops calling tools
    MAX_TURNS = 40
    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=16384,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect assistant content blocks
        assistant_content = response.content

        # Send any text blocks to the user (deduplicate consecutive identical msgs)
        for block in assistant_content:
            if block.type == "text" and block.text.strip():
                text = block.text.strip()
                if text != last_message_text:
                    await on_message(block.text)
                    last_message_text = text

        # If the model stopped without tool use, or beat book was already saved, we're done
        if response.stop_reason == "end_turn" or beat_book_done:
            break

        # If the model hit the token limit mid-generation, append the partial
        # response and ask it to continue — this avoids the loop-from-scratch
        # repetition bug that occurs when generate_beat_book's markdown is huge.
        if response.stop_reason == "max_tokens":
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": "Your response was cut off due to length. Please continue exactly where you left off.",
            })
            continue

        # Process tool calls
        if response.stop_reason == "tool_use":
            # Add assistant message to history
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name  = block.name
                tool_input = block.input
                tool_id    = block.id

                # Report tool status to the frontend
                if on_tool_status:
                    desc = TOOL_DESCRIPTIONS.get(tool_name, tool_name)
                    detail = ""
                    if tool_name == "list_stories_in_topic":
                        detail = tool_input.get("topic", "")
                    elif tool_name == "read_story":
                        idx = tool_input.get("index", "")
                        story = pipeline_result.get_story(idx) if isinstance(idx, int) else None
                        detail = story.get("title", f"#{idx}")[:60] if story else f"#{idx}"
                    elif tool_name == "search_stories":
                        detail = tool_input.get("query", "")
                    await on_tool_status(desc, detail)

                if tool_name == "interview_user":
                    # This is interactive — send to frontend and await answer
                    answer = await on_interview(tool_input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Reporter's answer: {answer}",
                    })

                elif tool_name == "generate_beat_book":
                    # Save the beat book
                    await on_beat_book(
                        tool_input.get("filename", "beat_book.md"),
                        tool_input.get("markdown_content", ""),
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Beat book saved successfully. You may now give a brief closing message.",
                    })
                    beat_book_done = True  # exit after this turn

                else:
                    # Local tool
                    result_str = execute_local_tool(tool_name, tool_input, pipeline_result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})

    if not beat_book_done:
        await on_message("✅ Agent session complete.")

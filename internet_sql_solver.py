import os
import json
import sqlite3
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

# load env vars
load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")

# tool implementations

def tool_wget(url, flags="-q -O -"):
    """Fetch content from the web with an explicit user intervention step."""
    cmd = f"wget {flags} {url}"
    print(f"The LLM wants to run: {cmd}")
    answer = input("Allow? (y/n): ").strip().lower()
    if answer not in ("y", "yes"):
        return "USER DENIED: command was not executed."

    try:
        # run wget command and capture output
        completed = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if completed.returncode != 0:
            return f"ERROR: wget failed with code {completed.returncode}: {completed.stderr.strip()}"
        return completed.stdout
    except Exception as e:
        return f"ERROR: wget execution exception: {e}"


def tool_execute_sql(query):
    """Execute a SQL query against a local sqlite database and return results."""
    try:
        conn = sqlite3.connect("data.db")
        cur = conn.cursor()
        cur.execute(query)
        if query.strip().lower().startswith("select"):
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return json.dumps({"columns": columns, "rows": rows}, default=str)
        else:
            conn.commit()
            return f"OK: {cur.rowcount} row(s) affected."
    except Exception as e:
        return f"ERROR: SQL execution failed: {e}"
    finally:
        conn.close()


dispatch = {
    "wget": tool_wget,
    "execute_sql": tool_execute_sql,
}

console = Console()

def create_message_panel(role: str, content: str) -> Panel:
    styles = {
        "user": ("bright_white on blue", "blue", "👤 User"),
        "assistant": ("bright_white on dark_green", "green", "🤖 Assistant"),
        "tool": ("black on yellow", "yellow", "🧪 Tool"),
    }
    text_style, border_color, title = styles.get(role, ("bright_white on grey23", "white", role))
    return Panel(content, title=title, border_style=border_color, padding=(1, 1))


def show_context_stack(messages: list) -> Panel:
    content = "\n".join([f"{i+1}. {m['role']}: {m.get('content','<tool call>')[:120]}" for i, m in enumerate(messages)])
    return Panel(content, title=f"Conversation ({len(messages)} entries)", border_style="bright_magenta", padding=(1, 1))


def run_loop(client, messages, tools):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message
        console.print(create_message_panel("assistant", message.content or "(no assistant text)"))

        if not getattr(message, "tool_calls", None):
            messages.append({"role": "assistant", "content": message.content})
            console.print(Panel("Final assistant response (no tools requested)", border_style="green"))
            return

        # append assistant message along with tool calls
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # execute each tool call
        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            console.print(create_message_panel("tool", f"Calling {name} with {json.dumps(args)}"))
            if name not in dispatch:
                result = f"ERROR: No dispatch function configured for {name}"
            else:
                result = dispatch[name](**args)
            console.print(create_message_panel("tool", f"Result: {result}"))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": result,
            })

        console.print(show_context_stack(messages))




def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment")
        return

    if OPENAI_API_ENDPOINT:
        client = OpenAI(api_key=api_key, base_url=OPENAI_API_ENDPOINT)
    else:
        client = OpenAI(api_key=api_key)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "wget",
                "description": "Download URL content with user confirmation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "flags": {"type": "string"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute SQL on a local SQLite database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "db_path": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    system_prompt = "You are a tool-enabled assistant. Use wget and execute_sql as needed. Please, only answer with a message without tool usage if you think the task is complete, do not ask for confirmation."
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        print("Enter your instruction for the agent (blank to quit):")
        user_prompt = input().strip()
        if not user_prompt:
            print("Exiting")
            return

        messages.append({"role": "user", "content": user_prompt})
        console.print(create_message_panel("user", user_prompt))
        run_loop(client, messages, tools)


if __name__ == "__main__":
    main()

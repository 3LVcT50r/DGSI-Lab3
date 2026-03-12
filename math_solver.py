# =============================================================================================================== #
# Simple Math Problem Solver using OpenAI Function Calling
# ===============================================================================================================
"""
🐧 Math Tutor Demo

Interactive CLI where students type high-school-level math problems in
natural language. The AI uses function calling to delegate calculations and
plotting to Python tools rather than guessing.

Requirements:
    pip install -r requirements.txt
    (ensure matplotlib, numpy, rich, python-dotenv are installed)

Usage:
    1. Create a .env file with your API key
    2. python math_solver.py

This script shares the same credential/endpoint setup as
`three_pigs_function_calling.py` but implements a completely different set of
tools oriented around arithmetic, algebra and simple plotting.
"""

import os
import json
import math
import uuid
from dotenv import load_dotenv
from openai import OpenAI
import logging

# rich for colored output
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich import box

# plotting libraries
import numpy as np
import matplotlib.pyplot as plt

# symbolic math (used by extended tools)
try:
    import sympy as sp
except ImportError:
    sp = None

# load env vars
load_dotenv()

# ---------- logging configuration -------------------------------------------------
LOG_FILE = "math_solver.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def log_interaction(user_input: str, response_data: dict, tool_results: list):
    """Append a brief record of the interaction to the logfile."""
    logger.info("=== interaction ===")
    logger.info(f"User input: {user_input}")
    logger.info("API response: %s", json.dumps(response_data))
    for r in tool_results:
        logger.info("Tool %s returned %s", r.get("name"), r.get("content"))
    logger.info("===================")

# initialize console and client placeholder
console = Console()
client = None

# configuration
MODEL = os.getenv("MODEL")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")

# ---------------------------------------------------------------------------
# system prompt for math tutor
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful high-school math tutor.  Students will present you with a
math problem written in plain English.  Your job is to solve the problem
step‑by‑step and provide a clear explanation.

IMPORTANT: whenever you need to perform a calculation (use `evaluate` or
`evaluate_expression`), solve an equation (`solve_linear` or `solve_equation`),
find roots/vertices, factor a polynomial/expression (`factor_quadratic` or
`factor_expression`), or produce a plot (`plot_parabola` or `plot_function`),
you MUST call one of the provided tools.  Do NOT attempt to "think" the answer
yourself.  The Python tools will do the real work and return accurate results. Explain your reasoning clearly for why you decided to use each calculation after tool use

Only use the available function names and parameters exactly as defined.  If
no tool is needed (for example, to rephrase or give general commentary) you may
respond directly.
"""

# ---------------------------------------------------------------------------
# tool definitions provided to the LLM
# ---------------------------------------------------------------------------
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate",
            "description": "Evaluate a numeric expression (e.g. (3/4 + 2/3) * 6).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_expression",
            "description": "Evaluate any arithmetic expression; wrapper around evaluate().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression to evaluate."}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_linear",
            "description": "Solve a linear equation in one variable (e.g. 2x + 5 = 17).",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "The equation to solve."}
                },
                "required": ["equation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": "Solve an algebraic equation (linear or quadratic) for x.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equation": {"type": "string", "description": "The equation to solve."}
                },
                "required": ["equation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "quadratic_roots",
            "description": "Compute the roots of a quadratic equation ax^2 + bx + c = 0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"}
                },
                "required": ["a", "b", "c"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "factor_quadratic",
            "description": "Factor a quadratic polynomial a*x^2 + b*x + c.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"}
                },
                "required": ["a", "b", "c"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "factor_expression",
            "description": "Factor a general algebraic expression (attempts symbolic factoring).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The expression to factor."}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vertex_parabola",
            "description": "Return the vertex (h,k) of y = a*x^2 + b*x + c.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"}
                },
                "required": ["a", "b", "c"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_parabola",
            "description": "Generate a PNG plot of y = a*x^2 + b*x + c over a given range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"},
                    "x_min": {"type": "number"},
                    "x_max": {"type": "number"},
                    "filename": {"type": "string", "description": "Optional filename for the plot (PNG)."}
                },
                "required": ["a", "b", "c", "x_min", "x_max",  "filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_function",
            "description": "Plot an arbitrary function of x over a range; returns the output file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Function of x to plot (e.g. sin(x) or x**3)."},
                    "x_min": {"type": "number"},
                    "x_max": {"type": "number"},
                    "output_file": {"type": "string", "description": "Filename to save the png"},
                    "filename": {"type": "string", "description": "Optional filename for the plot (PNG)."}
                },
                "required": ["expression", "x_min", "x_max",  "filename"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# IMPLEMENTATIONS OF THE TOOLS
# ---------------------------------------------------------------------------

def evaluate(expression: str) -> str:
    """Safely evaluate a simple numeric expression."""
    if not expression or not expression.strip():
        return "Error: empty expression"
    try:
        # use a restricted eval environment
        value = eval(expression, {"__builtins__": {}}, math.__dict__)
        return str(value)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def solve_linear(equation: str) -> str:
    """Solve a linear equation in the variable x, e.g. "2x+5=17".

    This routine handles simple equations with integer or floating coefficients
    on both sides and returns a single solution or a message if none/infinitely
    many exist.
    """
    try:
        eq = equation.replace(" ", "")
        left, right = eq.split("=")
        import re
        def coef(expr: str):
            # return (x_coef, constant)
            x_coef = 0.0
            const = 0.0
            # ensure unary plus signs are explicit
            expr2 = re.sub(r"(?=[-])", "+", expr)
            terms = expr2.split("+")
            for t in terms:
                if not t:
                    continue
                if 'x' in t:
                    t = t.replace('x', '')
                    if t in ['', '+']:
                        t = '1'
                    if t == '-':
                        t = '-1'
                    x_coef += float(t)
                else:
                    const += float(t)
            return x_coef, const
        lx, lc = coef(left)
        rx, rc = coef(right)
        a = lx - rx
        b = rc - lc
        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return "Infinite solutions"
            else:
                return "No solution"
        x = b / a
        return f"x = {x}"
    except Exception as e:
        return f"Error solving linear equation: {e}"


def quadratic_roots(a: float, b: float, c: float) -> str:
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        roots = [complex(-b / (2 * a), math.sqrt(-disc) / (2 * a))]
    else:
        r1 = (-b + math.sqrt(disc)) / (2 * a)
        r2 = (-b - math.sqrt(disc)) / (2 * a)
        roots = [r1, r2]
    return json.dumps(roots)


def factor_quadratic(a: float, b: float, c: float) -> str:
    """Return a string representing the factorization, if integer factors exist."""
    # try integer pair factors for ac
    ac = a * c
    for m in range(-100, 101):
        if m == 0:
            continue
        if ac % m == 0:
            n = ac // m
            if m + n == b:
                # (ax + m)(x + n/a?) but assume a=1 for simplicity
                if a == 1:
                    return f"(x + {m})(x + {n})"
    return "Cannot factor over integers."


def vertex_parabola(a: float, b: float, c: float) -> str:
    h = -b / (2 * a)
    k = a * h * h + b * h + c
    return json.dumps({"h": h, "k": k})


def plot_parabola(a: float, b: float, c: float, x_min: float, x_max: float, filename: str = None) -> str:
    if x_max <= x_min:
        return "Error: x_max must be greater than x_min"
    xs = np.linspace(x_min, x_max, 400)
    ys = a * xs ** 2 + b * xs + c
    plt.figure()
    plt.plot(xs, ys, label=f"y = {a}x^2 + {b}x + {c}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    # determine filename
    if filename and filename.strip():
        fname = filename if filename.lower().endswith(".png") else filename + ".png"
    else:
        safe = f"a{a}_b{b}_c{c}"
        fname = f"parabola_{safe}_{uuid.uuid4().hex}.png"
    plt.savefig(fname)
    plt.close()
    return fname


# new utility functions requested by the user

def evaluate_expression(expression: str) -> str:
    """General wrapper that evaluates any arithmetic expression."""
    return evaluate(expression)


def solve_equation(equation: str) -> str:
    """Attempt to solve an algebraic equation for x.

    This implementation prefers sympy when available to handle linear and
    quadratic (and more) equations symbolically.  If sympy is not installed it
    falls back to the simple routines already defined above.
    """
    if sp is not None:
        try:
            x = sp.symbols('x')
            sol = sp.solve(sp.sympify(equation), x)
            # sympy returns objects; make them JSON-friendly strings
            return json.dumps([str(s) for s in sol])
        except Exception as e:
            return f"Error solving equation with sympy: {e}"
    # fallback logic
    # determine if quadratic by presence of x^2 or x**2
    if 'x^2' in equation or 'x**2' in equation:
        # parse coefficients manually using regex
        try:
            eq = equation.replace(' ', '').replace('^', '**')
            left, right = eq.split('=')
            # move all to left side
            expr = f"({left})-({right})"
            # extract coefficients for x**2, x, constant
            # use simple evaluation by substituting values
            def coeff(pat, expr_val):
                # evaluate expression after replacing x with a numeric value
                return sp.expand(expr_val).coeff(x, pat) if sp else 0
            if sp:
                expr_sym = sp.sympify(expr)
                a = float(expr_sym.coeff(x, 2))
                b = float(expr_sym.coeff(x, 1))
                c = float(expr_sym.subs(x, 0))
                return quadratic_roots(a, b, c)
        except Exception:
            pass
    # default to linear solver
    return solve_linear(equation)


def factor_expression(expression: str) -> str:
    """Factor the given expression using symbolic algebra if available."""
    if sp is not None:
        try:
            expr_sym = sp.sympify(expression)
            fact = sp.factor(expr_sym)
            return str(fact)
        except Exception as e:
            return f"Error factoring expression with sympy: {e}"
    # fallback: only attempt quadratic factoring by parsing
    # look for ax^2+bx+c pattern
    import re
    m = re.match(r"\s*([+-]?\d*\.?\d*)x\^2([+-]?\d*\.?\d*)x([+-]?\d*\.?\d*)\s*$", expression.replace(' ', ''))
    if m:
        a = float(m.group(1) or 1)
        b = float(m.group(2) or 0)
        c = float(m.group(3) or 0)
        return factor_quadratic(a, b, c)
    return "Cannot factor expression."


def plot_function(expression: str, x_min: float, x_max: float, output_file: str = None, filename: str = None) -> str:
    """Plot an arbitrary function of x over the given range.

    The caller may provide the filename to save the png file, or the AI can
    suggest one via the `filename` parameter.  If both are provided, `filename`
    takes precedence.
    """
    if x_max <= x_min:
        return "Error: x_max must be greater than x_min"
    if not expression or not expression.strip():
        return "Error: empty expression"
    try:
        xs = np.linspace(x_min, x_max, 400)
        if sp is not None:
            x = sp.symbols('x')
            f = sp.lambdify(x, sp.sympify(expression), 'numpy')
            ys = f(xs)
        else:
            # fallback: use eval with x defined
            ys = np.array([eval(expression, {"__builtins__": {}}, {"x": val, **math.__dict__}) for val in xs])
        plt.figure()
        plt.plot(xs, ys, label=f"y = {expression}")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        # determine filename priority: filename param > output_file > autogenerated
        if filename and filename.strip():
            fname = filename if filename.lower().endswith(".png") else filename + ".png"
        elif output_file and output_file.strip():
            fname = output_file
        else:
            # derive from expression
            safe = expression.replace(' ', '').replace('**', '^')
            fname = f"plot_{safe}_{uuid.uuid4().hex}.png"
        plt.savefig(fname)
        plt.close()
        return fname
    except Exception as e:
        return f"Error plotting function: {e}"

# ---------------------------------------------------------------------------
# UI helpers (reused and simplified from three_pigs demo)
# ---------------------------------------------------------------------------

def create_message_panel(role: str, content: str) -> Panel:
    styles = {
        "user": ("bright_white on blue", "blue", "👤 Student"),
        "assistant": ("bright_white on dark_green", "green", "🤖 Tutor"),
        "tool": ("black on yellow", "yellow", "🧮 Tool Result"),
        "system": ("bright_white on purple4", "magenta", "⚙️ System"),
    }
    text_style, border_color, title = styles.get(role, ("bright_white on grey23", "white", role))
    return Panel(
        Text(content, style=text_style),
        title=title,
        title_align="left",
        border_style=border_color,
        padding=(0, 1)
    )


def show_context_stack(messages: list) -> Panel:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold bright_white on grey23", style="on grey23")
    table.add_column("#", style="bright_cyan on grey23", width=3)
    table.add_column("Role", style="bright_magenta on grey23", width=12)
    table.add_column("Content Preview", style="bright_white on grey23")
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        preview = content.replace("\n", " ") if content else "(tool_calls)"
        table.add_row(str(i), role, preview)
    return Panel(table, title=f"📚 Context Stack ({len(messages)} messages)", border_style="magenta", style="on grey23", padding=(0, 1))


def show_api_request(request_data: dict) -> Panel:
    json_str = json.dumps(request_data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", background_color="grey23", word_wrap=True)
    return Panel(syntax, title="📤 API Request", border_style="yellow", style="on grey23", padding=(0, 1))


def show_api_response(response_data: dict) -> Panel:
    json_str = json.dumps(response_data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", background_color="grey23", word_wrap=True)
    return Panel(syntax, title="📥 API Response", border_style="cyan", style="on grey23", padding=(0, 1))


def wait_for_llm():
    return Live(
        Panel(
            Spinner("dots", text=Text(" Waiting for LLM response...", style="bold black on yellow")),
            border_style="yellow",
            style="on yellow",
            padding=(0, 1)
        ),
        console=console,
        refresh_per_second=10
    )

# ---------------------------------------------------------------------------
# chat/solver loop
# ---------------------------------------------------------------------------

def run_solver():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    console.print(create_message_panel("system", SYSTEM_PROMPT))
    while True:
        console.print()
        problem = console.input("[bold blue]📝 Enter math problem (blank to quit): [/bold blue]")
        if not problem.strip():
            console.print(Panel(Text("Goodbye!", style="bold bright_white on grey23"), border_style="dim", style="on grey23"))
            break
        messages.append({"role": "user", "content": problem})
        console.print(create_message_panel("user", problem))

        request_data = {"model": MODEL, "messages": messages, "temperature": 0.2, "tools": AVAILABLE_TOOLS}
        if OPENAI_API_ENDPOINT:
            request_data["_endpoint"] = OPENAI_API_ENDPOINT
        console.print()
        console.print(show_api_request(request_data))

        with wait_for_llm():
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=AVAILABLE_TOOLS,
                temperature=0.2
            )
        assistant_message = response.choices[0].message
        response_data = {
            "id": response.id,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "message": {"role": "assistant", "content": assistant_message.content, "tool_calls": None}
        }
        if assistant_message.tool_calls:
            response_data["message"]["tool_calls"] = [
                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in assistant_message.tool_calls
            ]
        console.print()
        console.print(show_api_response(response_data))

        tool_results = []
        if assistant_message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in assistant_message.tool_calls
                ]
            })
            if assistant_message.content:
                console.print(create_message_panel("assistant", assistant_message.content))
            # execute each tool
            for tool_call in assistant_message.tool_calls:
                fname = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                console.print(create_message_panel("tool", f"Calling {fname} with {args}"))
                result = globals()[fname](**args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fname, "content": result})
                console.print(create_message_panel("tool", result))
                tool_results.append({"name": fname, "content": result})
            # follow-up
            follow_req = {"model": MODEL, "messages": messages, "tools": AVAILABLE_TOOLS, "temperature": 0.2}
            if OPENAI_API_ENDPOINT:
                follow_req["_endpoint"] = OPENAI_API_ENDPOINT
            console.print(show_api_request(follow_req))
            with wait_for_llm():
                follow = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    temperature=0.2
                )
            follow_content = follow.choices[0].message.content
            messages.append({"role": "assistant", "content": follow_content})
            console.print(show_api_response({
                "id": follow.id,
                "model": follow.model,
                "finish_reason": follow.choices[0].finish_reason,
                "message": {"role": "assistant", "content": follow_content}
            }))
            console.print(create_message_panel("assistant", follow_content))
        else:
            messages.append({"role": "assistant", "content": assistant_message.content})
            console.print(create_message_panel("assistant", assistant_message.content))

        console.print(show_context_stack(messages))
        # write log entry for this round
        log_interaction(problem, response_data, tool_results)


def main():
    global client
    console.clear()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(Panel(Text("OPENAI_API_KEY not found!"), border_style="red"))
        return
    if OPENAI_API_ENDPOINT:
        client = OpenAI(api_key=api_key, base_url=OPENAI_API_ENDPOINT)
    else:
        client = OpenAI(api_key=api_key)

    console.print(Panel(Text(f"Model: {MODEL}\nEndpoint: {OPENAI_API_ENDPOINT or 'https://api.openai.com/v1'}"), title="⚙️ Configuration", border_style="cyan", style="on grey23"))
    run_solver()

if __name__ == "__main__":
    main()

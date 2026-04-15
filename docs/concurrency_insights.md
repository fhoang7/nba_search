# Concurrency in Python: AI Engineer Notes

*From exploring NBA search ingest pipeline design*

---

## Ingest Architecture Patterns

Four meaningful approaches to data ingestion pipelines, in order of production maturity:

| Approach | Speed | Complexity | Best For |
|---|---|---|---|
| Sequential (default) | Slow | Low | Dev/prototype |
| `ThreadPoolExecutor` | Fast | Medium | Batch scripts |
| `asyncio` + `httpx` | Fast | High | Servers, complex fan-out |
| Idempotent/incremental | Same as base | Low | **Always add this** |

**Key rule:** Idempotency first, concurrency second. Rerunning a pipeline should never re-fetch or re-embed data you already have.

```python
def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["game_id"] for line in path.open()}

# Write in append mode, not overwrite
output_path.open("a")
```

---

## Three Levels of Parallelism

```
Level 3: Ray / Dask    — distributes work across machines/processes
Level 2: asyncio       — concurrent I/O within one process (coroutines)
Level 1: ThreadPool    — concurrent I/O within one process (OS threads)
```

These **compose**, they don't compete. With Ray, you'd use Ray for distribution + asyncio inside each worker for I/O concurrency. ThreadPoolExecutor is a shim for when you're stuck with sync libraries.

---

## async vs ThreadPoolExecutor

### The Core Difference

**Threads** are preemptive — the OS decides when to context-switch.  
**asyncio** is cooperative — you decide when to yield with `await`.

| | Threads | asyncio |
|---|---|---|
| Memory per concurrent task | ~1MB (stack) | ~1KB (coroutine) |
| 1,000 concurrent tasks | OS scheduler thrashes | Handles fine |
| Blocking call sneaks in | Other threads continue | **Event loop freezes** |
| Legacy sync library | Just works | Needs `run_in_executor` shim |
| Adoption cost | Zero refactoring | Viral up the call chain |

### async is viral, threads are not

Once one function is `async`, everything that calls it must also be `async`. The `await` keyword propagates upward through your entire call stack. A sync call inside an async function doesn't error — it **silently blocks everything**.

```python
# Looks fine, silently kills async performance:
async def fetch(url):
    return requests.get(url)  # ← blocking call, freezes event loop
```

ThreadPoolExecutor wraps any sync code with zero changes to the functions themselves.

### When to use each

```
Writing a server (arbitrary concurrent connections)?
  → async — connection density problem; threads don't scale to 10k+

Batch job / pipeline with fixed workload?
  → ThreadPoolExecutor — simpler, no viral refactoring debt

Need sophisticated coordination (timeouts, race, fan-out)?
  → async primitives are cleaner (gather, wait, timeout)

Everything else?
  → ThreadPoolExecutor, move on
```

---

## The AI Engineer Skill: Specification

Claude will write async or thread-based code — but it will default to whichever pattern is most common in training data, not most appropriate for your context.

**"Make it faster"** → Claude reaches for asyncio (looks impressive, fits generic I/O pattern)  
**"Batch script, ~20 requests, use ThreadPoolExecutor with max_workers=10, keep requests library, preserve partial results on failure"** → Claude writes the right thing

The skill isn't prompting. It's being able to:
1. Recognize which class of problem you have
2. Spec precisely enough to brief Claude like a junior engineer
3. Read generated code and know whether it's appropriate

Understanding tradeoffs *before* writing code is what separates engineers who use AI well from those who vibe-code and hope it works.

---

## FastAPI Note

FastAPI acknowledges async virality by letting you define route handlers as either `async def` or plain `def`. Plain `def` handlers are automatically run in a thread pool — so a blocking call won't freeze the event loop. The framework absorbs the boundary problem for you.

---

*Tags: #python #concurrency #async #threading #ai-engineering #production*

# Code Rules - Tabula Rasa Project

## 🎯 Core Principles

1. **Epistemic Transparency** - завжди вказуй confidence scores та джерела
2. **Clean Architecture** - розділення concerns (state, nodes, helpers, clients)
3. **Type Safety** - усі функції мають type hints
4. **Async First** - весь I/O через async/await
5. **Observability** - логування + LangSmith tracing

## 📝 Code Standards

### Naming Conventions

```python
# Functions - snake_case
async def extract_facts_node(state: AgentState) -> Dict[str, Any]:
    pass

# Classes - PascalCase
class Neo4jMessageStore:
    pass

# Constants - UPPER_CASE
MAX_ITERATIONS = 3

# Private - _leading_underscore
_global_instance = None

# Type vars - single uppercase
T = TypeVar('T')
```

### Import Order

```python
# 1. Standard library
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# 2. Third party
from pydantic import BaseModel, Field
from langsmith import traceable

# 3. Local - absolute imports only
from agent.state import AgentState
from clients.llm_client import get_llm_client
from config.settings import settings
```

### Docstring Format

```python
async def process_node(state: AgentState, param: str) -> Dict[str, Any]:
    """Brief description in one line. """
```

## 🔒 Type Safety Rules

### Required Type Hints

```python
# ✅ All parameters and return types
async def func(arg: str, opt: Optional[int] = None) -> Dict[str, Any]:
    pass

# ✅ Complex types
from typing import List, Dict, Tuple, Union

data: List[Dict[str, Any]] = []
result: Tuple[bool, float] = (True, 0.95)
value: Union[str, int] = "test"

# ✅ Pydantic models
class MyModel(BaseModel):
    field: str
    count: int = Field(default=0, ge=0)
```

### TypedDict for State

```python
# ✅ GOOD - strict typing
class AgentState(TypedDict):
    message_uid: str
    message_text: str
    confidence: float
    # ...

# ❌ BAD - loose typing
state = {}  # No structure
state: dict = {}  # Too generic
```

## ⚡ Async Patterns

### Proper Async Usage

```python
# ✅ GOOD - pure async
async def get_data() -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# ✅ GOOD - run blocking in executor
async def cpu_intensive() -> int:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_computation)
    return result

# ❌ BAD - blocking in async
async def bad_async():
    time.sleep(5)  # Blocks event loop!
    return requests.get(url)  # Blocking!
```

### Async Context Managers

```python
# ✅ GOOD
async with get_client() as client:
    result = await client.process()

# ✅ GOOD - cleanup
try:
    client = await get_client()
    result = await client.process()
finally:
    await client.close()
```

## 🧪 Testing Standards

### Test File Naming

```
tests/
  test_nodes.py          # Unit tests for nodes
  test_integration.py    # Integration tests
  conftest.py           # Fixtures
```

### Test Structure

```python
import pytest
from agent.state import create_initial_state

@pytest.mark.asyncio
async def test_happy_path():
    # Arrange
    state = create_initial_state(
        message_uid="test-001",
        message_text="test input"
    )
    
    # Act
    result = await node_function(state)
    
    # Assert
    assert "expected_key" in result
    assert result["expected_key"] == expected_value

@pytest.mark.asyncio
async def test_error_handling():
    # Test graceful degradation
    state = create_initial_state(
        message_uid="test-002",
        message_text=""  # Invalid
    )
    
    result = await node_function(state)
    
    # Should not crash
    assert "error" in result or result["output"] == []
```

## 📊 Logging Standards

### Log Levels

```python
# DEBUG - detailed diagnostic info
logger.debug(f"Retrieved {len(results)} items from cache")

# INFO - general flow information
logger.info("=== Starting Extract Facts Node ===")
logger.info(f"Processing message {uid}")

# WARNING - unexpected but handled
logger.warning(f"No context found for query: {query}")
logger.warning(f"Confidence {conf} below threshold {threshold}")

# ERROR - operation failed
logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
logger.error(f"Graphiti search failed: {e}")
```

### Structured Logging

```python
# ✅ GOOD - contextual information
logger.info(
    f"Conflict detected between messages",
    extra={
        "old_msg": old_uid,
        "new_msg": new_uid,
        "confidence": confidence
    }
)

# ❌ BAD - generic logs
logger.info("Something happened")
```

## 🔐 Security & Privacy

### User Isolation

```python
# ✅ GOOD - always filter by user_id
results = await db.get_messages(user_id=user_id)

# ❌ BAD - leaking cross-user data
results = await db.get_all_messages()  # Dangerous!
```

### Input Validation

```python
# ✅ GOOD - validate before processing
if not message_text or len(message_text.strip()) < 3:
    raise ValueError("Message too short")

if not is_valid_uuid(message_uid):
    raise ValueError("Invalid message UID")

# ✅ GOOD - Pydantic validation
class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    uid: str = Field(regex=r'^[a-f0-9-]+$')
```

## 🎨 Code Organization

### File Size Limits

- **Max 500 lines** per file
- **Max 50 lines** per function
- If longer → split into multiple files/functions

### Module Structure

```python
# Top of file
"""
Module description - what it does, why it exists.
"""

# Imports
import ...

# Constants
CONSTANT = value

# Type definitions
T = TypeVar('T')

# Classes/Functions
class MyClass:
    pass

def my_function():
    pass

# Global instances (if needed)
_instance = None

def get_instance():
    pass
```

## 🚫 Forbidden Patterns

### Never Use These

```python
# ❌ Bare except
try:
    risky()
except:  # DON'T
    pass

# ✅ Specific exceptions
try:
    risky()
except (ValueError, TypeError) as e:
    logger.error(f"Error: {e}")

# ❌ Mutable defaults
def func(items=[]):  # DON'T
    items.append(1)

# ✅ None default
def func(items: Optional[List] = None):
    if items is None:
        items = []

# ❌ Global state mutation
global_var = 0

def func():
    global global_var  # DON'T
    global_var += 1

# ✅ Encapsulation
class Counter:
    def __init__(self):
        self._count = 0
    
    def increment(self):
        self._count += 1
```

## 📐 Complexity Limits

- **Cyclomatic complexity** ≤ 10 per function
- **Max nesting depth** ≤ 4 levels
- **Max parameters** ≤ 5 per function

If exceeded → refactor into smaller functions.

## 🔄 State Management

### State Update Pattern

```python
# ✅ GOOD - return partial updates
async def node(state: AgentState) -> Dict[str, Any]:
    result = await process(state["input"])
    return {
        "output": result,
        "confidence": 0.95
    }

# ❌ BAD - return full state
async def node(state: AgentState) -> AgentState:
    state["output"] = result  # Mutation!
    return state
```

### Immutability

```python
# ✅ GOOD - create new objects
new_list = existing_list + [new_item]
new_dict = {**existing_dict, "key": "value"}

# ❌ BAD - mutate in place
existing_list.append(new_item)  # Side effect
existing_dict["key"] = "value"  # Mutation
```

## 📦 Dependency Management

### Import Rules

```python
# ✅ GOOD - explicit imports
from agent.state import AgentState, create_initial_state

# ❌ BAD - wildcard imports
from agent.state import *

# ✅ GOOD - qualified imports for clarity
import logging
logger = logging.getLogger(__name__)

# ❌ BAD - shadowing built-ins
from typing import list  # Shadows list()
```

### Circular Imports

```python
# Avoid by:
# 1. Move shared code to common module
# 2. Use TYPE_CHECKING

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent.state import AgentState  # Only for type checking
```

## 🎯 Performance Guidelines

### Efficient Operations

```python
# ✅ GOOD - batch operations
results = await client.batch_process(items)

# ❌ BAD - loop of async calls
results = []
for item in items:
    result = await client.process(item)  # N calls!
    results.append(result)

# ✅ GOOD - parallel execution
results = await asyncio.gather(*[
    client.process(item) for item in items
])
```

### Memory Management

```python
# ✅ GOOD - generators for large datasets
def process_large_file():
    with open(file) as f:
        for line in f:  # Lazy
            yield process(line)

# ❌ BAD - load everything
def process_large_file():
    with open(file) as f:
        lines = f.readlines()  # All in memory!
        return [process(line) for line in lines]
```

## ✅ Pre-Commit Checklist

Before every commit:

- [ ] All type hints present
- [ ] Docstrings on public functions
- [ ] Error handling implemented
- [ ] Logging statements added
- [ ] No print() calls
- [ ] Tests written and passing
- [ ] No TODO comments (create issues instead)
- [ ] No commented-out code
- [ ] Imports organized
- [ ] No linter errors

---

**Remember:** Clean code is not just about working code, it's about **maintainable**, **readable**, **testable** code that others (including future you) can understand and modify with confidence. 🎯

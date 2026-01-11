# Архітектура Memory Management

> Документація системи управління пам'яттю в Tabula Rasa Agent
> Версія: 2.0 (після рефакторингу на teach/solve architecture)

## Зміст

- [Огляд системи](#огляд-системи)
- [Архітектура пам'яті](#архітектура-памяті)
- [Потоки даних](#потоки-даних)
- [Операції з пам'яттю](#операції-з-памяттю)
- [Оптимізація та проблеми](#оптимізація-та-проблеми)

---

## Огляд системи

### Двошарова архітектура

Система використовує **два незалежні шари** для зберігання інформації:

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Graphiti (Temporal Knowledge Graph)          │
│  - Entities (сутності)                                  │
│  - Relations (зв'язки між сутностями)                   │
│  - Episodes (conversation turns)                        │
│  - Temporal reasoning (зміни в часі)                    │
│  → Для семантичного пошуку та reasoning                 │
└─────────────────────────────────────────────────────────┘
                            │
                            │ Пов'язані через episode_name
                            ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: Neo4j Message Store (Reference Tracking)     │
│  - Message nodes (з UIDs)                               │
│  - Episode nodes (links)                                │
│  - Timestamps, user IDs                                 │
│  → Для source references в відповідях                   │
└─────────────────────────────────────────────────────────┘
```

### Чому два шари?

**Graphiti** (Layer 1):
- Зберігає **структуровані знання** (entities + relations)
- Виконує **hybrid search** (semantic + BM25)
- Підтримує **temporal reasoning** (зміни фактів в часі)
- LLM автоматично витягує entities з episodes

**Neo4j Message Store** (Layer 2):
- Зберігає **raw messages** з унікальними UIDs
- Дозволяє **trackати джерела** (source references)
- Швидкий lookup: episode_name → message_uid
- Needed для відповідей з посиланнями на джерела

---

## Архітектура пам'яті

### Типи пам'яті в системі

Система імплементує **3 типи пам'яті** з research paper:

#### 1. Semantic Memory (Семантична пам'ять)

**Що зберігається:** Факти, знання, структуровані дані

**Де:** Graphiti → Entity/Relation nodes

**Приклад:**
```
Episode: "Київ - столиця України"
   ↓ LLM extraction
Entity(name="Київ", type="CITY")
Entity(name="Україна", type="COUNTRY")
Relation(Київ → столиця → Україна)
```

**Використання:**
- Retrieve context node шукає релевантні entities
- Hybrid search (embeddings + BM25)
- Reranking з BGE cross-encoder

#### 2. Episodic Memory (Епізодична пам'ять)

**Що зберігається:** Conversation turns (user + assistant pairs)

**Де:**
- Graphiti → Episode nodes (з timestamps)
- Neo4j → Message nodes (з UIDs)

**Приклад:**
```
Episode {
  name: "teach_msg-001",
  body: "User: Київ - столиця України\nAssistant: Дякую за інформацію...",
  timestamp: 2024-01-15T10:30:00Z
}
   ↓
Message {
  uid: "msg-001",
  text: "Київ - столиця України",
  episode_name: "teach_msg-001"
}
```

**Використання:**
- Кожен conversation turn = окремий episode
- Temporal queries (коли користувач щось сказав)
- Source references через episode_name → message_uid lookup

#### 3. Procedural Memory (Процедурна пам'ять)

**Що зберігається:** Як виконувати задачі (implicitly в ReAct steps)

**Де:** Не зберігається явно (на відміну від semantic/episodic)

**Як працює:**
- ReAct loop вчиться patterns з попередніх successful iterations
- Reasoning steps (thought → action → observation)
- Немає explicit storage, реконструюється з episodic memory

**Приклад:**
```
Завдання: "Створи рецепт салату"
   ↓ ReAct learns pattern
Thought: "Шукаю інформацію про інгредієнти з пам'яті"
Action: search
Observation: "Користувач раніше казав що любить помідори"
   ↓
Thought: "Можу створити салат з помідорів"
Action: answer
```

---

## Потоки даних

### TEACH Path (User → Agent)

Коли користувач **надає інформацію** (teaching):

```
1. classify_intent_node
   ↓ intent="teach"

2. extract_facts_node
   ↓ LLM витягує факти
   [{subject: "Київ", relation: "столиця", object: "Україна"}]

3. check_conflicts_node
   ↓ Перевіряє конфлікти з існуючими фактами
   Шукає в Graphiti: search("Київ столиця")

4a. Якщо conflicts → auto_resolve_node
    └─ Auto-accept нову інформацію (Tabula Rasa принцип)

4b. Якщо NO conflicts → generate_confirmation_node
    └─ LLM генерує підтвердження навчання

5. store_knowledge_node
   ├─ Graphiti.add_episode() → Entity/Relation extraction
   └─ Neo4j.save_message() → Message node з UID

   Episode: "teach_msg-001"
      ├─ Graphiti: Episode(body, entities, relations, timestamp)
      └─ Neo4j: Message(uid, text, episode_name="teach_msg-001")
```

**Коли зберігається пам'ять:** Завжди в кінці TEACH path (store_knowledge_node)

**Що саме зберігається:**
- Graphiti: Episode → LLM автоматично витягує entities/relations
- Neo4j: Raw message з унікальним UID

### SOLVE Path (Agent → User)

Коли користувач **ставить завдання** (asking):

```
1. classify_intent_node
   ↓ intent="solve"

2. retrieve_context_node
   ├─ Graphiti.search(query) → Hybrid search (semantic + BM25)
   └─ Results: [{content, episode_name, score, timestamp}]

3. Для кожного result:
   ├─ episode_name → Neo4j.get_message_uid_by_episode()
   └─ Додаємо source_msg_uid до context

4. react_loop_node
   ├─ Iterative reasoning (до 3 iterations)
   ├─ Thought: "Що потрібно зробити?"
   ├─ Action: "search" або "answer"
   └─ Observation: результати action

5. generate_answer_node
   ├─ LLM генерує відповідь з context
   ├─ Extract message UIDs використаних sources
   └─ Response + references
```

**Коли зберігається пам'ять:** **НІ!** SOLVE path НЕ зберігає нову інформацію

**Чому не зберігаємо:**
- SOLVE = retrieval + reasoning + generation
- Не створюємо нові факти, лише використовуємо існуючі
- Уникаємо дублювання (не зберігаємо generated відповіді як facts)

---

## Операції з пам'яттю

### 1. Збереження (TEACH path only)

#### store_knowledge_node (agent/nodes/store.py:19)

```python
async def store_knowledge_node(state: AgentState):
    # 1. Add episode to Graphiti
    episode_name = f"teach_{state['message_uid']}"

    episode_body = f"""User: {user_message}
Assistant: {assistant_message}"""

    await graphiti.add_episode(
        episode_body=episode_body,
        episode_name=episode_name,
        source_description=f"user:{user_id}, uid:{message_uid}",
        reference_time=timestamp  # Temporal reasoning
    )

    # 2. Save to Neo4j for references
    await message_store.save_message(
        uid=message_uid,
        text=user_message,
        episode_name=episode_name,
        user_id=user_id,
        timestamp=timestamp
    )
```

**Коли викликається:**
- В кінці TEACH path
- Після conflict resolution (якщо були конфлікти)
- Після generate_confirmation (якщо не було конфліктів)

**Що відбувається всередині Graphiti:**
```
1. Episode зберігається в Neo4j
2. LLM (Lapa) читає episode_body
3. Витягує entities: {name, type, summary}
4. Витягує relations: {source, target, type}
5. Створює nodes в Neo4j:
   - Episode node
   - Entity nodes (якщо нові)
   - Relation edges між entities
6. Embeddings для entities (HostedQwenEmbedder)
7. BGE reranking metadata
```

### 2. Витягування (SOLVE path)

#### retrieve_context_node (agent/nodes/retrieve.py:18)

```python
async def retrieve_context_node(state: AgentState):
    # Hybrid search в Graphiti
    search_results = await graphiti.search(
        query=state["message_text"],
        limit=settings.graphiti_search_limit  # Default: 10
    )

    # Додаємо source message UIDs
    retrieved_context = []
    for result in search_results:
        episode_name = result.get('episode_name')
        source_msg_uid = await get_message_uid_by_episode(episode_name)

        retrieved_context.append({
            "content": result['content'],
            "source_msg_uid": source_msg_uid,
            "timestamp": result['timestamp'],
            "score": result['score']
        })

    return {"retrieved_context": retrieved_context}
```

**Graphiti Hybrid Search:**
```
1. Semantic search (Qwen embeddings):
   - Embed query
   - Cosine similarity з entity embeddings
   - Top-K results

2. BM25 (keyword-based):
   - Tokenize query
   - TF-IDF scoring
   - Top-K results

3. Merge + Rerank:
   - Combine semantic + BM25 results
   - BGE cross-encoder reranking
   - Sort by final score
```

**Коли викликається:**
- На початку SOLVE path (після classify)
- Перед ReAct loop

**Фільтрація:**
- `relevance_threshold` = 0.3 (default)
- Skip empty results (len < 5 chars)
- Limit: 10 results

### 3. ReAct Additional Search

#### react_loop_node (agent/nodes/react.py:20)

```python
async def react_loop_node(state: AgentState):
    for iteration in range(max_iterations):  # До 3 iterations
        # 1. Thought
        thought = await llm.generate_async(thought_prompt)

        # 2. Action
        if "шукати" in thought.lower():
            action = "search"
            search_query = extract_search_query(thought)

            # Додатковий пошук в Graphiti
            search_results = await graphiti.search(
                query=search_query,
                limit=3  # Менше ніж initial retrieve
            )

            # Додаємо results до context
            for result in search_results:
                context_text += f"\n{result['content']}"

        elif "готовий" in thought.lower():
            action = "answer"
            break
```

**Чому додатковий search?**
- Initial retrieve може пропустити деталі
- ReAct iterative refines пошук
- Smaller limit (3 vs 10) для фокусу

**Чи зберігаємо ReAct steps?**
- **НІ!** Зберігаємо тільки в state для поточного виконання
- Не створюємо episodes з ReAct reasoning
- Reasoning реконструюється кожен раз заново

---

## Оптимізація та проблеми

### Чи є дублювання операцій?

#### ✅ Проблема: OLD architecture (nodes.py)

**Старий код** (retrieve_memory_node + save_to_memory_node):
```python
# ПРОБЛЕМА: Зберігали КОЖНУ interaction (teach + solve)
async def save_to_memory_node(state):
    # Зберігали навіть SOLVE відповіді як episodes
    episode_body = f"User: {user_msg}\nAssistant: {ai_msg}"
    await graphiti.add_episode(episode_body, ...)
```

**Що було не так:**
1. SOLVE path створював episodes з generated відповідями
2. LLM витягував "fake facts" з AI responses
3. Graph забруднювався non-factual information
4. Retrieval повертав власні відповіді агента як "knowledge"

#### ✅ Рішення: NEW architecture (teach/solve split)

**Новий код** (TEACH path only storage):
```python
# ТІЛЬКИ TEACH path зберігає knowledge
workflow.add_edge("store_knowledge", END)  # TEACH
workflow.add_edge("generate_answer", END)  # SOLVE (no storage!)
```

**Переваги:**
1. Чітке розділення: teach = store, solve = retrieve
2. Graph містить тільки user-provided facts
3. Немає self-pollution від AI responses
4. Episodic memory = тільки teaching moments

### Використання витягнутого контексту

#### retrieve_context_node → generate_answer_node

```python
# retrieve_context_node
return {"retrieved_context": [
    {
        "content": "Київ - столиця України",
        "source_msg_uid": "msg-001",
        "score": 0.95
    }
]}

# react_loop_node
context_text = "\n".join([
    f"[{i}] ({ctx['source_msg_uid']}): {ctx['content']}"
    for i, ctx in enumerate(retrieved_context)
])

# Використовується в prompt
thought_prompt = f"""
Контекст з пам'яті:
{context_text}

Завдання: {task}
"""

# generate_answer_node
system_prompt = f"""
🚫 TABULA RASA: Використовуй ТІЛЬКИ інформацію з контексту.

=== Контекст з пам'яті ===
{format_context(retrieved_context)}
=== Кінець контексту ===

Завдання користувача: {state['message_text']}
"""
```

**Як використовується:**
1. Initial retrieve → Base context для ReAct
2. ReAct iterative search → Додає до context
3. Generate answer → Весь accumulated context в prompt
4. Extract references → Message UIDs в response

### Redundancy Check

#### Чи є зайві memory operations?

**TEACH path:**
```
✅ extract_facts_node - НЕ зберігає, тільки витягує
✅ check_conflicts_node - НЕ зберігає, тільки search для comparison
✅ auto_resolve_node - НЕ зберігає, тільки decision logic
✅ store_knowledge_node - ЄДИНЕ місце де зберігаємо
```

**SOLVE path:**
```
✅ retrieve_context_node - Один раз на початку (broad search)
✅ react_loop_node - До 3 додаткових searches (focused queries)
✅ generate_answer_node - NO storage, тільки generation
```

**Висновок:** Немає дублювання. Кожна операція має чіткий purpose.

---

## Діаграма повного flow

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT MESSAGE                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  classify_intent     │
              │  (LLM classification)│
              └──────┬────────┬──────┘
                     │        │
         ┌───────────┘        └──────────┐
         │ TEACH                    SOLVE │
         ▼                                ▼
┌──────────────────┐            ┌──────────────────┐
│ extract_facts    │            │ retrieve_context │
│ (LLM structured) │            │ (Graphiti search)│
└────────┬─────────┘            └────────┬─────────┘
         │                                │
         ▼                                │
┌──────────────────┐                      │
│ check_conflicts  │                      │
│ (search + LLM)   │                      │
└────┬────────┬────┘                      │
     │        │                           │
  conflict  no conflict                   │
     │        │                           │
     ▼        ▼                           ▼
 ┌────┐   ┌─────┐              ┌──────────────────┐
 │auto│   │conf-│              │   react_loop     │
 │res │   │ irm │              │ (iterative reas.)│
 └─┬──┘   └──┬──┘              └────────┬─────────┘
   │         │                          │
   └────┬────┘                          │
        │                               │
        ▼                               ▼
┌──────────────────┐            ┌──────────────────┐
│ store_knowledge  │            │ generate_answer  │
├──────────────────┤            │ (NO STORAGE!)    │
│ 1. Graphiti      │            └────────┬─────────┘
│    add_episode   │                     │
│    → entities    │                     │
│    → relations   │                     │
│ 2. Neo4j         │                     │
│    save_message  │                     │
│    → UID link    │                     │
└────────┬─────────┘                     │
         │                               │
         ▼                               ▼
       ┌─────────────────────────────────┐
       │          RESPONSE               │
       │  TEACH: confirmation            │
       │  SOLVE: answer + references     │
       └─────────────────────────────────┘
```

---

## Налаштування

### Конфігурація (config/settings.py)

```python
# Graphiti settings
graphiti_search_limit: int = 10           # Max results from search
graphiti_relevance_threshold: float = 0.3  # Min score to include
graphiti_max_episode_length: int = 2048   # Max tokens per episode

# ReAct settings
max_react_iterations: int = 3             # Max ReAct loop iterations

# Embeddings
embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2"
use_hosted_embeddings: bool = True        # HostedQwenEmbedder

# Neo4j
neo4j_uri: str = "bolt://localhost:7687"
neo4j_database: str = "neo4j"
```

### Налаштування Graphiti Search

**Hybrid search balance:**
- Semantic weight: 0.7 (embeddings similarity)
- BM25 weight: 0.3 (keyword matching)
- Reranking: BGE cross-encoder (final scoring)

**Threshold explanation:**
- `relevance_threshold = 0.3` = досить liberal (включаємо багато results)
- Scores 0.0-1.0 (1.0 = perfect match)
- Нижче 0.3 = likely irrelevant

---

## Приклад end-to-end

### Scenario 1: TEACH → SOLVE

```python
# 1. User teaches fact
>>> "Київ - столиця України"

# TEACH path:
classify → intent="teach"
extract_facts → [{subject: "Київ", relation: "столиця", object: "Україна"}]
check_conflicts → search("Київ столиця") → no conflicts
generate_confirmation → "Дякую, я запам'ятав що Київ - столиця України"
store_knowledge:
  ├─ Graphiti: Episode("teach_msg-001", body="User: Київ...")
  │   └─ Entities: Київ(CITY), Україна(COUNTRY)
  │   └─ Relation: Київ -[столиця]-> Україна
  └─ Neo4j: Message(uid="msg-001", episode_name="teach_msg-001")

# 2. User asks question
>>> "Яка столиця України?"

# SOLVE path:
classify → intent="solve"
retrieve_context:
  ├─ Graphiti.search("Яка столиця України?")
  └─ Results: [{content: "Київ - столиця України", episode_name: "teach_msg-001", score: 0.95}]

  # Get source UID
  ├─ Neo4j.get_message_uid_by_episode("teach_msg-001") → "msg-001"
  └─ Add to context: {content: "...", source_msg_uid: "msg-001"}

react_loop (iteration 1):
  thought: "В контексті є інформація про столицю України - це Київ. Готовий відповісти."
  action: "answer"

generate_answer:
  ├─ System prompt з контекстом: "[msg-001]: Київ - столиця України"
  ├─ LLM generates: "Столиця України - Київ [msg-001]"
  └─ Extract references: ["msg-001"]

Response:
  response: "Столиця України - Київ [msg-001]"
  references: ["msg-001"]
```

### Scenario 2: Conflict resolution

```python
# 1. Initial fact
>>> "Моє улюблене місто - Харків"
# TEACH → stored as msg-001

# 2. Conflicting fact
>>> "Моє улюблене місто - Львів"

# TEACH path:
classify → intent="teach"
extract_facts → [{subject: "я", relation: "улюблене місто", object: "Львів"}]
check_conflicts:
  ├─ search("улюблене місто")
  ├─ Found: "Моє улюблене місто - Харків" (msg-001)
  ├─ LLM check_contradiction(old="Харків", new="Львів")
  └─ Result: {is_conflict: true, type: "direct", confidence: 0.9}

auto_resolve:
  ├─ Decision: ACCEPT new info (Tabula Rasa principle)
  ├─ Strategy: Replace old fact
  └─ Explanation: "Preference changed from Харків to Львів"

store_knowledge:
  # Graphiti smart enough to update relations
  # Old: я -[улюблене місто]-> Харків (deprecated)
  # New: я -[улюблене місто]-> Львів (active)
```

---

## Best Practices

### 1. Episode Naming Convention

```python
# TEACH episodes
episode_name = f"teach_{message_uid}"  # teach_msg-001

# Чому важливо:
# - Унікальність гарантована (message_uid unique)
# - Легко linkати з Neo4j
# - Prefix "teach_" = semantic meaning
```

### 2. Conflict Detection

```python
# Коли check_conflicts шукає:
await graphiti.search(query=extracted_fact_text, limit=5)

# Чому тільки 5 results?
# - Конфлікти швидше всього в top results
# - Reduce LLM calls для contradiction checking
# - Performance optimization
```

### 3. ReAct Iterations

```python
# Чому max_iterations = 3?
# - Iteration 1: Initial reasoning + search (якщо потрібно)
# - Iteration 2: Refine based on new context
# - Iteration 3: Final decision
# - Більше = diminishing returns + latency
```

### 4. Context Formatting

```python
# Чому додаємо source_msg_uid в context?
# 1. LLM бачить джерела: "[msg-001]: факт"
# 2. Може включити в response: "Базуючись на [msg-001]..."
# 3. Extract references легше
```

---

## Troubleshooting

### Problem 1: Duplicate episodes

**Симптом:** Same fact зберігається двічі

**Причина:**
```python
# Graphiti НЕ має built-in deduplication!
# Кожен add_episode створює новий Episode node
```

**Рішення:**
- check_conflicts_node детектує duplicates
- auto_resolve_node приймає новий (Tabula Rasa)
- LLM в Graphiti може merge entities (якщо same name)

### Problem 2: Low relevance scores

**Симптом:** retrieve_context повертає irrelevant results

**Причина:**
```python
# 1. Query too vague
# 2. Embeddings не зрозуміли Ukrainian
# 3. Threshold занадто низький
```

**Рішення:**
```python
# Збільшити threshold
graphiti_relevance_threshold: float = 0.5  # Було 0.3

# Або покращити query в ReAct
# ReAct може reformulate query на iteration 2
```

### Problem 3: Source UIDs missing

**Симптом:** `source_msg_uid = "unknown"` в retrieved context

**Причина:**
```python
# Episode в Graphiti НЕ має поля episode_name
# Або Neo4j Message не був created
```

**Рішення:**
```python
# Завжди зберігати в обох місцях:
await graphiti.add_episode(episode_name=name, ...)  # Graphiti
await message_store.save_message(episode_name=name, ...)  # Neo4j

# Check logs:
logger.info(f"Episode saved: {episode_name}")
logger.info(f"Message saved to Neo4j")
```

---

## Висновки

### Ключові принципи

1. **Bidirectional Flow**
   - TEACH path = user → agent (learning)
   - SOLVE path = agent → user (retrieval + reasoning)

2. **Epistemic Awareness**
   - Confidence scores для facts
   - Conflict detection з LLM reasoning
   - Source tracking (message UIDs)

3. **Temporal Reasoning**
   - Episodes з timestamps
   - Graphiti підтримує temporal queries
   - Preferences можуть змінюватись в часі

4. **Knowledge Quality**
   - Hybrid search (semantic + BM25)
   - Reranking (BGE cross-encoder)
   - Relevance threshold filtering

### Метрики успіху

**Good memory system:**
- ✅ TEACH path зберігає тільки user facts
- ✅ SOLVE path НЕ створює fake knowledge
- ✅ Conflicts детектуються та resolve автоматично
- ✅ Source references доступні в відповідях
- ✅ ReAct refines search iteratively

**Що НЕ робити:**
- ❌ Зберігати AI-generated відповіді як facts
- ❌ Skip conflict checking (може overwrite важливі facts)
- ❌ Ігнорувати source UIDs (no references)
- ❌ Зберігати episodic memory без timestamps

---

## Roadmap

### Potential Improvements

1. **Memory Consolidation**
   - Periodic merge similar entities
   - Deduplicate episodes
   - Archive old/irrelevant facts

2. **Advanced Conflict Resolution**
   - User confirmation для critical conflicts
   - Confidence-based merging
   - Multi-source fact verification

3. **Enhanced Retrieval**
   - Query expansion в ReAct
   - Multi-hop reasoning в graph
   - Contextual reranking

4. **Performance Optimization**
   - Cache frequent searches
   - Batch embedding generation
   - Async parallel searches

---

**Автор:** Tabula Rasa Agent Team
**Дата:** 2024-01-11
**Версія:** 2.0 (teach/solve architecture)
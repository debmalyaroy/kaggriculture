# **Architectural Design of a High-Performance Rust Evaluation Harness for Kaggriculture Agents**

The development of competitive agents for complex simulations such as the Kaggriculture competition necessitates rigorous, high-throughput testing environments. While the official framework provides an accessible Python-based execution loop, the performance overhead of running thousands of multi-agent simulations purely in Python often stifles rapid iteration and reinforcement learning at scale1. To address this computational bottleneck, developers frequently construct native, high-performance engines using systems programming languages like Rust, effectively creating exact clones of the environment while orchestrating Python-based competitor agents via Foreign Function Interfaces (FFI).

This comprehensive analysis examines the system architecture required to build an advanced diagnostic test harness in Rust that natively evaluates Kaggriculture Python agents. The design integrates robust execution logic, asynchronous and parallel simulation scaling strategies, precise memory management paradigms, fair evaluation heuristics via seat-swapping, and replay-driven clean-room validation2. By entirely circumventing the latency inherent to standard inter-process communication, this architecture unlocks the capacity for massive local leaderboard generation and evolutionary strategy testing.

## **System Architecture and Python Interoperability**

Creating a Rust engine that acts as a drop-in replacement for the Kaggriculture Python environment requires a highly decoupled dual-architecture approach. The engine itself must perfectly replicate the state machine of the game, rigorously enforcing resource economics, spatial grid logic, agent movement, and deterministic state transitions without relying on any Python dependencies. Concurrently, it must be capable of seamlessly executing reference agents, which are typically authored as plain, single-file Python modules or standalone functions4. The primary engineering challenge lies in bridging the high-performance, strictly owned Rust memory model with the dynamic, garbage-collected Python runtime.

### **Memory Management and the PyO3 Interface**

The most performant bridge between Rust and CPython is established using the bindings provided by the ecosystem, which permit the compilation of Rust code into a Python module, or conversely, allow Rust binaries to embed a full CPython interpreter5. Because the local Kaggriculture test harness acts as the primary execution driver, embedding the interpreter within the Rust execution context is the foundational operational mode.

Python’s memory model diverges significantly from Rust’s strict ownership semantics. In Python, all objects are shared and managed via reference counting, and there is no concept of exclusive, mutable references6. This fundamental difference means that any reference can theoretically mutate an object if the internal APIs permit it, a concept that violates Rust's borrow checker rules6. The interoperability layer addresses this structural disparity through highly specialized smart pointer types, notably pointer structures that interface directly with Python's reference counting mechanisms6.

To safely interact with the embedded Python interpreter, the Rust execution thread must acquire a specific lifetime token6. This token serves as cryptographic-level proof to the Rust compiler that the Global Interpreter Lock (GIL) is actively held by the current thread, satisfying CPython’s historical thread-safety constraints. Types bound to this lifetime contain the token intrinsically, granting them full access to the Python interpreter and offering a complete API for interacting with dynamic Python objects6.

When passing the Kaggriculture environment state—often a complex nested dictionary or a JSON-equivalent structure—from the Rust engine to the Python agent, serialization overhead becomes a primary computational bottleneck. Repeatedly allocating and serializing raw JSON strings completely negates the performance benefits of writing the engine in Rust. Instead, the harness architecture optimizes this data transfer by maintaining shared, zero-copy data structures wherever feasible.

| Interoperability Layer | Mechanism of Action | Performance Implication |
| :---- | :---- | :---- |
| **JSON Serialization** | Rust engine serializes state to string; Python parses to dictionary. | Highest latency; massive memory allocation overhead per tick. |
| **Direct PyDict Construction** | Rust natively allocates Python dictionary objects holding primitive values. | Moderate latency; avoids string parsing but incurs heavy Python object creation overhead. |
| **Shared Memory Arrays** | Rust maintains flat memory buffers; exposes them to Python as multi-dimensional numeric arrays. | Minimal latency; true zero-copy interoperability, allowing instantaneous state visibility. |

Representing the spatial grid state as multidimensional numeric arrays and passing them to Python via shared buffers ensures that the Python agent manipulates the exact memory allocated by Rust, drastically reducing inter-process communication latency7. Utilizing this embedded execution approach avoids the severe serialization costs and network latencies typical of external microservices using protocols like gRPC or HTTP2, rendering the synchronous invocation of agent logic highly efficient7. While external PyPy microservices might theoretically execute internal Python loops faster, the network and serialization penalties completely overwhelm the interpreter speed gains for the high-frequency tick rates required by Kaggriculture7. Furthermore, utilizing standard arrays as the "lingua franca" between the languages prevents massive tensor copying every time a state transitions from Rust to Python7.

### **Function Invocation Profiling and Execution Overhead**

The Kaggriculture engine operates by passing agent functions directly into the evaluation loop, or by resolving file paths to dynamic Python scripts1. A standard competitive agent is invoked sequentially at every game tick. Benchmarking the execution boundary reveals that single, isolated invocations of Python functions from Rust exhibit significant latency anomalies. Profiling demonstrates that executing a function once incurs a massive processing and initialization overhead, whereas running the identical function millions of times in a tight loop averages out to highly optimized, sub-microsecond latency8.

This phenomenon occurs because CPython heavily caches intermediate compiled units, applies internal optimizations, and manages live memory pooling across repeated executions6. When integrating the modular agent framework, the Rust harness must account for this by incorporating a strict "warm-up" phase. Before formal, timed evaluation begins, the engine should execute the agent function against a series of dummy observation states to initialize all internal caches, pre-compile the dynamic bytecode, and stabilize the memory allocator. Failing to warm up the execution environment leads to heavily skewed benchmarking results where the first map evaluation appears artificially slow compared to subsequent runs8.

Furthermore, optimizing the argument passing convention at the Rust-C-Python boundary yields substantial performance gains. By default, complex function signatures often utilize legacy calling conventions that dynamically construct expensive Python tuples and dictionaries for argument parsing9. The Rust harness can be engineered to utilize optimized C-level calling conventions where supported, replacing dynamic tuple creation with a raw array of argument pointers9.

| Execution Benchmark Configuration | Calling Convention | Mean Execution Latency |
| :---- | :---- | :---- |
| Default FFI Boundary | Standard Tuple/Dict Allocation | 67.8 ns |
| Pure Python Function | Native Interpreter | 43.0 ns |
| Optimized FFI Boundary | Raw C-Array Pointer Passing | 24.8 ns |

This micro-optimization reduces the internal argument parsing latency overhead—which typically hovers around 44 nanoseconds per call—providing an execution speed that frequently surpasses native Python-to-Python function invocations9. When scaling the diagnostic benchmarker to evaluate millions of environment ticks across thousands of parallel matches, shaving 40 nanoseconds off the function boundary translates to hours of saved computational time9.

## **Diagnostic Benchmarker and Local Harness Code Structures**

The Kaggle diagnostic benchmarker notebook serves as the standard template for local agent evaluation, providing an object-oriented, modular framework for testing code structures and agent logic3. To adapt this dynamic Python framework into a compiled, strongly typed Rust binary, the conceptual structure must be formalized into strictly typed traits and generic execution loops that enforce compile-time safety without sacrificing runtime flexibility.

### **Engine and Agent Trait Implementations**

The core architecture dictates a strict encapsulation of the Kaggriculture environment rules from the arbitrary agent logic. The structural entities in Rust are defined to manage state transitions cleanly, entirely preventing Python-specific types from leaking into the core simulation logic.

The primary abstraction is the SimulationEngine. This structure maintains the ground-truth game state, enforces movement and harvesting rules, calculates internal economic scores, and triggers discrete game ticks. Because it is implemented entirely in Rust, it operates exclusively on pure data structures and enums, ensuring maximum instruction cache locality and cache-line efficiency.

Interfacing with this engine is the AgentWrapper, which acts as the absolute boundary between the determinism of the Rust simulation and the unpredictability of the Python agent. The wrapper implements a standard Rust trait but internally holds pointers representing the loaded agent function or module6. Standard Kaggle submission agents dynamically determine their actions based on the observation and configuration parameters passed by the environment at every tick1. The AgentWrapper is responsible for acquiring the GIL, securely converting the Rust observation state into a readable Python format, invoking the Python callable, and subsequently decoding the returned action dictionary back into a strictly validated Rust enumeration.

The highest level of orchestration is handled by the DiagnosticBenchmarker component. This system schedules the match configurations, manages the parallel thread pools, collects detailed execution traces, and calculates the final Elo or TrueSkill metrics. Upon the conclusion of a match, it formats the simulation results into Kaggle-compliant JSON replays, allowing developers to utilize downstream visualizers directly3.

### **The Evaluation Smoke Test and Error Propagation**

A critical component derived from standard Kaggle methodologies is the environment smoke test11. Before deploying any newly updated agent into massive, compute-intensive training pipelines, the agent must be subjected to live mini-simulations. These smoke tests evaluate the agent locally against extreme edge-case states to verify robust error handling. The diagnostic benchmarker mandates two primary conditions for a successful pass: the test must complete without raising any Python exceptions, and the evaluation summary must report zero runtime environment errors12.

When calling Python code from a native language, unhandled exceptions in the Python script do not simply crash the process; they surface across the FFI boundary as specific error structs within Rust6. The test harness utilizes the interoperability error handling APIs to intercept these exceptions, extract the traceback strings, and log the exact failure directly to the diagnostic benchmarker output without unwinding the Rust execution stack6.

Instead of allowing the main Rust simulation engine to panic and crash the entire test suite, the AgentWrapper intercepts the exception and safely converts it into a standard Rust Result::Err. This graceful error propagation signals the SimulationEngine to immediately default the current game state to an agent loss due to code failure. This mirrors the precise penalty and sanitization mechanisms employed by the official Kaggle evaluation servers, ensuring that local testing provides high-fidelity feedback mirroring production behavior11.

## **Parallel Execution Strategies and Concurrency Models**

To radically improve testing and training throughput—a mandatory requirement for gathering statistically significant reinforcement learning metrics and evolutionary algorithms—the harness must extensively parallelize the execution of simulated matches2. Given that the entire Kaggriculture environment logic is rewritten in memory-safe Rust, the core engine itself is highly amenable to multi-threading. By integrating data parallelism libraries, sequential iterators across pending matches can be instantly replaced with parallel chunking mechanisms that distribute the workload dynamically across all available CPU cores, abstracting away the complex thread management and synchronization primitives5.

### **The Global Interpreter Lock Bottleneck**

The primary structural obstacle to achieving linear scaling in this mixed-language architecture is the CPython Global Interpreter Lock (GIL). Prior to the fully supported free-threaded Python architectures (which only become fully viable in Python 3.13 and 3.14), the GIL strictly ensures that only one operating system thread can interact with the Python interpreter and manipulate Python objects at any given time6.

If a hardware thread pool of 64 Rust worker threads attempts to run 64 simultaneous games, the non-Python logic—such as engine state transitions, spatial collision rule checking, and score incrementing—will execute flawlessly in parallel. However, the precise moment the workers need to query their respective Python agents for an action, they must acquire the GIL. This causes all 64 threads to block each other, effectively funneling the heavily parallelized execution down into a single-threaded bottleneck, destroying the throughput advantages of the Rust engine6.

To mitigate this catastrophic lock contention, the engine implements a highly optimized scheduling architecture designed to maximize the time threads spend fully detached from the Python interpreter. The synchronization abstractions provided by the FFI layer allow operations that do not strictly require Python object manipulation to completely release the lock6. Whenever the Rust engine is calculating the complex mechanics of the environment update step, it explicitly drops the GIL, allowing another thread to wake up and query its agent.

### **Batched Inference and Multiprocessing Mitigation**

In advanced competition configurations, agents frequently rely on heavy Machine Learning models, such as scikit-learn estimators or deep neural networks built in PyTorch. In these scenarios, standard GIL-dropping threading remains insufficient due to the sheer duration of the inference computations. To overcome this, the test harness architecture can deploy two distinct, highly scalable parallel execution topologies depending on the agent's internal framework.

The first strategy is Process-Level Parallelism, commonly analogous to sub-interpreters or multiprocessing. Instead of sharing one embedded Python interpreter across the entire Rust process space, the harness orchestrates a fleet of independent child processes. Each process contains its own isolated Python interpreter and, critically, its own dedicated GIL. The main Rust process acts as a central coordinator, simulating the exceptionally fast game engine logic internally and dispatching serialized states to the child agent processes via fast Inter-Process Communication techniques, such as memory-mapped files or named pipes. While this bypasses the GIL completely, it introduces memory overhead as each agent requires its own localized model weights and Python runtime.

The second, more elegant strategy is Batched GIL Acquisition utilizing an Actor-Model pattern. The Rust simulation threads execute in parallel until they reach a boundary requiring an agent action. Instead of individually blocking to acquire the GIL, the threads yield their current environment states to a centralized InferenceCoordinator queue. Once a sufficient batch of states is accumulated (e.g., 32 or 64 concurrent match states), a single dedicated thread acquires the GIL. This thread converts all accumulated states into one monolithic multi-dimensional array and queries the Python agents in a vectorized format7.

| Parallel Strategy | GIL Impact | Resource Utilization | Optimal Use Case |
| :---- | :---- | :---- | :---- |
| **Naive Threading** | High Contention | Low Memory | Simple heuristic Python scripts. |
| **Process-Level Isolation** | Zero Contention | High Memory | Agents requiring conflicting library versions. |
| **Batched Inference** | Minimal Contention | Moderate Memory | Neural network agents utilizing vectorized models. |

By vectorizing the inference step, the architecture minimizes the total number of GIL context switches, leveraging the massive parallel processing power of hardware accelerators like GPUs if the agent utilizes them. Furthermore, maintaining stateful Python objects in memory across repeated predictions eliminates the catastrophic latency penalty of constantly re-initializing agent weights and model architectures at every game tick7.

## **Seat-Swapping Mechanisms and Asymmetric Evaluation Mathematics**

In highly strategic environments like Kaggriculture, map generation, initial resource allocation, and turn-order mechanics frequently introduce inherent asymmetries into the game state. A foundational observation in the Kaggle meta-analysis is that playing from different starting positions, or "seats," inherently impacts the statistical likelihood of success. An agent might exhibit dominant, aggressive characteristics when initialized as Player 1 but struggle severely when forced into a defensive posture as Player 211. Consequently, an accurate local harness cannot mathematically rely on a single match configuration to determine agent superiority.

### **Algorithmic Variance Reduction and TrueSkill Calibration**

To combat spatial and initialization bias, the test harness implements mandatory seat-swapping mechanisms for all pairwise agent evaluations. When evaluating Agent A against Agent B, determining true algorithmic dominance requires isolating the agent's logic from the environmental noise. If an agent heavily relies on a specific map layout, its win rate will heavily fluctuate based on the pseudo-random number generator (PRNG) seed of the match.

The diagnostic benchmarker fundamentally addresses this by enforcing mirrored pairings11. Let the exact evaluation configuration for a single, deterministic map seed be denoted as a function of the seed. The resulting internal score metric for Agent A, playing from seat index ![][image1] against Agent B from seat index ![][image2], is calculated across the simulation. To compute the true comparative performance delta, the harness averages the results of two distinct matches: one where Agent A is Player 1, and an immediate follow-up match where Agent A is Player 2 on the exact same map seed11.

If the standard deviation of this combined metric remains statistically significant across hundreds of unique, procedurally generated environment seeds, the benchmarker can confidently declare one agent algorithmically superior. The Rust framework natively enforces this requirement at the scheduler level. For every generated map seed requested by the user, the engine automatically pushes two mirrored match instances onto the parallel execution queue, reversing the indices of the AgentWrapper array passed into the second instance.

This strict clean-room evaluation methodology prevents the phenomenon known as "ladder anxiety," where a competitor cannot trust their local validation scores because of high environmental variance. When generating massive leaderboards locally, the engine calculates internal Elo or TrueSkill ratings exclusively based on these paired interactions. This guarantees that no agent climbs the evaluation ranks purely by fortuitously sampling favorable initial spatial positions.

## **Replay-Driven Clean-Room Research and Validation Strategy**

A unique and defining feature of the Kaggle competitive ecosystem is the public availability of detailed match replays. These replays consist of comprehensive historical traces containing the exact initial game state and the sequential list of actions taken by all players at every tick, strictly serialized into JSON structures. Advanced competitors leverage these replays not just for visual analysis, but to reverse-engineer opponent behaviors or construct highly adaptive response models without ever possessing access to the opponent's raw source code11.

### **Replay Ingestion and Deterministic Engine Validation**

The Rust harness is engineered to natively ingest these server-generated Kaggle replays, utilizing them as a dual-purpose tool for both engine validation and agent adaptation. Because a replay is a historical record of one complete match rather than an executable script11, the parser must meticulously reconstruct the timeline.

The foremost utility of replay ingestion is proving that the Rust engine is an exact, flawless clone of the original Python environment. To achieve this, the harness operates in a strict replay-verification mode. It reads a production trace downloaded directly from the Kaggle servers13. The parser extracts the original initialization seed and the array of actions submitted by the players at every tick. The Rust engine is then initialized with the identical seed. Instead of querying live AgentWrapper instances for decisions, the engine intercepts the execution loop and injects the historical actions directly into its state transition function.

At the conclusion of every tick, the resulting internally calculated Rust state is cryptographically hashed or directly field-by-field compared against the state observed in the Kaggle replay JSON. Any deviation, even a single mismatched resource integer, signifies a logic error or floating-point discrepancy in the Rust implementation. This automated verification loop allows developers to maintain absolute parity with the opaque server-side mechanics of the Kaggriculture environment.

### **Adaptive Replay Agents and Behavioral Cloning**

Beyond strict engine validation, the diagnostic benchmarker empowers researchers to train highly adaptive agents by treating historical replays as interactive, local environments14. The harness exposes an execution interface that continuously pipes historical states into the local agent logic1. This workflow facilitates the development of replay-derived, clean-room research candidates13. By evaluating an agent's intended action against the actual historical action taken by a high-ranking player in that exact state, the harness can compute a cross-entropy loss metric, effectively enabling local behavioral cloning.

This mechanism enables developers to locally simulate battling the highest echelons of the competition meta without requiring the opponent's private files. A live, locally developed agent can be pitted directly against a "Replay Agent." This Replay Agent is a lightweight execution wrapper that bypasses logic generation entirely; it simply reads the required action from the historical trace and submits it to the Rust engine.

To ensure the integrity of the evaluation against Replay Agents, the harness locks the internal random number generator seed to perfectly match the historical match. This forces the environment to unfold exactly as it did in the traced game, validating whether the local agent possesses the strategic depth necessary to disrupt the opponent's strategy from the opposing seat11. If the local agent manages to win the match, the researcher proves their new logic can outmaneuver the historical sequence of the top-tier competitor11.

## **Extensibility and the Future of Native Python Binding Ecosystems**

The highly modular architecture of the Kaggriculture test harness allows the Rust engine to scale computationally far beyond the limitations of standard Kaggle evaluation scripts. Because the framework establishes a rigidly defined boundary between the deterministic simulation layer (managed by Rust) and the probabilistic inference layer (managed by Python), the overarching system is remarkably extensible.

This architectural integration leverages the strengths of both ecosystems. Data scientists and machine learning engineers can remain entirely within the Python ecosystem—utilizing industry-standard libraries like NumPy, pandas, PyTorch, and scikit-learn for model training—while the heavy computational burden of millions of discrete game state transitions is handled transparently at native machine speed5.

Furthermore, the ecosystem provides robust deployment tooling. By leveraging build systems specifically designed for native extensions, the entire Rust harness can be compiled into a standalone binary wheel (.whl) and distributed seamlessly back to Python developers5. This allows end-users to install the highly optimized Rust engine via standard package managers and interact with it exactly as if it were a native Python module, completely abstracting the underlying systems-level complexity5. As asynchronous programming paradigms continue to dominate agent architectures, the integration of asynchronous bindings further streamlines the translation of non-blocking I/O operations between Python and Rust15.

Ultimately, by carefully managing execution lifetimes, avoiding excessive object serialization, leveraging zero-copy memory patterns, and mathematically balancing match evaluations through mirrored seat-swapping, the test harness achieves an order of magnitude improvement in throughput. It provides a mathematically rigorous, structurally safe, and exceptionally rapid execution environment that empowers the continuous evaluation and evolution of dominant Kaggriculture meta-strategies.

#### **Works cited**

> 1. Getting Started: Test Locally & Submit \- Kaggriculture | Kaggle, [https://www.kaggle.com/competitions/kaggriculture/overview/getting-started-test-locally-submit](https://www.kaggle.com/competitions/kaggriculture/overview/getting-started-test-locally-submit)  
> 2. Kaggriculture | Kaggle, [https://www.kaggle.com/competitions/kaggriculture/discussion/737013](https://www.kaggle.com/competitions/kaggriculture/discussion/737013)  
> 3. Kaggriculture | Kaggle, [https://www.kaggle.com/competitions/kaggriculture/discussion/737035](https://www.kaggle.com/competitions/kaggriculture/discussion/737035)  
> 4. kaggriculture\_utils\_v1 \- Kaggle, [https://www.kaggle.com/code/degnonguidi/kaggriculture-utils-v1](https://www.kaggle.com/code/degnonguidi/kaggriculture-utils-v1)  
> 5. How to Call Rust from Python | Towards Data Science, [https://towardsdatascience.com/calling-rust-from-python/](https://towardsdatascience.com/calling-rust-from-python/)  
> 6. Calling Python from Rust \- PyO3 user guide, [https://pyo3.rs/v0.29.2/python-from-rust](https://pyo3.rs/v0.29.2/python-from-rust)  
> 7. ML Predictions or: Calling Python functions from Rust FAST, [https://users.rust-lang.org/t/ml-predictions-or-calling-python-functions-from-rust-fast/80309](https://users.rust-lang.org/t/ml-predictions-or-calling-python-functions-from-rust-fast/80309)  
> 8. Rust pyo3 function is faster in python when ran multiple times, [https://stackoverflow.com/questions/73502325/rust-pyo3-function-is-faster-in-python-when-ran-multiple-times-instead-of-once](https://stackoverflow.com/questions/73502325/rust-pyo3-function-is-faster-in-python-when-ran-multiple-times-instead-of-once)  
> 9. PyO3 performance analysis: function overheads \#1607 \- GitHub, [https://github.com/PyO3/pyo3/issues/1607](https://github.com/PyO3/pyo3/issues/1607)  
> 10. Performance Comparison: Rust vs PyO3 vs Python. \- Reddit, [https://www.reddit.com/r/rust/comments/hv7u82/performance\_comparison\_rust\_vs\_pyo3\_vs\_python/](https://www.reddit.com/r/rust/comments/hv7u82/performance_comparison_rust_vs_pyo3_vs_python/)  
> 11. Kaggriculture: Findings from Zero to Top Meta \- Kaggle, [https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta)  
> 12. Kaggriculture: Local Agent Evaluation Harness \- Kaggle, [https://www.kaggle.com/code/mansiaggarwal88/kaggriculture-local-agent-evaluation-harness](https://www.kaggle.com/code/mansiaggarwal88/kaggriculture-local-agent-evaluation-harness)  
> 13. Kaggriculture | Multi-Route Farming Agent \- Kaggle, [https://www.kaggle.com/code/flexonafft/kaggriculture-adaptive-replay-agent?scriptVersionId=340601309](https://www.kaggle.com/code/flexonafft/kaggriculture-adaptive-replay-agent?scriptVersionId=340601309)  
> 14. Pokemon TCG AI: My Agent Won 95% and Was Still Bad, [https://alanscottencinas.com/pokemon-tcg-ai-agent/](https://alanscottencinas.com/pokemon-tcg-ai-agent/)  
> 15. Accelerating Python with Rust: The PyO3 Revolution \- YouTube, [https://www.youtube.com/watch?v=\_33zs20Sy0k](https://www.youtube.com/watch?v=_33zs20Sy0k)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAcCAYAAACtQ6WLAAAAkElEQVR4XmNgGBxAEIgZ0QVBgBWI/wOxJ7oECAgD8XMg1kSXgAEOdAGCgJkBYiwGUAHi00D8GojlkSVcgLifAeL8dCCOQJYEOdsUiDmBeAcQKyJLwoAOEL9nwBEADUD8D10QBPiB+AQQXwdiZSAORJa0AeLfQDwJiEsZ0BzlywAJUxC9nAGLf0HBJokuOOIBACXsEVMbAsH9AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAsklEQVR4XmNgGNqAFYiF0AXRgRgQnwPin+gS6EAdiF8C8XV0CbIADxALAzEjugQymADEF4D4IRBvB2J+VGkIAPkyhwFiki8Q/wfidBQVUBANpTmAeCsQfwViY4Q0JpAG4gdAfBqIBVGlUIENEP8G4vkMBDxUzoDHfTBAtLUuQPyPgQRrYSGAAmSB2AqImRkgwfIJiPVRVEDBawaEJCi1TAdiFhQVUBAMxLOAeBIDxNThDwBwKR3PC74dWAAAAABJRU5ErkJggg==>
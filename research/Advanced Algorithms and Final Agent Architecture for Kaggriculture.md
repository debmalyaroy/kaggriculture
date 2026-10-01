# **Detailed Design Document: 1-Week Hybrid Architecture for Kaggriculture**

This design document outlines the end-to-end architecture and step-by-step pipeline required to build a tournament-winning agent for the Kaggriculture competition in exactly one week. Given the strict constraints—10GB of top-tier replay data, a single RTX 4060 (8GB VRAM) laptop, and Kaggle's free tier—this architecture bypasses compute-heavy Reinforcement Learning. Instead, it relies on an **Offline Operations Research (OR) Pipeline** and a **Behavior Cloning (BC) Sprint**.

## **1\. System Architecture & Pipeline Distribution**

To survive the hardware constraints, the computational workload is strictly bifurcated between your local laptop and the Kaggle cloud instances.

\+-----------------------------------------------------------------------------------+

| PHASE 1: LOCAL LAPTOP (CPU & Rust Engine)                                                                      |

|                                                                                                                                                               |

| \[World Generator\] \-----\> \[SCIP MILP Solver\] \-----\> \[Tape Generator\]                           |

| (Game physics & rules) (Maximizes terminal coin) (Outputs 720-step static tape)    |

|                                                                                                                                                               |

| \[10GB Raw JSON Replays\] \-\> \[Rust Data Distiller\] \-\> \[.safetensors (1GB)\]                     |

\+-----------------------------------------------------------------------------------+

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| (Upload via Kaggle API)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;V

\+-----------------------------------------------------------------------------------+

| PHASE 2: KAGGLE CLOUD (L4x4 96GB GPU)                                                                         |

|                                                                                                                                                                |

| \[.safetensors Dataset\] \-----\> \[Decision Transformer BC Training\]                                    |

| (10-hour burst sessions, saves to /kaggle/working)                                                              |

\+-----------------------------------------------------------------------------------+

| (Download weights)

V

\+-----------------------------------------------------------------------------------+

| PHASE 3: SUBMISSION GENERATION (Local) |

| |

| \[Slot 1: Sparse Closed-Loop\] \[Slot 2: Adaptive BC Predator\] |

| (Uses Offline Tape \+ Market Rules) (Uses BC Weights \+ Action Masking) |

\+-----------------------------------------------------------------------------------+

## **2\. Phase 1: The Offline Operations Research (OR) Pipeline**

This pipeline runs entirely on your local laptop's CPU. Its goal is to generate the "baseline" physical farming mechanics so the agent never makes a biological mistake (e.g., forgetting to water a crop).

### **2.1 World Generator & SCIP Solver**

The **World Generator** is a Rust module that defines the deterministic state transitions of the Kaggriculture engine. You feed these transition rules into an open-source MILP solver like SCIP or OR-Tools.

The solver's objective is to maximize $Cash\_{720}$ by actively modeling the environment's dynamic price engine. Rather than assuming static prices, the MILP solver incorporates the exact non-linear price functions (e.g., quadratic glut curves, log shapes, and hinge constraints) and the steady supply drain caused by town shop consumption. Because the opponent's specific market actions are unknown during offline generation, the solver optimizes against an *isolated* market projection. It mathematically projects the expected price degradation caused by its own sales, perfectly balancing production volume against the market's capacity to absorb it before hitting the $1 price floor.

To build the offline Operations Research (OR) tape using a Mixed Integer Linear Programming (MILP) solver like SCIP, you must mathematically model the physical rules of the Kaggriculture simulation. The solver's goal is to find the exact sequence of actions that maximizes terminal wealth over 720 turns, which is then used as the rigid physical baseline for dynamic market wrappers.

Here is how the Kaggriculture mechanics translate into SCIP logic.

### **1\. Sets and Constants**

These represent the hardcoded rules of the game engine that do not change.

* **Time:** $T \= 720$ (Total turns, equivalent to 30 days $\\times$ 24 turns).  
* **Days:** $D \= 30$ (Used for daily resets like watering and animal feeding).  
* **Initial State:** $M\_{0} \= 3000$ (Starting money).  
* **Capacity:** $S\_{cap} \= 100$ (Maximum non-seed items in the shed).  
* **Asset Costs:** $C\_{seed}^{wheat} \= 10$, $C\_{animal}^{cow} \= 400$, etc.  
* **Labor Costs:** $C\_{hire}(n)$ follows the Fibonacci sequence $\\{1, 1, 2, 3, 5, 8, \\dots\\}$.

### **2\. Decision Variables (Actions)**

These are the integer variables SCIP controls at every turn $t \\in \\{0, \\dots, 719\\}$ to shape the farm's schedule.

* $x\_{plant, c, t} \\in \\mathbb{Z}^+$: Number of crop type $c$ planted at turn $t$.  
* $x\_{water, c, t} \\in \\mathbb{Z}^+$: Number of crop type $c$ watered at turn $t$.  
* $x\_{harvest, c, t} \\in \\mathbb{Z}^+$: Number of crop type $c$ harvested at turn $t$.  
* $x\_{feed, a, t} \\in \\mathbb{Z}^+$: Number of animal type $a$ fed at turn $t$.  
* $x\_{hire, t} \\in \\{0, \\dots, W\_{max}\\}$: Number of hands hired on the day encompassing turn $t$.  
* $x\_{buy, c, t} \\in \\mathbb{Z}^+$: Quantity of item $c$ purchased from the market.  
* $x\_{sell, c, t} \\in \\mathbb{Z}^+$: Quantity of item $c$ sold to the market.

### **3\. State Variables**

These variables track the consequences of the decision variables over time.

* $M\_t \\in \\mathbb{R}^+$: Total money available at turn $t$.  
* $I\_{c, t} \\in \\mathbb{Z}^+$: Inventory of item $c$ in the shed at turn $t$.  
* $P\_{c, t} \\in \\mathbb{Z}^+$: Number of mature plants of type $c$ alive at turn $t$.  
* $A\_{a, t} \\in \\mathbb{Z}^+$: Number of animals of type $a$ alive at turn $t$.

### **4\. The Objective Function**

The solver is instructed to maximize the money variable at the final turn of the season.

$$\\text{Maximize} \\quad M\_{719}$$

### **5\. Core Constraints**

SCIP relies on linear constraints to enforce the physical boundaries of the simulation.

**A. Labor Constraint (Action Budget):**

The total number of localized physical actions (planting, watering, harvesting, feeding) performed in a single turn cannot exceed the farmer plus the currently hired hands.

$$\\sum\_{c} x\_{plant, c, t} \+ \\sum\_{c} x\_{water, c, t} \+ \\sum\_{c} x\_{harvest, c, t} \+ \\sum\_{a} x\_{feed, a, t} \\le 1 \+ x\_{hire, t}$$

**B. Shed Capacity Constraint:**

The sum of all non-seed inventory must never exceed the 100-item limit. Any violation represents the destruction of inventory.

$$\\sum\_{c \\notin seeds} I\_{c, t} \\le 100 \\quad \\forall t$$

**C. Cash Flow Constraint (Liquidity):**

Money at the next turn equals current money, minus procurement and labor costs, plus sales revenue. SCIP uses a projected price curve $P(t)$ to estimate market revenue, ensuring $M\_t$ never drops below zero.

$$M\_{t+1} \= M\_t \- \\sum\_{c} (x\_{buy, c, t} \\cdot C\_{seed}^{c}) \- \\text{Fibonacci}(x\_{hire, t}) \+ \\sum\_{c} (x\_{sell, c, t} \\cdot P\_c(t))$$

**D. Biological Constraints (Watering and Feeding):**

To ensure crops don't turn into weeds and animals don't escape, you add logical constraints bridging the 24-turn daily windows. For every animal $A\_{a}$ alive on day $d$, there must be a corresponding feed action using wheat inventory.

$$\\sum\_{t \\in \\text{day } d} x\_{feed, a, t} \\ge A\_{a, d}$$

$$I\_{wheat, t+1} \= I\_{wheat, t} \- x\_{feed, a, t}$$

### **How it Works in Practice (PySCIPOpt)**

Because Kaggriculture is complex, generating the tape means defining these relationships in Python using a wrapper like PySCIPOpt.

Python

from pyscipopt import Model, quicksum

&nbsp;

\# Initialize solver

model \= Model("Kaggriculture\_Offline\_Tape")

&nbsp;

\# Define variables for a simplified 720-turn horizon

money \= {}

inventory\_wheat \= {}

hire\_hands \= {}

sell\_wheat \= {}

&nbsp;

for t in range(720):

&nbsp;&nbsp;&nbsp;&nbsp;money\[t\] \= model.addVar(vtype="C", name=f"M\_{t}", lb=0)

&nbsp;&nbsp;&nbsp;&nbsp;inventory\_wheat\[t\] \= model.addVar(vtype="I", name=f"Inv\_W\_{t}", lb=0, ub=100)

&nbsp;&nbsp;&nbsp;&nbsp;hire\_hands\[t\] \= model.addVar(vtype="I", name=f"Hire\_{t}", lb=0, ub=15)

&nbsp;&nbsp;&nbsp;&nbsp;sell\_wheat\[t\] \= model.addVar(vtype="I", name=f"Sell\_W\_{t}", lb=0)

&nbsp;

\# Set initial conditions

model.addCons(money\[0\] \== 3000\)

model.addCons(inventory\_wheat\[0\] \== 0\)

&nbsp;

\# Apply constraints across all turns

for t in range(719):

&nbsp;&nbsp;&nbsp;&nbsp;\# Action budget constraint (simplified: selling is a market action, planting/harvesting take farm actions)

&nbsp;&nbsp;&nbsp;&nbsp;\# Cash flow transition

&nbsp;&nbsp;&nbsp;&nbsp;model.addCons(money\[t+1\] \== money\[t\] \+ (sell\_wheat\[t\] \* 25\) \- hire\_hands\[t\]) \# simplified labor cost

&nbsp;&nbsp;&nbsp;&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;\# Inventory transition

&nbsp;&nbsp;&nbsp;&nbsp;model.addCons(inventory\_wheat\[t+1\] \== inventory\_wheat\[t\] \- sell\_wheat\[t\])

&nbsp;

\# Set Objective

model.setObjective(money\[719\], "maximize")

model.optimize()

&nbsp;

Once model.optimize() finishes, you extract the values of the decision variables (e.g., $x\_{hire, t}$, $x\_{plant, t}$) for every $t \\in 0 \\dots 719$. You save this precise output as a hardcoded Python array (tape.py). During the live Kaggle matches, your agent ignores the live game state for farming tasks and strictly executes the actions from the tape, wrapping it only with the dynamic market sales logic to protect against mirror-matches.

&nbsp;

### **2.2 Tape Generator**

Because running SCIP during the live Kaggle match would cause a timeout, the **Tape Generator** takes the solver's output and serializes it into a static Python dictionary1. This file (often around 300KB) is what you actually submit.

**Tape Generator Output Example (tape.py):**

&nbsp;

&nbsp;

&nbsp;

Python

\# Generated by local Rust/SCIP pipeline  
FARMER\_TAPE \= {  
&nbsp;&nbsp;&nbsp;&nbsp;0: \["PLANT", "WHEAT"\],  
&nbsp;&nbsp;&nbsp;&nbsp;1: \["WATER"\],  
&nbsp;&nbsp;&nbsp;&nbsp;\# ... fully populated to step 719  
}  
HANDS\_TAPE \= {  
&nbsp;&nbsp;&nbsp;&nbsp;0: \[\],  
&nbsp;&nbsp;&nbsp;&nbsp;100: \[\["HARVEST"\], \["PLANT", "MELON"\], \["WATER"\], \["WATER"\]\],  
&nbsp;&nbsp;&nbsp;&nbsp;\# ... Fibonacci-optimized labor scheduling  
}

## **3\. Phase 2: Data Distillation & Behavior Cloning**

To build the "Adaptive Predator" (Slot 2), you will utilize the 10GB of expert replay data.

### **3.1 Local Data Distillation (Laptop)**

Do not parse 10GB of JSON on Kaggle; it will crash the 17GB RAM limit. Write a Rust script to extract only the observations and actions from the winning player of each match. Compress these into chunked .safetensors files.

To process a 10GB JSON file on a machine with only 16GB of RAM, you must completely avoid loading the entire file into memory at once. If you attempt to use standard parsing methods (like `json.load()` in Python or reading the entire file string into memory in Rust), the object overhead will easily exceed your system's limits and cause an Out of Memory (OOM) crash.

Instead, you must use a **streaming (iterative) parsing strategy** coupled with **chunked disk writing**. Here is how to execute this within your local data distillation pipeline:

1. **Iterative JSON Parsing:**  
   * **If using Rust (Recommended for speed):** Use the `serde_json` crate, specifically `Deserializer::from_reader(file).into_iter()`. This allows you to stream the massive JSON file directly from your SSD, reading and discarding one game episode at a time. The memory is freed immediately after an episode is processed.  
   * **If using Python:** Do not use the default `json` library. Instead, use the `ijson` library, which yields JSON elements one by one as an iterator without holding the entire document tree in RAM.  
2. **Targeted Extraction:** You do not need to save the visual state, logging metadata, or engine overhead. As you stream the file, extract only the pure observations and actions. Your goal during this extraction is to identify if the exact 720-turn field and market schedules appear consistently across multiple games, filtering out noisy or purely reactive plays.  
3. **Chunked Tensor Serialization:** As you extract the arrays of states and actions, accumulate them in a fixed-size buffer in your RAM (e.g., an array that holds exactly 500MB worth of data). Once that buffer is full, pause the parser, serialize the buffer into a highly compressed `.safetensors` or `.npy` file, write it to your disk, and clear the memory buffer. Repeat this process until the 10GB file is fully consumed.

By utilizing a streaming parser and chunking your outputs, your active RAM footprint will never exceed the size of your single 500MB buffer, leaving plenty of overhead for your operating system. This safely distills the raw data down to manageable tensors, allowing you to train your models to recognize top-tier tactics like mirror-safe hysteresis and price-gated tranches.

Using **DuckDB** is an outstanding tactical decision, and given your strict 6-day deadline, it is actually highly preferable to writing a custom Rust parser from scratch.

DuckDB is explicitly designed for **out-of-core execution** (processing datasets larger than your system RAM) and vectorized query execution. It will seamlessly handle your 10GB JSON file on your 16GB laptop without triggering an Out of Memory (OOM) crash, saving you precious days of software engineering.

Here is exactly how to replace the custom Rust parser with a Python-based DuckDB pipeline to prepare your dataset for Kaggle:

### **1\. Why DuckDB Solves the 16GB RAM Constraint**

When you query a 10GB JSON file using DuckDB's `read_json_auto` function, DuckDB does not load the entire document tree into memory. Instead, it streams the JSON from your SSD, processing it in localized chunks. If the intermediate data exceeds your available RAM, DuckDB automatically pages (spills) the overflow to your SSD, guaranteeing your laptop will not crash.

### **2\. The DuckDB Data Distillation Pipeline**

Instead of writing complex Rust structs to deserialize the game states, you can use a few lines of SQL to extract the `observations` and `actions`, filter for the winning player, and export the data directly into compressed **Parquet** chunks. Parquet is highly compressed and natively supported by PyTorch and HuggingFace, making it perfect for your Kaggle upload.

Here is the exact Python code to run on your laptop to distill the 10GB dataset:

Python

import duckdb

&nbsp;

\# 1\. Connect to a persistent DuckDB file (this allows it to use the disk for swap space)

con \= duckdb.connect('kaggriculture\_distillation.db')

&nbsp;

\# 2\. Configure DuckDB to strictly respect your 16GB laptop limit

con.execute("PRAGMA memory\_limit='10GB';")

con.execute("PRAGMA threads=8;") \# Adjust based on your CPU cores

&nbsp;

\# 3\. Stream the JSON, filter for the winner, and export to 500MB Parquet chunks

\# Note: Adjust the JSON path / UNNEST logic based on the exact Kaggle replay structure

query \= """

COPY (

&nbsp;&nbsp;&nbsp;&nbsp;SELECT&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;step,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;observation,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;action

&nbsp;&nbsp;&nbsp;&nbsp;FROM read\_json\_auto('10GB\_top\_replays.json')

&nbsp;&nbsp;&nbsp;&nbsp;WHERE final\_rank \= 1  \-- Only clone the behavior of the winning agent

) TO 'kaggle\_dataset/'&nbsp;

(FORMAT PARQUET, PER\_THREAD\_OUTPUT TRUE, FILE\_SIZE\_BYTES 500000000);

"""

&nbsp;

print("Starting streaming extraction. This will take a few minutes...")

con.execute(query)

print("Data successfully distilled into 500MB Parquet chunks\!")

&nbsp;

### **3\. Uploading and Training on Kaggle**

Once DuckDB finishes, you will have a folder (`kaggle_dataset/`) containing several heavily compressed Parquet files (e.g., `data_0.parquet`, `data_1.parquet`), reducing your 10GB JSON footprint to roughly 1-2GB.

You upload this folder to Kaggle. Because you exported them as Parquet files, you do not need to build a complex data loader for your Decision Transformer. You can stream them directly into your PyTorch training loop using the HuggingFace `datasets` library, which also uses memory-mapping to keep RAM usage low:

Python

from datasets import load\_dataset

import torch

&nbsp;

\# Load the DuckDB-generated Parquet chunks directly in your Kaggle notebook

dataset \= load\_dataset("parquet", data\_files="/kaggle/input/your-dataset-name/\*.parquet")

&nbsp;

\# Stream the batches directly to your L4x4 GPU for Behavior Cloning

for batch in dataset\["train"\].iter(batch\_size=128):

&nbsp;&nbsp;&nbsp;&nbsp;obs\_tensor \= torch.tensor(batch\["observation"\]).to("cuda")

&nbsp;&nbsp;&nbsp;&nbsp;action\_tensor \= torch.tensor(batch\["action"\]).to("cuda")

&nbsp;&nbsp;&nbsp;&nbsp;

&nbsp;&nbsp;&nbsp;&nbsp;\# ... execute forward pass and backprop ...

&nbsp;

### **Summary of the Time-Saving Impact**

By substituting the Rust parser with DuckDB, you compress a 2-day low-level programming task into a 10-minute SQL query. This allows you to finish your data processing on Day 1, giving you maximum time to utilize your Kaggle GPU quotas to train your Behavior Cloning predator agent (Slot 2\) before the September 23 deadline.

&nbsp;

### **3.2 Burst Training (Kaggle L4x4 GPU)**

Upload the .safetensors to Kaggle. You will train a sequence model (Decision Transformer) to imitate the expert players. Because Kaggle sessions die after 12 hours, your PyTorch code must save checkpoints periodically.

**BC Training Loop Snippet (Kaggle Notebook):**

&nbsp;

&nbsp;

&nbsp;

Python

import torch  
import torch.nn as nn  
from safetensors.torch import load\_file

class BehaviorCloningAgent(nn.Module):  
&nbsp;&nbsp;&nbsp;&nbsp;def \_\_init\_\_(self):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;super().\_\_init\_\_()  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\# Simplified Transformer for sequence modeling  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.encoder \= nn.Linear(OBSERVATION\_DIM, 128)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.transformer \= nn.TransformerEncoderLayer(d\_model=128, nhead=4)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.head \= nn.Linear(128, ACTION\_DIM)

&nbsp;&nbsp;&nbsp;&nbsp;def forward(self, x):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x \= self.encoder(x)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x \= self.transformer(x)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return self.head(x)

\# Burst Training loop  
model \= BehaviorCloningAgent().to("cuda")  
optimizer \= torch.optim.AdamW(model.parameters(), lr=1e-4)

\# Load chunked expert data  
data \= load\_file("/kaggle/input/kaggriculture-expert-data/chunk\_0.safetensors")

for epoch in range(MAX\_EPOCHS):  
&nbsp;&nbsp;&nbsp;&nbsp;optimizer.zero\_grad()  
&nbsp;&nbsp;&nbsp;&nbsp;logits \= model(data\["observations"\].to("cuda"))  
&nbsp;&nbsp;&nbsp;&nbsp;loss \= nn.CrossEntropyLoss()(logits, data\["actions"\].to("cuda"))  
&nbsp;&nbsp;&nbsp;&nbsp;loss.backward()  
&nbsp;&nbsp;&nbsp;&nbsp;optimizer.step()  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# Crucial: Save checkpoint to working directory for session recovery  
&nbsp;&nbsp;&nbsp;&nbsp;torch.save(model.state\_dict(), f"/kaggle/working/bc\_checkpoint\_ep{epoch}.pt")

## **4\. Phase 3: Building The Submissions**

You will submit two distinct agents. This ensures you have a mathematical rating floor (Slot 1\) and a high-variance top-10 contender (Slot 2).

### **4.1 Submission Slot 1: Sparse Closed-Loop (The Anchor)**

This agent runs the static OR tape for all physical farming but uses dynamic, rule-based logic for selling1. Because the shared market forces competitors to shift from "what to build" to "when to sell", this agent monitors the live price engine to prevent dumping inventory into a crashed market2. It implements two advanced heuristics:

> 1. **Mirror-Safe Hysteresis:** Checks if the opponent is running a cloned strategy (e.g., the exact same land and labor scaling). If so, it front-runs their expected market dump3.  
> 2. **Price-Gated Tranches:** Premium items (MILK, WOOL, STRAWBERRY, MELON) are released in small, bounded quantities only when the market price exceeds a hardcoded gate3.

**Slot 1 Full Code Structure (main.py):**

&nbsp;

&nbsp;

&nbsp;

Python

from tape import FARMER\_TAPE, HANDS\_TAPE, PROCUREMENT\_TAPE

PRICE\_GATES \= {"MELON": 120, "MILK": 100, "STRAWBERRY": 80, "WOOL": 150}

def is\_mirror\_match(my\_farm, opp\_farm):  
&nbsp;&nbsp;&nbsp;&nbsp;"""Detects if opponent is running an identical 'Staged Economic Herd'."""  
&nbsp;&nbsp;&nbsp;&nbsp;return len(my\_farm.get("unlocked\_quadrants", \[\])) \== len(opp\_farm.get("unlocked\_quadrants", \[\]))

def price\_gated\_release(shed, market\_prices, is\_mirrored):  
&nbsp;&nbsp;&nbsp;&nbsp;"""Dynamically sorts and releases SELL orders."""  
&nbsp;&nbsp;&nbsp;&nbsp;dynamic\_sells \= \[\]  
&nbsp;&nbsp;&nbsp;&nbsp;for item, qty in shed.items():  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if qty \<= 0 or item \== "WHEAT":  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;continue  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;current\_price \= market\_prices.get(item, 0)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;gate\_price \= PRICE\_GATES.get(item, 1)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\# Front-run the opponent if farms are mirrored  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if is\_mirrored or current\_price \>= gate\_price:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\# Release in bounded tranches to limit non-linear price damage  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dynamic\_sells.append(\["SELL", item, min(qty, 12)\])  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# Sort to sell items closest to their 1-coin floor first  
&nbsp;&nbsp;&nbsp;&nbsp;dynamic\_sells.sort(key=lambda x: market\_prices.get(x\[1\], 0), reverse=True)  
&nbsp;&nbsp;&nbsp;&nbsp;return dynamic\_sells

def agent(obs):  
&nbsp;&nbsp;&nbsp;&nbsp;step \= obs.get("step", 0)  
&nbsp;&nbsp;&nbsp;&nbsp;player \= obs\["player"\]  
&nbsp;&nbsp;&nbsp;&nbsp;opp \= 1 if player \== 0 else 0  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# 1\. Sparse physical execution (Never deviates from the OR tape)  
&nbsp;&nbsp;&nbsp;&nbsp;farmer\_action \= FARMER\_TAPE.get(step, \["PASS"\])  
&nbsp;&nbsp;&nbsp;&nbsp;hands\_actions \= HANDS\_TAPE.get(step, \[\])  
&nbsp;&nbsp;&nbsp;&nbsp;market\_actions \= PROCUREMENT\_TAPE.get(step, \[\])  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# 2\. Closed-Loop Market Evaluation  
&nbsp;&nbsp;&nbsp;&nbsp;is\_mirrored \= is\_mirror\_match(obs\["farms"\]\[player\], obs\["farms"\]\[opp\])  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# 3\. Inject dynamic sales into the pre-planned market queue  
&nbsp;&nbsp;&nbsp;&nbsp;dynamic\_sells \= price\_gated\_release(obs\["private"\]\["shed"\], obs\["market"\]\["prices"\], is\_mirrored)  
&nbsp;&nbsp;&nbsp;&nbsp;market\_actions.extend(dynamic\_sells)  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;return {"farmer": farmer\_action, "hands": hands\_actions, "market": market\_actions}

### **4.2 Submission Slot 2: Adaptive BC Predator**

This agent utilizes the PyTorch weights generated on Kaggle. However, neural networks occasionally hallucinate mathematically impossible actions (e.g., trying to harvest an empty tile). Because you are running this on standard CPU Kaggle nodes during the actual match inference, you must implement **Action Masking** using a lightweight Python state tracker to prevent illegal moves.

**Slot 2 Full Code Structure (main.py):**

&nbsp;

&nbsp;

&nbsp;

Python

import torch  
import numpy as np

\# Load the weights trained during Phase 2  
MODEL \= BehaviorCloningAgent()  
MODEL.load\_state\_dict(torch.load("bc\_checkpoint\_final.pt", map\_location="cpu"))  
MODEL.eval()

def generate\_action\_mask(obs):  
&nbsp;&nbsp;&nbsp;&nbsp;"""  
&nbsp;&nbsp;&nbsp;&nbsp;Creates a boolean mask of legal actions based on current constraints&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;(e.g., shed capacity, available cash, tile occupancy).  
&nbsp;&nbsp;&nbsp;&nbsp;"""  
&nbsp;&nbsp;&nbsp;&nbsp;mask \= np.ones(ACTION\_DIM, dtype=bool)  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# Example constraint: Mask out BUY\_LAND if cash \< 1000  
&nbsp;&nbsp;&nbsp;&nbsp;if obs\["farms"\]\[obs\["player"\]\]\["money"\] \< 1000:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mask\[ACTION\_INDEX\_BUY\_LAND\] \= False  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;return mask

def agent(obs):  
&nbsp;&nbsp;&nbsp;&nbsp;\# 1\. Convert nested JSON observation to flat normalized tensor  
&nbsp;&nbsp;&nbsp;&nbsp;obs\_tensor \= preprocess\_observation(obs)  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# 2\. Generate Action Mask to prevent illegal moves  
&nbsp;&nbsp;&nbsp;&nbsp;legal\_mask \= generate\_action\_mask(obs)  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;with torch.no\_grad():  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;logits \= MODEL(obs\_tensor)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# 3\. Apply the mask (Set illegal logits to \-infinity)  
&nbsp;&nbsp;&nbsp;&nbsp;logits\[\~legal\_mask\] \= \-float('inf')  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;\# 4\. Select the best legal action  
&nbsp;&nbsp;&nbsp;&nbsp;best\_action\_idx \= torch.argmax(logits).item()  
&nbsp;&nbsp;&nbsp;&nbsp;  
&nbsp;&nbsp;&nbsp;&nbsp;return format\_action\_output(best\_action\_idx)

## **5\. Summary: How to Choose and Manage Your Submissions**

With 1 week remaining, you will finalize and upload both slots prior to September 23\.

> 1. **Why Slot 1 is required:** The Bradley-Terry rating system severely punishes losses to lower-rated players. Slot 1 (Sparse Closed-Loop) guarantees you will absolutely crush standard heuristic bots, establishing a highly stable 2400-2500 rating floor. It mathematically cannot make an unforced error.  
> 2. **Why Slot 2 is required:** Slot 1 will plateau when opponents predict its tape. Slot 2 (The BC Predator) has learned the complex, multi-agent dynamics of the top 10 players. It can dynamically adapt to demand shocks and read the opponent's strategy, giving you the high-variance edge needed to breach the 2800+ rating brackets.

By running these side-by-side on the leaderboard, you leverage the stability of Operations Research and the adaptability of Machine Learning simultaneously.

## **6\. Rough Timeline**

Given you only have one week before the competition entry deadline on September 23, 2026, and the final submission on September 30, you must bypass exploratory research and execute a highly compressed pipeline. Since you already have the Rust engine and tape generator built, you possess a massive advantage in execution speed; Rust's zero-copy memory models and CPU performance will allow you to generate rollouts rapidly without bottlenecking your GPU.

Here is your day-by-day architectural sprint to finalize the agent within one week:

**Days 1-2: Finalize the Micro-Planner & Plan Repair (Rust)** Since your Rust deterministic tape is functioning, immediately implement the Hungarian-CBS pipeline to steer your laborers (PC-TAPF). Do not attempt to build a "clean slate" global replanner. Instead, implement Dependency-Directed Plan Repair using Rust memory arenas. When your upcoming neural network makes a 3-day market adjustment (like a shop unlock), your engine should quickly roll back only the specific fractured tasks and patch them locally without throwing away the rest of the valid tape.

**Day 3: Action Masking and Data Distillation** Extract the 10GB of Top 10 leaderboard replays. Filter this data to isolate the absolute best execution tapes and market timings. Simultaneously, implement strict tensor-shape and action-mask assertions in your Rust interface. If a labor route is physically impossible, mask the action so your neural network mathematically cannot select it. This drastically reduces the state space and is mandatory for training within a 1-week timeframe.

**Days 4-5: Behavior Cloning (BC) Initialization** You do not have time for pure reinforcement learning from random weights. Use your single high-end GPU (e.g., RTX 5090\) to run Behavior Cloning on the distilled 10GB dataset. Train a transformer-based policy—utilizing BF16 forward passes and `torch.compile` for speed—to perfectly imitate the Top 10 players' procurement and market liquidation timings. This supervised learning phase will give the model its foundational mechanics immediately.

**Day 6: RL Fine-Tuning via IMPALA** Take your strong Behavior Cloning checkpoint and transition to reinforcement learning. Have your Rust asynchronous CPU actors simulate games using this policy, writing short rollout sequences to lock-free shared-memory buffers for the GPU learner. Crucially, force the agent to play against a pool of its own frozen historical checkpoints and a delayed moving teacher network. This stabilizes the learning process, ensuring the agent optimizes for the final win/loss reward without catastrophically forgetting the baseline behaviors it learned during BC.

**Day 7: Validation and Kaggle Submission** Halt training and run a strict chronological holdout validation. Test your final policy against replays generated *after* your code freeze to ensure it has not overfit to historical data. Package your agent into a pure-Python runtime that interacts with your compiled Rust binary.

**Critical Deadline Reminder:** You must accept the competition rules on the Kaggle website and finalize any team mergers before 11:59 PM UTC on September 23, 2026\. Submit your two active slots: one running your most stable deterministic route, and one running your new RL-finetuned adaptive predator.

## **Appendix**

To build an agent capable of achieving a 97% win rate in a highly complex simulation like Kaggriculture, you should construct a hybrid pipeline that transitions from supervised imitation learning to offline reinforcement learning (RL) supported by a high-throughput testing harness.

Based on successful architectures used in similar adversarial Kaggle environments, here are the specific tactics to build your agent and harness:

**1\. Behavior Cloning (BC) Initialization** Begin by parsing replays exclusively from top-tier players to extract model observations and legal actions. Train your initial policy to perfectly imitate these expert moves. This solves the "cold start" problem; instead of relying on pure self-play to randomly discover how to navigate the grid or avoid starving livestock, BC provides these fundamental execution skills immediately.

**2\. Decision Transformer Architecture** Frame the simulation's decision-making process as a sequence modeling problem. By utilizing a Decision Transformer, the model processes the offline dataset as a continuous stream of states, actions, and rewards. This memory-based architecture excels at sequence pattern recognition from static datasets, allowing the agent to effectively map long-term credit assignment and condition its future actions on early-game variables, such as randomized shop unlocks.

**3\. Enhanced Trajectory Stitching** A primary weakness of standard sequence modeling is the inability to recover if the environment forces the agent off its memorized path (e.g., a random weed spawn disrupting a precise route). To counter this, integrate techniques like Generalized Advantage Estimation into your offline RL training to improve "trajectory stitching". This enables the model to dynamically combine optimal sub-trajectories from different expert games, allowing it to seamlessly shift to a recovery path when unexpected stochastic events occur.

**4\. RL Fine-Tuning Against Historical Pools** Once the basic mechanics are learned via BC, transition to fine-tuning the policy with RL, using the final game result as the primary reward signal. Crucially, the agent should not just play against its current self. It must be evaluated against a diverse pool of frozen historical opponents and a delayed copy of its own policy (a teacher model). This prevents catastrophic forgetting, stabilizes the learning process, and ensures the agent learns to counter a wide variety of strategies.

**5\. High-Throughput Training Harness** To generate the volume of data required for RL fine-tuning, you must build an infrastructure designed for speed. The harness should utilize asynchronous CPU rollout actors running the game simulation, which write short rollout sequences into shared-memory buffers. A GPU learner then consumes these rollouts, batches the inference requests, and periodically copies updated weights back to the inference workers.

**6\. Compute Requirements** This heavily optimized pipeline does not require massive supercomputing clusters. The architecture can be effectively trained over the course of about a week using a single high-end consumer GPU (such as an NVIDIA RTX 5090\) or a standard datacenter accelerator (like an NVIDIA V100). However, because managing the shared-memory rollout buffers and reconstructing state histories is highly memory-intensive, you will need a robust multi-core CPU and significant system RAM to prevent bottlenecks.

## **Alternate**

When you are starting from a "clean slate" on Day 1 of a competition with no top-player data to imitate, you cannot rely on Behavior Cloning or Decision Transformers. Instead, you must synthesize your own expert data using operations research, and guide your reinforcement learning (RL) with heavily constrained self-play.

Here is the step-by-step methodology to build an optimal agent from scratch:

**1\. Build a High-Throughput Local Simulator** Without Kaggle's public replays, you must generate your own data at scale. You need to build a highly optimized local simulation harness—often implemented in C++ or fast Python—capable of running thousands of asynchronous game environments simultaneously. This allows you to generate millions of interactions in a few hours, bypassing the lack of public data.

**2\. Generate "Clean Slate" Baselines via Operations Research (MILP)** Before applying machine learning, you must mathematically solve the deterministic parts of the game (crop growth times, shed capacity, labor costs). You can use Mixed Integer Linear Programming (MILP) solvers, such as OR-Tools, to jointly optimize the entire 30-day timetable and resource allocation under a "clean slate" heuristic that temporarily ignores the opponent's market interference. These solvers will output the absolute mathematical maximum yield for a single farm, providing your initial baseline algorithms (e.g., the optimal 12-cow schedule).

**3\. Implement Strict Action Masking** If you unleash an RL agent (like PPO) from scratch, it will waste massive amounts of compute randomly discovering basic engine mechanics—like the fact that buying a 4,000-coin land parcel with only 3,000 coins is impossible, or that not feeding cows causes them to flee. To prevent this, you must implement strict action masking. By programmatically masking out illegal, irrelevant, or instantly fatal actions at every step, you drastically reduce the size of the action space. The RL model uses this mask so that only valid action classes contribute to the loss, forcing the agent to only explore within the bounds of logical farm management.

**4\. Iterative Self-Play Against a Historical Pool** With your MILP baselines and action masks in place, you can begin RL training. Instead of random self-play, which is highly unstable, use an iterative pool approach:

* First, train your RL agent exclusively against the static, hardcoded MILP baseline bots you created in Step 2\.  
* Once the RL agent learns how to exploit the baseline's market timing and wins consistently, "freeze" a checkpoint of that RL policy.  
* Add this frozen checkpoint to a pool of historical opponents.  
* Continue training the live RL agent against a randomized mixture of the MILP bots and the frozen historical checkpoints.

This ensures your agent learns a robust, generalized market strategy because it must continuously adapt to beat its own previous iterations without catastrophically forgetting how to beat the original baselines.

## **Base Play**

To build an agent capable of achieving a 97% win rate in a highly complex simulation like Kaggriculture, you should construct a hybrid pipeline that transitions from supervised imitation learning to offline reinforcement learning (RL) supported by a high-throughput testing harness.

Based on successful architectures used in similar adversarial Kaggle environments, here are the specific tactics to build your agent and harness:

**1\. Behavior Cloning (BC) Initialization** Begin by parsing replays exclusively from top-tier players to extract model observations and legal actions. Train your initial policy to perfectly imitate these expert moves. This solves the "cold start" problem; instead of relying on pure self-play to randomly discover how to navigate the grid or avoid starving livestock, BC provides these fundamental execution skills immediately.

**2\. Decision Transformer Architecture** Frame the simulation's decision-making process as a sequence modeling problem. By utilizing a Decision Transformer, the model processes the offline dataset as a continuous stream of states, actions, and rewards. This memory-based architecture excels at sequence pattern recognition from static datasets, allowing the agent to effectively map long-term credit assignment and condition its future actions on early-game variables, such as randomized shop unlocks.

**3\. Enhanced Trajectory Stitching** A primary weakness of standard sequence modeling is the inability to recover if the environment forces the agent off its memorized path (e.g., a random weed spawn disrupting a precise route). To counter this, integrate techniques like Generalized Advantage Estimation into your offline RL training to improve "trajectory stitching". This enables the model to dynamically combine optimal sub-trajectories from different expert games, allowing it to seamlessly shift to a recovery path when unexpected stochastic events occur.

**4\. RL Fine-Tuning Against Historical Pools** Once the basic mechanics are learned via BC, transition to fine-tuning the policy with RL, using the final game result as the primary reward signal. Crucially, the agent should not just play against its current self. It must be evaluated against a diverse pool of frozen historical opponents and a delayed copy of its own policy (a teacher model). This prevents catastrophic forgetting, stabilizes the learning process, and ensures the agent learns to counter a wide variety of strategies.

**5\. High-Throughput Training Harness** To generate the volume of data required for RL fine-tuning, you must build an infrastructure designed for speed. The harness should utilize asynchronous CPU rollout actors running the game simulation, which write short rollout sequences into shared-memory buffers. A GPU learner then consumes these rollouts, batches the inference requests, and periodically copies updated weights back to the inference workers.

**6\. Compute Requirements** This heavily optimized pipeline does not require massive supercomputing clusters. The architecture can be effectively trained over the course of about a week using a single high-end consumer GPU (such as an NVIDIA RTX 5090\) or a standard datacenter accelerator (like an NVIDIA V100). However, because managing the shared-memory rollout buffers and reconstructing state histories is highly memory-intensive, you will need a robust multi-core CPU and significant system RAM to prevent bottlenecks.

&nbsp;

#### **Works cited**

> 1. 22/24 Unseen Lineages | v41 Sparse Closed Loop \- Kaggle, [https://www.kaggle.com/code/kaitofukami/22-24-unseen-lineages-v41-sparse-closed-loop/log?scriptVersionId=340030138](https://www.kaggle.com/code/kaitofukami/22-24-unseen-lineages-v41-sparse-closed-loop/log?scriptVersionId=340030138)  
> 2. Kaggriculture: Findings from Zero to Top Meta \- Kaggle, [https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta](https://www.kaggle.com/code/raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta)  
> 3. V29-R1 | Adaptive Market Hysteresis \- Kaggle, [https://www.kaggle.com/code/boatlee/v29-r1-adaptive-market-hysteresis](https://www.kaggle.com/code/boatlee/v29-r1-adaptive-market-hysteresis)
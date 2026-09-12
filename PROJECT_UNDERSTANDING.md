# 🧠 Project Understanding Guide: Buy or Wait?

This document is your **study guide** for the HackerRank Orchestrate Hackathon. It breaks down exactly how your AI financial agent works, its architecture, and the algorithms it uses. Read this to prepare for any questions the judges might ask!

---

## 🎯 1. High-Level Goal
The goal of this project is to build an **AI-powered financial decision agent**. For a given purchase request, the agent must decide if the user should pay in full, use installments, make a partial payment, wait, or decline the purchase. 

It makes this decision by deeply analyzing the user's financial life: reconstructing their cash flow, converting multiple currencies, processing text messages (e.g., "my salary is delayed"), reading receipts, and simulating their bank balance for the next **90 days** to ensure their balance never drops below their personal minimum threshold.

---

## 🏗️ 2. System Architecture & Workflow

The system is designed as a **7-Step Deterministic Pipeline**. The judges care about *reproducibility* and *correctness*, so the core financial engine is completely deterministic (rule-based), while LLMs (Large Language Models) are only used for extracting data from unstructured text/images.

Here is the step-by-step workflow:

### **Step 1: Ingest & Normalize (`ingest.py`)**
- Loads all CSV data (profiles, events, exchange rates, payment options).
- **Multi-Hop Currency Conversion**: Not all currencies have direct exchange rates. The engine uses a graph-based approach to convert currencies. For example, if we need IDR to ZAR, and we only have IDR->USD and USD->ZAR, the engine automatically routes the conversion through USD.

### **Step 2: Information Extraction (`llm_extract.py`)**
- **OCR (Images)**: For financial events with missing amounts (blank in the CSV), the agent looks at the associated receipt/payslip image. It can use a Vision LLM (like Llama 3.2 Vision) to extract the exact amount.
- **Message Classification**: Parses the user's recent chat messages to find crucial financial changes that override the CSV data. It looks for:
  - *Salary amendments* (e.g., a pay cut or raise).
  - *Date delays* (e.g., salary delayed to the 25th).
  - *Cancellations* (e.g., a contract was cancelled).
  - *Rent increases*.

### **Step 3 & 4: 90-Day Forecast Engine (`forecast.py`)**
This is the **brain** of the application. It projects the user's balance for the next 90 days.
- **The Balance Snapshot Insight**: A key design decision was realizing that the `current_available_balance` *already* includes all settled past events. We only project *future* events that haven't hit the bank yet.
- **Recurring Pattern Detection**: The engine scans past settled events to find recurring patterns (e.g., Netflix every 30 days). It groups them by `category` and `description`, calculates the average interval, and projects the *next* payment date.
- **Recency Filter**: It checks if a pattern is still active. If a recurring payment hasn't happened in months, it assumes the subscription was cancelled and stops projecting it.

### **Step 5: Plan Generation & Ranking (`plan_ranker.py`)**
Once we have the 90-day simulation, we test different payment options:
1. **Full Payment**: Is it affordable today without dropping below the minimum balance?
2. **Installments**: We simulate the exact installment schedule (e.g., 3 payments over 3 months) against the 90-day forecast to ensure *every single payment* is safe.
3. **Partial Payment**: Can we pay a chunk today, and the rest on a safe date before the deadline?
4. **Wait**: If it's not affordable today, on what future date does it become affordable?
5. **Spending Changes**: If it's not affordable, can we stop or reduce flexible subscriptions (like "entertainment") to make it affordable?

**Ranking**: It ranks the valid plans based on strict preferences: Meeting the deadline > Requiring no lifestyle changes > Lowest total cost > Earliest start > Fewest payments.

### **Step 6: Deterministic Verification (`verify.py`)**
Before outputting, the system runs a massive suite of invariant checks on every single decision to ensure it didn't break any hackathon rules (e.g., making sure partial payments exactly equal the requested amount, checking if dates are in chronological order, ensuring amounts don't go negative).

### **Step 7: Output Generation (`output_writer.py`)**
Writes the exact `output.csv` format required by the judges.

---

## 🧠 3. Key Algorithms to Mention to Judges

If the judges ask about the technical depth, talk about these three specific algorithms:

### A. "Binary Search for Affordability"
To calculate the `amount_safe_to_pay`, we don't just guess. We use a **Binary Search algorithm**.
We know the amount must be between `0` and `requested_amount`. The binary search tests a midpoint amount, simulates the next 90 days day-by-day, and checks if the balance ever drops below the minimum. If it drops, the amount is too high. If it stays safe, the amount might be too low. It converges on the exact penny that is safe to pay in about 60 iterations.

### B. "Double-Count Prevention in Forecasting"
A common mistake in financial forecasting is double-counting. If a user gets paid on the 15th, and today is the 16th, the salary is *already in the balance*. If the engine carelessly projects a 30-day recurring salary, it might add that salary again. Our engine explicitly checks the `settlement_date` of the last occurrence and only projects the *next* occurrence to prevent hallucinatory cash flows.

### C. "Multi-Hop Exchange Rate Graph"
Exchange rates were provided as a flat list. Our system builds an undirected graph of currency pairs. When converting from Currency A to Currency B, it checks for a direct rate, a reversed rate (1/rate), and then does a graph traversal to find a two-hop path (e.g., ZAR -> USD -> EUR).

---

## 💬 4. How to Explain the LLM Strategy

**Judge Question:** *"Why did you use `--no-llm` for the main run, or how does your LLM integration work?"*

**Your Answer:**
"The core of my agent is 100% deterministic, written in Python, because financial decisions and bank balances require absolute mathematical correctness. You cannot hallucinate a bank balance. 
However, I designed the pipeline to be a hybrid. The LLM (configured to use NVIDIA NIM or Groq) acts as a specialized 'Extractor Layer'. It only does what neural networks do best: OCR on messy receipt images and intent classification on chat messages. It converts that unstructured data into structured JSON, which is then fed back into the deterministic forecast engine. For testing and reliability during the hackathon, the pipeline can run entirely on fallbacks (`--no-llm`), guaranteeing a fast (0.3 seconds), consistent evaluation."

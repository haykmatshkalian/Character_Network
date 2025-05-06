# Armenian Character Network Analysis

This project uses **network graphs**, **NLP techniques** (including name entity recognition and optional sentiment analysis), and **GPT-based filtering** to analyze relationships among characters in Armenian literary texts. As an example, the system is designed to work with any Armenian novel in PDF format and outputs a clear, visual graph of character interactions.

---

## Why This Project?

The relationships among characters are often the heart of any compelling novel. In complex narratives, it’s easy to lose track of who’s who and how they’re connected—especially for readers who like to skim. This project was built to **automatically extract and visualize character networks** from literary works, helping readers, researchers, and literary enthusiasts better understand the structure of stories.

> Imagine loading a novel and instantly getting a complete visual map of the characters and how they relate—like CliffsNotes, but smarter and interactive.

---

## Features

- ✅ Extract Armenian text directly from any PDF novel
- ✅ Detect potential names using basic heuristics
- ✅ **Filter actual character names** using AI (via OpenRouter)
- ✅ Count co-occurrences based on paragraph proximity
- ✅ Visualize character networks with `networkx`

---

## Output Example

Once processed, the system produces a **co-occurrence network graph** where:

- **Nodes** = characters
- **Node size** = importance (based on frequency)
- **Edges** = relationships (based on shared paragraphs)

---

## Core Techniques

### 1. Name Entity Recognition (NER)
Identifies all capitalized or contextually probable names in Armenian text. Afterward, GPT filters the list to remove non-character terms like places or generic nouns.

### 2. GPT Filtering
Instead of manually maintaining a list of valid characters, GPT-4 is used to **refine the extracted names**, keeping only character names. This is especially helpful in texts where names aren't standard.

### 3. Co-occurrence Analysis
Characters appearing in the same paragraph are considered connected. The number of co-appearances is used to determine the edge strength.

### 4. (Optional) Sentiment Analysis
Analyzes tone and context to estimate sentiment between characters. Friendly relationships are rendered in bright colors, while hostile ones appear darker.

### 5. Network Graph Visualization
Constructed with `networkx` and rendered via `matplotlib`. Armenian font (`NotoSansArmenian`) ensures correct label rendering.

### 6. PySpark (Optional)
If processing massive novels or multiple books, Spark can be integrated for distributed computation of name recognition and sentiment scoring.

---

## Getting Started

### 1. Clone the Repository

git clone https://github.com/your-username/armenian-character-network.git
cd armenian-character-network


### 2. Place Your PDF

Put your Armenian novel (e.g., sample.pdf) in the root of the project folder.

### 3. Create a Virtual Environment (optional but recommended)

python3 -m venv venv
source venv/bin/activate


### 4. Install Required Python Packages

pip3 install -r requirements.txt

### 5. Set Up Environment Variables

Create a file named .env in the root folder:

OPENROUTER_API_KEY=your_openrouter_api_key_here

### 6. Run the Script

python3 main.py


### 7. View the Results
A graph will be saved as character_network.png in the same directory.

You’ll also see console output showing extracted and filtered names.

### 8. (Optional) Troubleshooting Tips
No names detected? Make sure the PDF contains Armenian character names.

API error? Double-check your API key and internet connection.




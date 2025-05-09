import fitz  # PyMuPDF
import os
import re
import torch
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import openai
import httpx
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from matplotlib import font_manager, rcParams

# ========== CONFIGURATION ==========
PDF_PATH = "sample.pdf"
TXT_PATH = "armenian_text.txt"
FONT_PATH = "NotoSansArmenian-Regular.ttf"
CHUNK_SIZE = 3000
OVERLAP = 500

# ========== STEP 1: Extract text from PDF ==========
def extract_text_from_pdf(pdf_path):
    cleaned_paragraphs = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw_text = page.get_text("text")
            paragraphs = re.split(r'\n{2,}', raw_text)
            for para in paragraphs:
                single_line = re.sub(r'\n+', ' ', para).strip()
                if single_line:
                    cleaned_paragraphs.append(single_line)
    return "\n\n".join(cleaned_paragraphs)

pdf_text = extract_text_from_pdf(PDF_PATH)
with open(TXT_PATH, "w", encoding='utf-8') as f:
    f.write(pdf_text)

# ========== STEP 2: Load NER Model ==========
tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
device = 0 if torch.cuda.is_available() else -1
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

# ========== STEP 3: Extract raw NER names ==========
def extract_ner_names(text):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - OVERLAP

    raw_names = []
    for chunk in chunks:
        raw_names.extend([
            entity["word"]
            for entity in ner_pipeline(chunk)
            if entity["entity_group"] == "PER"
        ])
    return list(set(filter(None, raw_names)))

# ========== STEP 4: Normalize names with OpenRouter ==========
openai.api_key = "Youre-API-Key"
openai.base_url = "https://openrouter.ai/api/v1"

def normalize_names_with_openrouter(ner_names):
    print("🔁 Normalizing names with OpenRouter...")

    prompt = (
        "You are an Armenian character name normalizer. "
        "From the input list of named entities (some might be inflected, duplicated or partial), "
        "extract and return a **unique list of canonical, full Armenian character names** in JSON array format. "
        "Do NOT return nicknames or fragments like 'Ս', 'Սամս', 'Սամսոնի' — only the complete version like 'Սամսոն'.\n\n"
        "Input:\n" + json.dumps(ner_names, ensure_ascii=False)
    )

    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "CharacterGraphApp",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You normalize Armenian character names for literary analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        print("OpenRouter Content:", content)

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        normalized = json.loads(content)
        if not isinstance(normalized, list):
            print("❌ Normalization error: Response was not a list")
            return []

        print(f"✅ Final character names ({len(normalized)}): {normalized}")
        return normalized

    except Exception as e:
        print(f"❌ Normalization error: {e}")
        return []

# ========== STEP 5: Build character graph ==========
def build_character_graph(text, character_names):
    paragraphs = text.split('\n\n')
    connections = {}
    frequencies = dict.fromkeys(character_names, 0)
    char_set = set(character_names)

    for para in paragraphs:
        found = [name for name in character_names if name in para]
        for name in found:
            frequencies[name] += 1
        for name1, name2 in itertools.combinations(found, 2):
            if name1 != name2:
                pair = tuple(sorted((name1, name2)))
                connections[pair] = connections.get(pair, 0) + 1

    G = nx.Graph()
    for name in character_names:
        G.add_node(name, size=frequencies[name] * 300)  # Node size factor

    for (n1, n2), weight in connections.items():
        G.add_edge(n1, n2, weight=weight)

    return G

# ========== STEP 6: Visualization ==========
def visualize_graph(G, font_path):
    if G.number_of_nodes() == 0:
        print("⚠️ No characters to visualize.")
        return

    try:
        font_prop = font_manager.FontProperties(fname=font_path)
        rcParams['font.family'] = font_prop.get_name()
    except Exception as e:
        print(f"⚠️ Could not load font: {e}. Using default font.")
        font_prop = None

    node_sizes = [G.nodes[n]["size"] for n in G.nodes]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]

    plt.figure(figsize=(12, 10))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color='skyblue',
            edge_color='gray', width=edge_weights, alpha=0.7)

    if font_prop:
        nx.draw_networkx_labels(G, pos, font_size=12, font_family=font_prop.get_name())
    else:
        nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Character Co-occurrence Graph", fontsize=16)
    plt.tight_layout()
    plt.show()

# ========== STEP 7: Run ==========
with open(TXT_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

print("Device set to use", "cuda" if torch.cuda.is_available() else "cpu")

raw_ner_names = extract_ner_names(text)
print(f"NER names ({len(raw_ner_names)}):", raw_ner_names)

character_names = normalize_names_with_openrouter(raw_ner_names)
print(f"Final character names ({len(character_names)}):", character_names)

graph = build_character_graph(text, character_names)
visualize_graph(graph, FONT_PATH)

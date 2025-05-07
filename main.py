import fitz  # PyMuPDF
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import ast
from openai import OpenAI
import matplotlib.font_manager as fm
import os
import re
from wordcloud import WordCloud

# ========== STEP 1: Extract text from PDF ==========

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

pdf_text = extract_text_from_pdf("sample.pdf")

with open("armenian_text.txt", "w", encoding='utf-8') as f:
    f.write(pdf_text)

with open("armenian_text(simple connections).txt", "w", encoding='utf-8') as f:
    f.write(pdf_text)

# ========== STEP 2: Load text ==========

with open('armenian_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ========== STEP 3: OpenRouter API setup ==========

client = OpenAI(
    api_key="YOUR_API_KEY",  # Replace with your actual OpenAI API key
    base_url="https://openrouter.ai/api/v1"
)

# ========== STEP 4: Utility functions ==========

def is_valid_armenian_name(word):
    return (
        len(word) > 1 and
        all('Ա' <= c <= 'ֆ' or c in "և" for c in word) and
        not any(char.isdigit() or 'a' <= char.lower() <= 'z' for char in word)
    )

def normalize_name(name):
    suffixes = ['ի', 'ին', 'իս', 'ով', 'ովա', 'ոյին', 'ս', 'ովան', 'ոն']
    for suffix in sorted(suffixes, key=len, reverse=True):
        if name.endswith(suffix) and len(name) - len(suffix) >= 2:
            return name[:-len(suffix)]
    return name

def extract_characters_from_text(text):
    all_names = set()
    chunk_size = 3000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    for chunk in chunks:
        print(f"Processing chunk:\n{chunk[:200]}")  # Log chunk content

        prompt = f"""
        Տես այս հայկական տեքստը և վերադարձիր միայն **մարդկային կերպարների անունների Python list**, առանց բացատրության և առանց ուրիշ բովանդակության։
        Օրինակ վերադարձի ձևը պետք է լինի այսպես՝ ["Արամ", "Լիլո", "Հովհաննես"]
        Խնդրում եմ միայն անուններ։

        Տեքստը՝
        {chunk}
        """
        try:
            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Արա այնպես, որ վերադարձնի միայն կերպարների անուններ Python list-ի տեսքով։ Կարևոր է՝ անունը սկսվում է մեծատառով և հայկական անուն է։"},
                          {"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content.strip()
            print(f"API response for chunk:\n{response_text}")  # Log API response

            if not response_text or not response_text.startswith("[") or not response_text.endswith("]"):
                print("⚠️ Empty or invalid response from API for chunk.")
                continue

            names = ast.literal_eval(response_text)
            if isinstance(names, list):
                valid_names = [
                    normalize_name(name.strip()) for name in names if is_valid_armenian_name(name.strip())
                ]
                all_names.update(valid_names)
            else:
                print("⚠️ Response was not a list:", response_text)

        except Exception as e:
            print(f"❌ Failed to parse names from chunk:\n{chunk[:200]} ...\nError: {e}")

    return list(all_names)

# ========== STEP 5: Extract characters ==========

character_names = extract_characters_from_text(text)
print("Characters extracted:", character_names)

# ========== STEP 6: Build co-occurrence network ==========

paragraphs = text.split('\n\n')
connections = {}
for para in paragraphs:
    found = [name for name in character_names if re.search(r'\b' + re.escape(name) + r'\w*\b', para)]
    for name1, name2 in itertools.combinations(found, 2):
        pair = tuple(sorted((name1, name2)))
        connections[pair] = connections.get(pair, 0) + 1

# ========== STEP 7: Build graph ==========

G = nx.Graph()
G.add_nodes_from(character_names)
for (name1, name2), weight in connections.items():
    G.add_edge(name1, name2, weight=weight)

# ========== STEP 8: Visualize ==========

pos = nx.spring_layout(G, seed=42, k=2.5)
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=2000)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 0.5 for w in weights], alpha=0.6)

font_path = "NotoSansArmenian-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

for node, (x, y) in pos.items():
    plt.text(
        x, y,
        node,
        fontsize=12,
        fontproperties=font_prop,
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.2')
    )

plt.title('Character Network', fontsize=16, fontproperties=font_prop)
plt.axis('off')
plt.tight_layout()
plt.show()

# ========== STEP 9: Generate word cloud ==========

if character_names:
    name_counts = {name: character_names.count(name) for name in set(character_names)}
    wordcloud = WordCloud(font_path=font_path, width=800, height=600).generate_from_frequencies(name_counts)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
else:
    print("No valid character names found. Word cloud cannot be generated.")

import json
import pickle

# Load JSON data
json_path = "data/validationset.json"
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)


# Load LangChain documents
pkl_path = "data/langchain_documents.pkl"
with open(pkl_path, "rb") as f:
    langchain_documents = pickle.load(f)

# Create mapping from 'source' to 'context' from LangChain documents
source_to_context = {
    doc.metadata.get("source"): doc.metadata.get("context")
    for doc in langchain_documents
    if "source" in doc.metadata and "context" in doc.metadata
}

def enrich_samples(samples):
    for sample in samples:
        retrieved_ids = sample.get("retrieved_contexts", [])
        full_contexts = [
            source_to_context.get(rid)
            for rid in retrieved_ids
            if rid in source_to_context
        ]
        sample["retrieved_contexts_full"] = full_contexts

# Apply to all lists under singleturn and multiturn
for section in ["singleturn", "multiturn"]:
    if section in json_data:
        for key, samples in json_data[section].items():
            if isinstance(samples, list):
                enrich_samples(samples)

# Save the updated data
with open("data/final_enriched.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print("Updated JSON saved as final_enriched.json")


import streamlit as st
import json
import random
import os
from collections import defaultdict
from prompt_templates import evaluation_templates, general_intro_prompt

DATA_FILE = "ratings_data.json"
VALIDATION_FILE = "validation_set.json"

# --- Helper Functions ---

def load_ratings():
    ratings = {"Experten": {}, "Crowd": {}}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                ratings = json.load(f)
            except json.JSONDecodeError:
                st.error("Error loading ratings data. The file may be corrupted.")
    return ratings

def save_ratings(ratings):
    with open(DATA_FILE, "w") as f:
        json.dump(ratings, f, indent=4)


def get_lowest_coverage_metric(validation_data, ratings, it_background):
    # Map IT background to keys in ratings.json
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    sample_metrics = {}
    for example in validation_data["examples"]:
        sample_id = example["id"]
        rubric = example["rubric"]
        for metric in validation_data["rubrics"].get(rubric, []):
            sample_metrics.setdefault(sample_id, []).append(metric)

    # Collect all metrics from the validation data
    all_metrics = set()
    for metrics in sample_metrics.values():
        all_metrics.update(metrics)

    metric_vote_counts = defaultdict(list)
    for sample_id, metrics in sample_metrics.items():
        for metric in metrics:
            # Check for votes for this metric and sample ID
            vote_count = ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("vote_count", 0)
            metric_vote_counts[metric].append(vote_count)

    metric_coverage = {}
    for metric in all_metrics:  # Iterate through ALL metrics
        vote_counts = metric_vote_counts[metric]
        if vote_counts:
            metric_coverage[metric] = sum(vote_counts) / len(vote_counts)
        else:
            metric_coverage[metric] = 0  # Metrics with no votes get coverage 0

    # Find the metric with the lowest coverage
    if not metric_coverage:
        return random.choice(list(all_metrics)) if all_metrics else None  # Handle case with no metrics

    lowest_coverage_metric = min(metric_coverage, key=metric_coverage.get)
    return lowest_coverage_metric


def get_samples_for_metric(metric, ratings, it_background):
    # Map IT background to keys in ratings.json
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    
    samples = [s for s in validation_set["examples"] if metric in validation_set["rubrics"].get(s["rubric"], [])]
    samples_with_vote_count = []
    for sample in samples:
        vote_count = ratings[ratings_key].get(sample["id"], {}).get(metric, {}).get("vote_count", 0)
        samples_with_vote_count.append((sample, vote_count))
    samples_with_vote_count.sort(key=lambda x: x[1])
    return [sample for sample, _ in samples_with_vote_count]

def generate_prompt(sample, metric):
    template = evaluation_templates[metric]["prompt"]
    required_attributes = evaluation_templates[metric]["required_attributes"]
    rating_scale = evaluation_templates[metric]["rating_scale"]

    sample_data = {}
    for attr in required_attributes:
        if attr in sample:
            sample_data[attr] = sample[attr]
        elif attr == "dialog" and "dialog" in sample:
            sample_data[attr] = "\n".join([f"{turn[0]}: {turn[1]}" for turn in sample["dialog"]])
        elif attr == "dialog_a" and "dialogs" in sample:
            sample_data[attr] = "\n".join([f"{turn[0]}: {turn[1]}" for turn in sample["dialogs"][0]])
        elif attr == "dialog_b" and "dialogs" in sample:
            sample_data[attr] = "\n".join([f"{turn[0]}: {turn[1]}" for turn in sample["dialogs"][1]])
        elif attr == "response_a" and "responses" in sample:
            sample_data[attr] = sample["responses"][0]
        elif attr == "response_b" and "responses" in sample:
            sample_data[attr] = sample["responses"][1]
        else:
            raise ValueError(f"Missing required attribute '{attr}' in sample {sample['id']}")
    
    return template.format(**sample_data), rating_scale

def save_rating(sample_id, metric, rating, it_background):
    # Map IT background to keys in ratings.json
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    
    if sample_id not in st.session_state.ratings[ratings_key]:
        st.session_state.ratings[ratings_key][sample_id] = {}
    if metric not in st.session_state.ratings[ratings_key][sample_id]:
        st.session_state.ratings[ratings_key][sample_id][metric] = {"votes": [], "vote_count": 0}
    st.session_state.ratings[ratings_key][sample_id][metric]["votes"].append(rating)
    st.session_state.ratings[ratings_key][sample_id][metric]["vote_count"] += 1


def start_new_round():
    chosen_metric = get_lowest_coverage_metric(validation_set, st.session_state.ratings, st.session_state.it_background)
    if chosen_metric is None:
        st.error("Error: Could not determine the next metric.")
        st.stop()
    if chosen_metric not in evaluation_templates:
        st.error(f"Error: The metric {chosen_metric} is not defined in the evaluation_templates")
        st.stop()
    st.session_state.current_metric = chosen_metric
    st.session_state.samples = get_samples_for_metric(st.session_state.current_metric, st.session_state.ratings, st.session_state.it_background)
    st.session_state.num_samples_this_round = min(5, len(st.session_state.samples))
    st.session_state.samples = st.session_state.samples[:5]
    st.session_state.sample_count = 0
    st.session_state.round_over = False
    st.session_state.round_count += 1
    if st.session_state.samples:
        st.session_state.current_sample = st.session_state.samples.pop(0)
    else:
        st.session_state.round_over = True

# Load the validation set
try:
    with open(VALIDATION_FILE, "r") as f:
        validation_set = json.load(f)
except FileNotFoundError:
    st.error(f"Error: {VALIDATION_FILE} not found. Please ensure the file exists.")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Error: Failed to decode json data from {VALIDATION_FILE}: {e}")
    st.stop()

# Initialize session state
if "ratings" not in st.session_state:
    st.session_state.ratings = load_ratings()
if "samples" not in st.session_state:
    st.session_state.samples = []
if "current_sample" not in st.session_state:
    st.session_state.current_sample = None
if "current_metric" not in st.session_state:
    st.session_state.current_metric = None
if "sample_count" not in st.session_state:
    st.session_state.sample_count = 0
if "round_over" not in st.session_state:
    st.session_state.round_over = False
if "user_rating" not in st.session_state:
    st.session_state.user_rating = None
if "num_samples_this_round" not in st.session_state:
    st.session_state.num_samples_this_round = 0
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "it_background" not in st.session_state:
    st.session_state.it_background = None
if "round_count" not in st.session_state:
    st.session_state.round_count = 0


def main():
    st.title("RAG Answer Rating App")

    if not st.session_state.app_started:
        st.write("Bitte geben Sie an, ob Sie über einen IT-Hintergrund verfügen.")
        st.session_state.it_background = st.radio("IT-Hintergrund:", ("Ja", "Nein"))

        if st.button("Start"):
            st.session_state.app_started = True
            st.rerun()
        return
    
    if st.session_state.round_over:
        st.success("Danke für Ihre Bewertungen!")
        if st.button("Next Round"):
            start_new_round()
            st.rerun()
        return

    if not st.session_state.current_sample:
        start_new_round()

    # --- Step Count Display (Framed and Centered) ---
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        ">
            <div style="
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                text-align: center;
                width: 100%;
            ">
                Sample {st.session_state.sample_count + 1}/{st.session_state.num_samples_this_round}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(general_intro_prompt)
    st.write(f"Bewertungsdimension: {st.session_state.current_metric}")
    prompt, rating_scale = generate_prompt(st.session_state.current_sample, st.session_state.current_metric)
    st.write(prompt)
    
    radio_key = f"user_rating_{st.session_state.round_count}_{st.session_state.sample_count}"
    user_rating = st.radio(
        "Bewertung:",
        rating_scale,
        key=radio_key,
        horizontal=True
    )

    if st.button("Next"):
        key = f"user_rating_{st.session_state.round_count}_{st.session_state.sample_count}"
        current_rating = st.session_state.get(key)
        
        if current_rating is None:
            st.warning("Bitte eine Bewertung auswählen, bevor Sie fortfahren.")
        else:
            save_rating(st.session_state.current_sample["id"], st.session_state.current_metric, current_rating, st.session_state.it_background)
            st.session_state.sample_count += 1

            if st.session_state.sample_count >= st.session_state.num_samples_this_round or not st.session_state.samples:
                st.session_state.round_over = True
                save_ratings(st.session_state.ratings)
            else:
                st.session_state.current_sample = st.session_state.samples.pop(0)
            
            st.rerun()

if __name__ == "__main__":
    main()

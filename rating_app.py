import streamlit as st
import json
import random
import os
import supabase
import html
from collections import defaultdict
# --- IMPORTANT: Update prompt_templates.py for pairwise metrics ---
# Ensure quality_pairwise requires ['query', 'answer_a', 'answer_b']
# Ensure multi_turn_quality_pairwise requires ['history_a', 'history_b'] (and maybe 'query')
from prompt_templates import evaluation_templates, general_intro_prompt

DATA_FILE = "ratings_data.json"
VALIDATION_FILE_A = "validationset.json"
# --- Added second validation file constant ---
VALIDATION_FILE_B = "validationset-b.json"
MODE = "local"  # supported modes: "local", "supabase"

# --- Define Pairwise Metrics ---
PAIRWISE_METRICS = {"quality_pairwise", "multiturn_quality_pairwise"}

# --- Helper Functions ---

def init_supabase():
    # ... (no changes needed)
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or Key not found in secrets")
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client


def load_ratings(supabase_client):
    ratings = {"Experten": {}, "Crowd": {}}

    if MODE == "supabase":
        try:
            response = supabase_client.schema("api").table("ratings").select("*").execute()
            data = response.data

            for row in data:
                rater_type = row["rater_type"]
                sample_id = row["sample_id"] # This ID represents the sample or sample pair
                metric = row["metric"]
                vote = row["vote"]
                swap_positions = row.get("swap_positions") # Use .get for safety

                if sample_id not in ratings[rater_type]:
                    ratings[rater_type][sample_id] = {}
                if metric not in ratings[rater_type][sample_id]:
                    # Initialize structure for both single and pairwise
                    ratings[rater_type][sample_id][metric] = {"votes": [], "swap_history": []}

                ratings[rater_type][sample_id][metric]["votes"].append(vote)
                # Only append swap_history if it's relevant (pairwise) and exists
                if metric in PAIRWISE_METRICS and swap_positions is not None:
                     ratings[rater_type][sample_id][metric]["swap_history"].append(swap_positions)
                elif metric not in PAIRWISE_METRICS:
                     # For non-pairwise, append None or a consistent value if needed later
                     ratings[rater_type][sample_id][metric]["swap_history"].append(None)


        except Exception as e:
            st.error(f"Error loading ratings from Supabase: {e}")
            # Fallback or default initialization
            ratings = {"Experten": {}, "Crowd": {}}

    else:  # local mode
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                try:
                    ratings = json.load(f)
                    # Ensure compatibility and recalculate vote counts
                    for group in ratings:
                        for sample_id in ratings[group]:
                            for metric in ratings[group][sample_id]:
                                # Ensure base structure
                                if "votes" not in ratings[group][sample_id][metric]:
                                    ratings[group][sample_id][metric]["votes"] = []
                                if "swap_history" not in ratings[group][sample_id][metric]:
                                     ratings[group][sample_id][metric]["swap_history"] = [None] * len(ratings[group][sample_id][metric]["votes"]) # Add placeholders if missing

                                # Ensure swap_history length matches votes length
                                if len(ratings[group][sample_id][metric]["swap_history"]) != len(ratings[group][sample_id][metric]["votes"]):
                                     # Attempt to fix - might need better logic depending on cause
                                     st.warning(f"Inconsistent votes/swap_history length for {sample_id}/{metric}. Resetting swap_history.")
                                     ratings[group][sample_id][metric]["swap_history"] = [None] * len(ratings[group][sample_id][metric]["votes"])

                                # Add vote_count (temporary for logic, removed on save)
                                ratings[group][sample_id][metric]["vote_count"] = len(
                                    ratings[group][sample_id][metric]["votes"]
                                )
                except json.JSONDecodeError:
                    st.error("Error loading ratings data. File may be corrupted.")
                    ratings = {"Experten": {}, "Crowd": {}}
                except Exception as e:
                    st.error(f"Error processing local ratings data: {e}")
                    ratings = {"Experten": {}, "Crowd": {}}
        else:
             ratings = {"Experten": {}, "Crowd": {}}


    # Ensure base structure exists
    if "Experten" not in ratings: ratings["Experten"] = {}
    if "Crowd" not in ratings: ratings["Crowd"] = {}

    return ratings


def save_ratings(ratings, supabase_client):
    ratings_copy = json.loads(json.dumps(ratings)) # Deep copy

    # Remove temporary "vote_count"
    for group in ratings_copy:
        for sample_id in ratings_copy[group]:
            for metric in ratings_copy[group][sample_id]:
                if "vote_count" in ratings_copy[group][sample_id][metric]:
                    del ratings_copy[group][sample_id][metric]["vote_count"]
                # Ensure swap_history length matches votes length before saving
                if len(ratings_copy[group][sample_id][metric].get("swap_history", [])) != len(ratings_copy[group][sample_id][metric].get("votes", [])):
                     st.error(f"CRITICAL SAVE ERROR: Mismatch votes/swap_history for {sample_id}/{metric}. Data might be corrupted.")
                     # Decide how to handle: skip saving this entry? Abort?
                     # For now, let's just log and continue, but this needs attention
                     continue # Skip this metric to avoid saving inconsistent data


    if MODE == "supabase":
        try:
            # --- Upsert logic is generally better for Supabase ---
            # Assumes a unique constraint on (rater_type, sample_id, metric, maybe_rater_id?)
            # If each vote is a unique row, insert is fine, but managing updates is harder.
            # Let's stick to INSERT for now, assuming each vote is logged.

            data_to_insert = []
            for rater_type, sample_data in ratings_copy.items():
                for sample_id, metric_data in sample_data.items():
                    for metric, vote_data in metric_data.items():
                        votes = vote_data.get("votes", [])
                        swaps = vote_data.get("swap_history", [None] * len(votes)) # Default swaps to None if missing

                        # Ensure lists have same length (should be guaranteed by load/save logic now)
                        if len(votes) != len(swaps):
                             st.error(f"Data inconsistency detected before Supabase save for {sample_id}/{metric}. Skipping.")
                             continue

                        for vote, swap_position in zip(votes, swaps):
                            data_to_insert.append({
                                "rater_type": rater_type,
                                "sample_id": sample_id, # Represents sample or pair ID
                                "metric": metric,
                                "vote": vote,
                                "swap_positions": swap_position if metric in PAIRWISE_METRICS else None,
                            })

            # Clear existing ratings before inserting new ones?
            # This depends on whether you want a full overwrite or append.
            # For simplicity, let's assume append (each vote is a new record).
            # If overwrite needed:
            # supabase_client.schema("api").table("ratings").delete().neq("id", -1).execute() # Careful!

            chunk_size = 100
            for i in range(0, len(data_to_insert), chunk_size):
                chunk = data_to_insert[i:i+chunk_size]
                supabase_client.schema("api").table("ratings").insert(chunk).execute()

        except Exception as e:
            st.error(f"Error saving ratings to Supabase: {e}")
    else: # local mode
        try:
            with open(DATA_FILE, "w") as f:
                json.dump(ratings_copy, f, indent=4)
        except IOError as e:
            st.error(f"Error saving ratings locally: {e}")


# --- Adapted get_lowest_coverage_metric ---
def get_lowest_coverage_metric(validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    # 1. Get all possible metrics (single, multi, pairwise)
    all_metrics = set()
    # Add single/multi-turn metrics from validation_data_a (structure assumed same as _b)
    for turn_type, criteria in validation_data_a.get("evaluation_criteria", {}).items():
        for category, metrics in criteria.items():
            all_metrics.update(metrics)
    # Add pairwise metrics explicitly
    all_metrics.update(PAIRWISE_METRICS)

    if not all_metrics:
        st.warning("No evaluation criteria found in validation set.")
        return None

    # 2. Map samples/pairs to their applicable metrics
    sample_metrics_map = defaultdict(list) # For single/multi-turn
    pair_metrics_map = defaultdict(list)   # For pairwise

    # Process validation_data_a for single/multi-turn metrics
    for turn_type in ["singleturn", "multiturn"]:
        if turn_type in validation_data_a:
            for category, samples in validation_data_a[turn_type].items():
                category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                for sample in samples:
                    sample_id = sample.get("id")
                    if sample_id:
                        # Add non-pairwise metrics
                        sample_metrics_map[sample_id].extend([m for m in category_metrics if m not in PAIRWISE_METRICS])

    # Identify valid pairs and map pairwise metrics
    ids_a = {s['id'] for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
    ids_b = {s['id'] for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
    common_ids = ids_a.intersection(ids_b)

    for sample_id in common_ids:
        # Determine turn type for the pair (assuming consistent across files)
        turn_type_a = None
        category_a = None
        for tt in ["singleturn", "multiturn"]:
             if tt in validation_data_a:
                 for cat, samples in validation_data_a[tt].items():
                     if any(s.get("id") == sample_id for s in samples):
                         turn_type_a = tt
                         category_a = cat
                         break
             if turn_type_a: break

        if turn_type_a and category_a:
             # Get applicable pairwise metrics for this category
             category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type_a, {}).get(category_a, [])
             applicable_pairwise = [m for m in category_metrics if m in PAIRWISE_METRICS]
             if applicable_pairwise:
                 pair_metrics_map[sample_id].extend(applicable_pairwise)


    # 3. Calculate vote counts per metric
    metric_vote_counts = defaultdict(list)

    # Ensure the ratings structure exists
    if ratings_key not in ratings: ratings[ratings_key] = {}

    # Count votes for single/multi-turn metrics
    for sample_id, metrics in sample_metrics_map.items():
        for metric in metrics:
            vote_count = len(ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", []))
            metric_vote_counts[metric].append(vote_count)

    # Count votes for pairwise metrics
    for sample_id, metrics in pair_metrics_map.items():
        for metric in metrics:
            # Vote count for a pair is stored under the shared sample_id and the pairwise metric
            vote_count = len(ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", []))
            metric_vote_counts[metric].append(vote_count)

    # 4. Calculate average coverage per metric
    metric_coverage = {}
    for metric in all_metrics:
        vote_counts = metric_vote_counts.get(metric, [])
        num_entities = 0 # Number of samples or pairs this metric applies to

        if metric in PAIRWISE_METRICS:
            num_entities = sum(1 for sample_id in pair_metrics_map if metric in pair_metrics_map[sample_id])
        else:
            num_entities = sum(1 for sample_id in sample_metrics_map if metric in sample_metrics_map[sample_id])

        if num_entities > 0:
            metric_coverage[metric] = sum(vote_counts) / num_entities if vote_counts else 0
        else:
            metric_coverage[metric] = 0 # Metric defined but no samples/pairs apply

    # 5. Find the metric with the lowest coverage
    if not metric_coverage:
        return random.choice(list(all_metrics)) if all_metrics else None

    min_coverage = min(metric_coverage.values())
    lowest_coverage_metrics = [m for m, cov in metric_coverage.items() if cov == min_coverage]

    return random.choice(lowest_coverage_metrics) if lowest_coverage_metrics else None


# --- Adapted get_samples_for_metric ---
def get_samples_for_metric(metric, validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    relevant_entities = [] # Can store single samples or pairs

    # Ensure the ratings structure exists
    if ratings_key not in ratings: ratings[ratings_key] = {}

    if metric in PAIRWISE_METRICS:
        # Find pairs with the same ID in both files
        samples_a_dict = {s['id']: s for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
        samples_b_dict = {s['id']: s for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
        common_ids = set(samples_a_dict.keys()).intersection(samples_b_dict.keys())

        for sample_id in common_ids:
            sample_a = samples_a_dict[sample_id]
            sample_b = samples_b_dict[sample_id]

            # Check if the metric applies to this pair's category/turn_type
            turn_type_a = None
            category_a = None
            for tt in ["singleturn", "multiturn"]:
                 if tt in validation_data_a:
                     for cat, samples in validation_data_a[tt].items():
                         if any(s.get("id") == sample_id for s in samples):
                             turn_type_a = tt
                             category_a = cat
                             break
                 if turn_type_a: break

            if turn_type_a and category_a:
                 category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type_a, {}).get(category_a, [])
                 if metric in category_metrics:
                     # Calculate vote count for this pair and metric
                     vote_count = len(ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", []))
                     relevant_entities.append(((sample_a, sample_b), vote_count)) # Store pair tuple

    else: # Single/Multi-turn metric
        for turn_type in ["singleturn", "multiturn"]:
            if turn_type in validation_data_a:
                for category, samples in validation_data_a[turn_type].items():
                    category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                    if metric in category_metrics:
                        for sample in samples:
                            sample_id = sample.get("id")
                            if sample_id:
                                vote_count = len(ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", []))
                                relevant_entities.append((sample, vote_count)) # Store single sample

    # Sort entities (samples or pairs) by vote count
    relevant_entities.sort(key=lambda x: x[1])

    # Return only the sample objects or pair tuples
    return [entity for entity, _ in relevant_entities]


# --- Helper function for formatting history ---
def format_chat_history(history_list):
    """Formats a list of chat turns into a readable HTML string."""
    formatted_lines = []
    for turn in history_list:
        role = turn.get('role', 'unknown').lower()
        # Basic HTML escaping for content to prevent accidental tag injection
        # You might want a more robust library like `bleach` if content can be complex
        import html
        content = html.escape(turn.get('content', '[missing content]'))

        prefix = "Unbekannt" # Default prefix
        if role == 'user':
            prefix = "Nutzer"
        elif role == 'assistant' or role == 'bot': # Handle common variations
            prefix = "Bot"
        elif role == 'system':
             prefix = "System" # Or decide to skip system messages

        # Only add lines with known roles or if you want to show unknown ones
        if prefix != "Unbekannt" or role == 'unknown': # Adjust this condition if needed
            # Use bold for the prefix for better visual separation
            formatted_lines.append(f"<b>{prefix}:</b> {content}")

    # Join lines with the HTML break tag
    return "<br>".join(formatted_lines)

# --- Adapted generate_prompt ---
def generate_prompt(entity, metric, swap_options=False):
    if metric not in evaluation_templates:
        raise ValueError(f"Metric '{metric}' not found in evaluation_templates.")

    template_config = evaluation_templates[metric]
    template = template_config["prompt"]
    required_attributes = template_config["required_attributes"]
    rating_scale = template_config["rating_scale"]

    sample_data = {} # For formatting the base instruction template
    content_a = "[Antwort A nicht gefunden]" # Default content for option A answer
    content_b = "[Antwort B nicht gefunden]" # Default content for option B answer
    missing_attrs = []
    formatted_shared_history = None # To store formatted shared history if applicable

    is_pairwise = isinstance(entity, tuple) and len(entity) == 2

    if is_pairwise:
        # --- PAIRWISE LOGIC ---
        sample_a_orig, sample_b_orig = entity
        sample_id = sample_a_orig.get("id", "N/A") # Assume IDs match

        # Determine which sample provides content for A and B based on swap_options
        sample_for_a = sample_b_orig if swap_options else sample_a_orig
        sample_for_b = sample_a_orig if swap_options else sample_b_orig

        # 1. Handle Shared History (if required by the metric)
        if "history" in required_attributes:
            # Fetch history from one sample (assuming they are identical)
            history_list = sample_a_orig.get("history", [])
            if history_list:
                formatted_shared_history = format_chat_history(history_list)
                # Add to sample_data if the template uses {shared_history} placeholder
                sample_data["history"] = formatted_shared_history
            else:
                missing_attrs.append("history")
                sample_data["history"] = "[Chat-Verlauf nicht gefunden]"

        # 2. Populate other base template data (e.g., query, if needed)
        #    Exclude 'history' and attributes ending in _a/_b
        base_template_attrs = [
            attr for attr in required_attributes
            if not (attr.endswith('_a') or attr.endswith('_b') or attr == 'history')
        ]
        for attr in base_template_attrs:
            value = sample_a_orig.get(attr, sample_b_orig.get(attr))
            if value is not None:
                sample_data[attr] = value
            else:
                missing_attrs.append(attr)
                sample_data[attr] = f"[Attribute '{attr}' nicht gefunden]"

        # 3. Extract the differing content (e.g., answer_a, answer_b) for columns
        attr_a = next((attr for attr in required_attributes if attr.endswith('_a')), None)
        attr_b = next((attr for attr in required_attributes if attr.endswith('_b')), None)

        if not attr_a or not attr_b:
             st.error(f"Pairwise metric '{metric}' misconfiguration: Missing required attributes ending in '_a' and '_b' in evaluation_templates.")
             formatted_prompt = "[Fehler: Fehlende _a/_b Attribute in Konfiguration]"
             return formatted_prompt, rating_scale # Return early

        base_attr_a = attr_a[:-2] # e.g., "answer" from "answer_a"
        base_attr_b = attr_b[:-2] # e.g., "answer" from "answer_b"

        # Get content for column A (using sample_for_a)
        if base_attr_a in sample_for_a:
            raw_content_a = str(sample_for_a[base_attr_a])
            content_a = html.escape(raw_content_a) # Escape the differing content
        else:
            missing_attrs.append(attr_a)
            content_a = f"[Attribute '{base_attr_a}' nicht in Sample A gefunden ({'getauscht' if swap_options else 'original'})]"

        # Get content for column B (using sample_for_b)
        if base_attr_b in sample_for_b:
            raw_content_b = str(sample_for_b[base_attr_b])
            content_b = html.escape(raw_content_b) # Escape the differing content
        else:
            missing_attrs.append(attr_b)
            content_b = f"[Attribute '{base_attr_b}' nicht in Sample B gefunden ({'original' if swap_options else 'getauscht'})]"

        # 4. Construct Final Display (Instructions + Columns)
        try:
            # Format the base instruction part (which might now include shared_history)
            formatted_instruction_prompt = template.format(**sample_data)
        except KeyError as e:
             st.error(f"Error formatting base prompt for metric '{metric}', sample/pair {sample_id}: Missing key {e}. Check template placeholders.")
             formatted_instruction_prompt = f"[Error formatting instructions: Missing key {e}]"
        except Exception as e:
             st.error(f"An unexpected error occurred during base prompt formatting for metric '{metric}', sample/pair {sample_id}: {e}")
             formatted_instruction_prompt = "[Error formatting instructions]"

        # --- HTML Structure: Instructions first, then columns for differing answers ---
        # (CSS can remain the same as before)
        html_columns = f"""
<style>
.column-container {{
    display: flex;
    flex-wrap: wrap; /* Allow wrapping on smaller screens */
    gap: 24px;       /* Space between columns */
    margin-top: 20px;
    justify-content: center; /* Center columns if they wrap */
}}

.column {{
    /* --- MODIFICATION START --- */
    flex: 1 1 300px; /* Allow columns to grow and shrink, base width 300px */
    /* Removed max-width: 45%; to let them fill more space */
    min-width: 280px; /* Ensure minimum width */
    /* --- MODIFICATION END --- */
    border: 1px solid #dcdcdc;
    padding: 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    transition: box-shadow 0.3s ease;
    word-wrap: break-word;
    overflow-wrap: break-word;
}}

.column:hover {{
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}}

.column h3 {{
    margin-top: 0;
    margin-bottom: 10px;
    font-size: 1.2rem;
    border-bottom: 1px solid #eee;
    padding-bottom: 8px;
    color: #333;
}}

.column-content {{
    font-size: 1rem;
    line-height: 1.5;
    color: #444;
}}

.column-content b {{ /* For history prefix */
    color: #111;
}}
</style>

{formatted_instruction_prompt}

<div class="column-container">
    <div class="column">
        <h3>Antwort A</h3>
        <div class="column-content">{content_a}</div>
    </div>
    <div class="column">
        <h3>Antwort B</h3>
        <div class="column-content">{content_b}</div>
    </div>
</div>
"""
        formatted_prompt = html_columns
        # --- End HTML construction ---

    else:
        # --- SINGLE SAMPLE LOGIC (Remains the same) ---
        sample = entity
        sample_id = sample.get("id", "N/A")
        for attr in required_attributes:
            if attr == "history" and "history" in sample:
                 sample_data[attr] = format_chat_history(sample.get("history", []))
            elif attr in sample:
                 sample_data[attr] = sample[attr]
            else:
                missing_attrs.append(attr)
                sample_data[attr] = f"[Attribute '{attr}' not found in sample]"

        try:
            formatted_prompt = template.format(**sample_data)
        except KeyError as e:
            st.error(f"Error formatting prompt for metric '{metric}', sample {sample_id}: Missing key {e}. Check template placeholders and 'required_attributes'.")
            formatted_prompt = "[Error formatting prompt]"
        except Exception as e:
            st.error(f"An unexpected error occurred during prompt formatting for metric '{metric}', sample {sample_id}: {e}")
            formatted_prompt = "[Error formatting prompt]"


    if missing_attrs:
        st.warning(f"Missing required attributes {missing_attrs} for metric '{metric}' in sample/pair {sample_id}. Check 'required_attributes' in evaluation_templates.")

    return formatted_prompt, rating_scale


# --- Adapted save_rating ---
def save_rating(sample_id, metric, rating, it_background, is_pairwise, swap_options, supabase_client=None):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    # Ensure nested dictionaries exist
    if ratings_key not in st.session_state.ratings: st.session_state.ratings[ratings_key] = {}
    if sample_id not in st.session_state.ratings[ratings_key]: st.session_state.ratings[ratings_key][sample_id] = {}
    if metric not in st.session_state.ratings[ratings_key][sample_id]:
        st.session_state.ratings[ratings_key][sample_id][metric] = {"votes": [], "swap_history": []}
    elif "votes" not in st.session_state.ratings[ratings_key][sample_id][metric]:
         st.session_state.ratings[ratings_key][sample_id][metric]["votes"] = []
    elif "swap_history" not in st.session_state.ratings[ratings_key][sample_id][metric]:
         st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"] = []


    # Determine swap value to save
    swap_value_to_save = swap_options if is_pairwise else None

    # Append the new rating and swap status
    st.session_state.ratings[ratings_key][sample_id][metric]["votes"].append(rating)
    st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"].append(swap_value_to_save)

    # --- Save to Supabase or Local File ---
    if MODE == "supabase" and supabase_client:
        try:
            insert_data = {
                "rater_type": ratings_key,
                "sample_id": sample_id, # Represents sample or pair ID
                "metric": metric,
                "vote": rating,
                "swap_positions": swap_value_to_save, # Save swap status
            }
            response = supabase_client.schema("api").table("ratings").insert(insert_data).execute()
            # Optional error checking on response
        except Exception as e:
            st.error(f"Error saving rating to Supabase: {e}")
    elif MODE == "local":
        save_ratings(st.session_state.ratings, None)


# --- Adapted start_new_round ---
def start_new_round():
    # Ensure validation_sets are loaded
    if 'validation_sets' not in globals():
         st.error("Validation sets not loaded.")
         st.stop()

    validation_data_a = validation_sets['a']
    validation_data_b = validation_sets['b']

    chosen_metric = get_lowest_coverage_metric(
        validation_data_a, validation_data_b, st.session_state.ratings, st.session_state.it_background
    )
    if chosen_metric is None:
        st.error("Error: Could not determine the next metric.")
        st.stop()

    if chosen_metric not in evaluation_templates:
        st.error(f"Error: Metric '{chosen_metric}' not defined in evaluation_templates.")
        st.stop()

    st.session_state.current_metric = chosen_metric
    st.session_state.is_pairwise = chosen_metric in PAIRWISE_METRICS

    # Get samples or pairs for the chosen metric
    st.session_state.entities = get_samples_for_metric(
        st.session_state.current_metric, validation_data_a, validation_data_b, st.session_state.ratings, st.session_state.it_background
    )

    if not st.session_state.entities:
         st.warning(f"No samples/pairs found for the chosen metric '{chosen_metric}'. Trying next round.")
         st.session_state.round_over = True
         return

    # Determine number of entities (samples/pairs) for this round
    st.session_state.num_entities_this_round = min(5, len(st.session_state.entities))
    st.session_state.entities_this_round = st.session_state.entities[:st.session_state.num_entities_this_round]

    st.session_state.entity_count = 0 # Renamed from sample_count
    st.session_state.round_over = False
    st.session_state.round_count += 1
    st.session_state.current_sample = None # Reset single sample view
    st.session_state.current_sample_pair = None # Reset pair view

    if st.session_state.entities_this_round:
        current_entity = st.session_state.entities_this_round.pop(0)
        if st.session_state.is_pairwise:
            st.session_state.current_sample_pair = current_entity
            st.session_state.swap_options = random.choice([True, False]) # Randomize swap for pairs
        else:
            st.session_state.current_sample = current_entity
            st.session_state.swap_options = False # No swap for single samples
    else:
        st.warning("Started round but no entities available.")
        st.session_state.round_over = True


# --- Load BOTH validation sets ---
validation_sets = {}
try:
    with open(VALIDATION_FILE_A, "r", encoding="utf-8") as f:
        validation_sets['a'] = json.load(f)
    with open(VALIDATION_FILE_B, "r", encoding="utf-8") as f:
        validation_sets['b'] = json.load(f)

    # --- VERSION CHECK START ---
    version_a = validation_sets['a'].get("version")
    version_b = validation_sets['b'].get("version")

    if version_a is None or version_b is None:
        st.error(
            f"Fehler: 'version'-Schlüssel fehlt in einer oder beiden Validierungsdateien. "
            f"({VALIDATION_FILE_A}: {'vorhanden' if version_a is not None else 'fehlt'}, "
            f"{VALIDATION_FILE_B}: {'vorhanden' if version_b is not None else 'fehlt'}). "
            f"Bitte fügen Sie einen 'version'-Schlüssel hinzu."
        )
        st.stop()

    if version_a != version_b:
        st.error(
            f"Fehler: Versionskonflikt zwischen Validierungsdateien! "
            f"'{VALIDATION_FILE_A}' hat Version '{version_a}', "
            f"aber '{VALIDATION_FILE_B}' hat Version '{version_b}'. "
            f"Bitte stellen Sie sicher, dass beide Dateien die gleiche Version haben."
        )
        st.stop()
    # --- VERSION CHECK END ---

except FileNotFoundError as e:
    st.error(f"Error: Validation file not found: {e.filename}. Please ensure both {VALIDATION_FILE_A} and {VALIDATION_FILE_B} exist.")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Error: Failed to decode JSON data from validation file. Check syntax: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading validation files: {e}")
    st.stop()


# --- Initialize session state ---
st.session_state.setdefault("ratings", {})
st.session_state.setdefault("entities", []) # Stores samples OR pairs for the current metric
st.session_state.setdefault("entities_this_round", []) # Stores entities for the current round batch
st.session_state.setdefault("current_sample", None) # For single sample display
st.session_state.setdefault("current_sample_pair", None) # For pairwise display
st.session_state.setdefault("current_metric", None)
st.session_state.setdefault("entity_count", 0) # Tracks progress within the round batch
st.session_state.setdefault("round_over", True)
st.session_state.setdefault("user_rating", None)
st.session_state.setdefault("num_entities_this_round", 0) # Total entities in the current batch
st.session_state.setdefault("app_started", False)
st.session_state.setdefault("it_background", None)
st.session_state.setdefault("round_count", 0)
st.session_state.setdefault("swap_options", False) # Controls A/B swap for pairwise
st.session_state.setdefault("is_pairwise", False) # Tracks if current metric is pairwise

# Initialize Supabase client or load local ratings
supabase_client = None
if MODE == "supabase":
    try:
        supabase_client = init_supabase()
        st.session_state.ratings = load_ratings(supabase_client)
    except ValueError as e:
        st.error(f"Supabase initialization failed: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Supabase setup: {e}")
        st.stop()
elif MODE == "local":
    st.session_state.ratings = load_ratings(None)
else:
    st.error(f"Invalid MODE '{MODE}'. Use 'local' or 'supabase'.")
    st.stop()


def main():
    st.title("RAG Answer Rating App")

    # --- Initial User Setup ---
    if not st.session_state.app_started:
        st.write("Willkommen! Danke für die Teilnahme an dieser Bewertung.")
        st.write("Bitte geben Sie an, ob Sie über einen IT-Hintergrund verfügen.")
        it_background_choice = st.radio(
            "IT-Hintergrund:",
            ("Ja", "Nein"),
            key="it_background_radio",
            horizontal=True,
            index=None
        )
        if st.button("Start", key="start_button"):
            if it_background_choice is None:
                st.warning("Bitte wählen Sie eine Option für den IT-Hintergrund.")
            else:
                st.session_state.it_background = it_background_choice
                st.session_state.app_started = True
                # --- MODIFICATION START ---
                # Directly start the first round here instead of just setting round_over=True
                start_new_round()
                # Rerun to reflect the state changes from start_new_round()
                st.rerun()
                # --- MODIFICATION END ---
        return # Important: Still return here to prevent rest of main() executing on this initial run

    # --- Round Management ---
    # This block will now be skipped on the rerun immediately after clicking "Start"
    # because start_new_round() sets round_over to False (if samples are found)
    if st.session_state.round_over:
        if st.session_state.round_count > 0:
             st.success(f"Runde {st.session_state.round_count} abgeschlossen. Danke für Ihre Bewertungen!")
        if st.button("Nächste Runde starten", key="next_round_button"):
            start_new_round()
            if not st.session_state.round_over:
                 st.rerun()
        return

    # ... (rest of the main function remains the same) ...
    # Check for inconsistent state
    current_entity_for_display = st.session_state.current_sample_pair if st.session_state.is_pairwise else st.session_state.current_sample
    if not current_entity_for_display and not st.session_state.round_over:
         st.warning("Zustandsfehler: Kein aktuelles Sample/Paar. Starte neue Runde.")
         # Attempt to recover by starting a new round
         start_new_round()
         if not st.session_state.round_over:
             st.rerun()
         else:
             # If starting a new round still results in round_over, stop here
             st.info("Keine weiteren Aufgaben verfügbar.") # Or a more appropriate message
             return # Stop execution for this run


    # --- Display Current Entity (Sample or Pair) ---
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; text-align: center; width: 100%;">
                Item {st.session_state.entity_count + 1} / {st.session_state.num_entities_this_round} (Runde {st.session_state.round_count})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(general_intro_prompt)
    st.write(f"**Bewertungsdimension:** {st.session_state.current_metric}")

    try:
        # Pass the correct entity (sample or pair) and swap_options
        prompt, rating_scale = generate_prompt(
            current_entity_for_display,
            st.session_state.current_metric,
            st.session_state.swap_options # Pass swap state
        )
        st.write("---")
        st.markdown(prompt, unsafe_allow_html=True)
        st.write("---")

    except ValueError as e:
        st.error(f"Fehler beim Generieren des Prompts: {e}")
        st.button("Problem melden und nächste Runde starten", on_click=lambda: setattr(st.session_state, 'round_over', True))
        return
    except Exception as e:
         st.error(f"Unerwarteter Fehler bei der Prompt-Generierung: {e}")
         st.button("Problem melden und nächste Runde starten", on_click=lambda: setattr(st.session_state, 'round_over', True))
         return

    # --- Rating Input ---
    # Determine sample ID for the key (consistent for pairs)
    current_id = current_entity_for_display[0]['id'] if st.session_state.is_pairwise else current_entity_for_display['id']
    radio_key = f"user_rating_{st.session_state.round_count}_{current_id}_{st.session_state.entity_count}"
    st.session_state.user_rating = st.radio(
        "Ihre Bewertung:",
        rating_scale,
        key=radio_key,
        horizontal=True,
        index=None
    )

    # --- Next Button Logic ---
    if st.button("Weiter", key="next_entity_button"):
        current_rating = st.session_state.user_rating

        if current_rating is None:
            st.warning("Bitte wählen Sie eine Bewertung aus, bevor Sie fortfahren.")
        else:
            try:
                # Get the ID (consistent for single or pair)
                sample_id_to_save = current_id

                save_rating(
                    sample_id_to_save,
                    st.session_state.current_metric,
                    current_rating,
                    st.session_state.it_background,
                    st.session_state.is_pairwise, # Pass pairwise status
                    st.session_state.swap_options, # Pass swap status
                    supabase_client
                )
                st.session_state.entity_count += 1

                # Check if the round batch is finished
                if st.session_state.entity_count >= st.session_state.num_entities_this_round or not st.session_state.entities_this_round:
                    st.session_state.round_over = True
                    st.session_state.current_sample = None
                    st.session_state.current_sample_pair = None
                    st.session_state.entities_this_round = []
                else:
                    # Get the next entity from the current round's batch
                    next_entity = st.session_state.entities_this_round.pop(0)
                    if st.session_state.is_pairwise:
                        st.session_state.current_sample_pair = next_entity
                        st.session_state.current_sample = None
                        st.session_state.swap_options = random.choice([True, False]) # Randomize swap for next pair
                    else:
                        st.session_state.current_sample = next_entity
                        st.session_state.current_sample_pair = None
                        st.session_state.swap_options = False # Reset swap for single sample

                st.session_state.user_rating = None
                st.rerun()

            except KeyError as e:
                 st.error(f"Fehler beim Speichern: Fehlender Schlüssel {e}. Entity: {current_entity_for_display}")
                 st.session_state.round_over = True
                 st.rerun()
            except Exception as e:
                 st.error(f"Fehler beim Speichern der Bewertung: {e}")
                 st.session_state.round_over = True
                 st.rerun()


if __name__ == "__main__":
    main()

# /home/dns/repos/streamlit-rating-app/rating_app.py
import streamlit as st
import json
import random
import os
import supabase
import html
import markdown
from collections import defaultdict, Counter # Added Counter
from prompt_templates import evaluation_templates, general_intro_prompt

# ... (Keep other constants: DATA_FILE, VALIDATION_FILE_A/B, MODE, TARGET_VOTES, SAMPLES_PER_ROUND, PAIRWISE_METRICS) ...
DATA_FILE = "data/ratings_data.json"
VALIDATION_FILE_A = "data/validationset-test.json"
VALIDATION_FILE_B = "data/validationset-test-b.json"
MODE = "supabase"
TARGET_VOTES = 3
SAMPLES_PER_ROUND = 1
PAIRWISE_METRICS = {"quality_pairwise", "multiturn_quality_pairwise"}

# --- NEW Helper Function: Get Effective Vote Count ---
def get_effective_vote_count(sample_id, metric, ratings_data_for_sample_metric, validation_data_a):
    """
    Calculates the effective number of votes for a sample/metric,
    handling the special aggregation for context relevance metrics in local mode.
    """
    is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
    votes_list = ratings_data_for_sample_metric.get("votes", [])

    if not is_context_relevance_metric or MODE != "local":
        # Standard count for non-context metrics or Supabase mode (needs server-side aggregation)
        return len(votes_list)
    else:
        # --- Local Mode: Context Relevance Aggregation ---
        # Find the sample to know how many contexts are expected
        num_expected_contexts = 0
        found_sample = None
        for turn_type in ["singleturn", "multiturn"]:
            if turn_type in validation_data_a:
                for category, samples in validation_data_a[turn_type].items():
                    for sample in samples:
                        if sample.get("id") == sample_id:
                            # Check if the metric applies to this sample's category
                            category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                            if metric in category_metrics:
                                retrieved_contexts = sample.get('retrieved_contexts', [])
                                # Ensure retrieved_contexts is a list before getting len
                                if isinstance(retrieved_contexts, list):
                                    num_expected_contexts = len(retrieved_contexts)
                                else:
                                    num_expected_contexts = 0 # Treat as 0 if not a list
                                found_sample = True
                                break
                    if found_sample: break
            if found_sample: break

        if num_expected_contexts == 0:
            return 0 # No contexts to rate, so 0 effective votes

        # Count ratings per context index
        context_index_counts = Counter()
        for vote in votes_list:
            if isinstance(vote, (list, tuple)) and len(vote) == 2:
                # Expected format: (rating, context_index)
                rating_val, index = vote
                if isinstance(index, int) and 0 <= index < num_expected_contexts:
                    context_index_counts[index] += 1
            # else: ignore malformed vote entries for counting purposes

        # Effective votes = minimum count across all expected indices
        if len(context_index_counts) < num_expected_contexts:
            # If not all context indices have received at least one vote
            return 0
        else:
            # All indices have >= 1 vote, find the minimum count
            min_votes = min(context_index_counts[i] for i in range(num_expected_contexts))
            return min_votes

def init_supabase():
    # ... (no changes needed)
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or Key not found in secrets")
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client

# --- Load/Save Functions (Ensure they handle the potential tuple format for context votes locally) ---
# Modify load_ratings slightly to ensure structure even if file is empty/new
def load_ratings(supabase_client):
    ratings = {"Experten": {}, "Crowd": {}}

    if MODE == "supabase":
        # ... (Keep existing Supabase load logic) ...
        # NOTE: This does NOT aggregate context relevance votes correctly yet.
        try:
            response = supabase_client.schema("api").table("ratings").select("*").execute()
            data = response.data

            for row in data:
                rater_type = row["rater_type"]
                sample_id = row["sample_id"] # This ID represents the sample or sample pair
                metric = row["metric"]
                vote = row["vote"]
                swap_positions = row.get("swap_positions")
                # context_index = row.get("context_index") # Fetch if needed

                if sample_id not in ratings[rater_type]:
                    ratings[rater_type][sample_id] = {}
                if metric not in ratings[rater_type][sample_id]:
                    ratings[rater_type][sample_id][metric] = {"votes": [], "swap_history": []}

                # Store raw vote; aggregation happens later if needed by logic accessing this
                ratings[rater_type][sample_id][metric]["votes"].append(vote)

                if "swap_history" not in ratings[rater_type][sample_id][metric]:
                     ratings[rater_type][sample_id][metric]["swap_history"] = [] # Initialize if missing

                # Append swap history, ensuring length matches votes
                current_votes_len = len(ratings[rater_type][sample_id][metric]["votes"])
                swap_val = swap_positions if metric in PAIRWISE_METRICS and swap_positions is not None else None
                # Pad swap_history if needed
                while len(ratings[rater_type][sample_id][metric]["swap_history"]) < current_votes_len -1:
                     ratings[rater_type][sample_id][metric]["swap_history"].append(None)
                if len(ratings[rater_type][sample_id][metric]["swap_history"]) < current_votes_len:
                     ratings[rater_type][sample_id][metric]["swap_history"].append(swap_val)


        except Exception as e:
            st.error(f"Error loading ratings from Supabase: {e}")
            ratings = {"Experten": {}, "Crowd": {}} # Fallback

    else:  # local mode
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f: # Ensure correct encoding
                    ratings = json.load(f)
                    # Basic validation and structure check
                    if not isinstance(ratings, dict): ratings = {}
                    if "Experten" not in ratings: ratings["Experten"] = {}
                    if "Crowd" not in ratings: ratings["Crowd"] = {}

                    for group in ratings:
                        if not isinstance(ratings[group], dict): ratings[group] = {}
                        for sample_id in list(ratings[group].keys()): # Iterate over keys copy
                            if not isinstance(ratings[group][sample_id], dict):
                                ratings[group][sample_id] = {}
                                continue
                            for metric in list(ratings[group][sample_id].keys()): # Iterate over keys copy
                                if not isinstance(ratings[group][sample_id][metric], dict):
                                    ratings[group][sample_id][metric] = {"votes": [], "swap_history": []}
                                    continue

                                # Ensure 'votes' and 'swap_history' lists exist
                                if "votes" not in ratings[group][sample_id][metric] or not isinstance(ratings[group][sample_id][metric]["votes"], list):
                                    ratings[group][sample_id][metric]["votes"] = []
                                if "swap_history" not in ratings[group][sample_id][metric] or not isinstance(ratings[group][sample_id][metric]["swap_history"], list):
                                    ratings[group][sample_id][metric]["swap_history"] = []

                                # Ensure swap_history length matches votes length (simple padding)
                                votes_len = len(ratings[group][sample_id][metric]["votes"])
                                swap_len = len(ratings[group][sample_id][metric]["swap_history"])
                                if swap_len < votes_len:
                                    ratings[group][sample_id][metric]["swap_history"].extend([None] * (votes_len - swap_len))
                                elif swap_len > votes_len:
                                     ratings[group][sample_id][metric]["swap_history"] = ratings[group][sample_id][metric]["swap_history"][:votes_len] # Truncate extra swaps

                                # DO NOT add vote_count here, calculate dynamically when needed

            except json.JSONDecodeError:
                st.error(f"Error loading ratings data from {DATA_FILE}. File might be corrupted. Starting fresh.")
                ratings = {"Experten": {}, "Crowd": {}}
            except Exception as e:
                st.error(f"Error processing local ratings data: {e}")
                ratings = {"Experten": {}, "Crowd": {}}
        else:
             ratings = {"Experten": {}, "Crowd": {}} # File doesn't exist

    # Final check for top-level keys
    if "Experten" not in ratings: ratings["Experten"] = {}
    if "Crowd" not in ratings: ratings["Crowd"] = {}

    return ratings


def save_ratings(ratings, supabase_client):
    # No need to remove vote_count as it's not stored persistently
    ratings_copy = json.loads(json.dumps(ratings)) # Deep copy for safety

    # Perform consistency check before saving (Keep this part)
    for group in ratings_copy:
        for sample_id in ratings_copy[group]:
            for metric in ratings_copy[group][sample_id]:
                votes = ratings_copy[group][sample_id][metric].get("votes", [])
                swaps = ratings_copy[group][sample_id][metric].get("swap_history", [])
                if len(votes) != len(swaps):
                     st.error(f"CRITICAL SAVE ERROR: Mismatch votes ({len(votes)}) / swap_history ({len(swaps)}) for {sample_id}/{metric}. Attempting to fix by padding swap_history.")
                     # Attempt fix: Pad swap_history
                     swap_len = len(swaps)
                     votes_len = len(votes)
                     if swap_len < votes_len:
                         ratings_copy[group][sample_id][metric]["swap_history"].extend([None] * (votes_len - swap_len))
                     else: # Should not happen if load_ratings fixed it, but as fallback
                         ratings_copy[group][sample_id][metric]["swap_history"] = swaps[:votes_len]
                     # Re-check length after fix attempt
                     if len(ratings_copy[group][sample_id][metric]["votes"]) != len(ratings_copy[group][sample_id][metric]["swap_history"]):
                          st.error(f"--> FIX FAILED for {sample_id}/{metric}. Skipping save for this metric to avoid corruption.")
                          continue # Skip saving this potentially corrupt metric entry


    if MODE == "supabase":
        # --- REMOVE THE BULK INSERT LOGIC ---
        # The actual saving to Supabase happens individually in the save_rating function.
        # This function (save_ratings) is called in local mode to overwrite the file,
        # but in Supabase mode, the individual inserts in save_rating are sufficient.
        # You might log a message here if needed, but no database operation is required.
        # st.info("Supabase mode: Individual ratings saved via save_rating.")
        pass # Nothing to do here for saving

    else: # local mode (Keep this part as is)
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f: # Ensure correct encoding
                json.dump(ratings_copy, f, indent=4, ensure_ascii=False) # Use ensure_ascii=False for non-latin chars
        except IOError as e:
            st.error(f"Error saving ratings locally to {DATA_FILE}: {e}")
        except Exception as e:
             st.error(f"Unexpected error saving ratings locally: {e}")



# --- MODIFIED get_lowest_coverage_metric ---
def get_lowest_coverage_metric(validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    st.write("--- Debug: Calculating Lowest Coverage Metric ---") # DEBUG

    # 1. Get all possible metrics
    all_metrics = set(evaluation_templates.keys()) # Get all defined metrics
    st.write(f"All defined metrics: {all_metrics}") # DEBUG

    if not all_metrics:
        st.warning("No evaluation templates found.")
        return None

    # 2. Map samples/pairs to their applicable metrics
    sample_metrics_map = defaultdict(list)
    pair_metrics_map = defaultdict(list)

    # Process validation_data_a for single/multi-turn metrics
    # ... (Keep the existing logic for populating sample_metrics_map) ...
    for turn_type in ["singleturn", "multiturn"]:
        if turn_type in validation_data_a:
            for category, samples in validation_data_a[turn_type].items():
                category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                for sample in samples:
                    sample_id = sample.get("id")
                    if sample_id:
                        retrieved_contexts = sample.get('retrieved_contexts')
                        has_empty_context = not retrieved_contexts
                        history = sample.get('history')
                        has_history = isinstance(history, list) and len(history) > 0

                        for metric in category_metrics:
                            if metric in PAIRWISE_METRICS: continue # Skip pairwise here

                            metric_config = evaluation_templates.get(metric, {})
                            metric_req_attrs = metric_config.get('required_attributes', [])
                            is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
                            metric_requires_history = 'history' in metric_req_attrs

                            # Skip checks
                            if is_context_relevance_metric and has_empty_context: continue
                            if metric_requires_history and not has_history: continue

                            sample_metrics_map[sample_id].append(metric)

    # Identify valid pairs and map pairwise metrics
    # ... (Keep the existing logic for populating pair_metrics_map) ...
    ids_a = {s['id'] for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
    ids_b = {s['id'] for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
    common_ids = ids_a.intersection(ids_b)
    st.write(f"Common IDs for pairwise: {len(common_ids)}") # DEBUG

    samples_a_dict = {s['id']: s for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}

    for sample_id in common_ids:
        sample_a = samples_a_dict.get(sample_id)
        if not sample_a: continue

        turn_type_a, category_a = None, None
        # Find category for sample_a
        for tt in ["singleturn", "multiturn"]:
             if tt in validation_data_a:
                 for cat, samples in validation_data_a[tt].items():
                     if any(s.get("id") == sample_id for s in samples):
                         turn_type_a, category_a = tt, cat
                         break
             if turn_type_a: break

        if turn_type_a and category_a:
             category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type_a, {}).get(category_a, [])
             applicable_pairwise = [m for m in category_metrics if m in PAIRWISE_METRICS]

             if applicable_pairwise:
                 history_a = sample_a.get('history')
                 pair_has_history = isinstance(history_a, list) and len(history_a) > 0

                 for metric in applicable_pairwise:
                     metric_config = evaluation_templates.get(metric, {})
                     metric_req_attrs = metric_config.get('required_attributes', [])
                     metric_requires_history = 'history' in metric_req_attrs

                     if metric_requires_history and not pair_has_history: continue
                     # Add context skip logic if needed

                     pair_metrics_map[sample_id].append(metric)
                     st.write(f"Pairwise metric {metric} applicable to sample ID {sample_id}") # DEBUG

    # 3. Calculate effective vote counts per metric using the helper function
    metric_effective_vote_counts = defaultdict(list) # Stores list of *effective* vote counts
    if ratings_key not in ratings: ratings[ratings_key] = {}
    user_ratings = ratings.get(ratings_key, {})

    # Calculate for single/multi-turn samples
    for sample_id, metrics in sample_metrics_map.items():
        for metric in metrics:
            ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
            effective_count = get_effective_vote_count(sample_id, metric, ratings_data, validation_data_a)
            metric_effective_vote_counts[metric].append(effective_count)

    # Calculate for pairwise samples
    for sample_id, metrics in pair_metrics_map.items():
        for metric in metrics:
            # Pairwise doesn't have special context aggregation, use standard count
            ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
            vote_count = len(ratings_data.get("votes", [])) # Standard count for pairwise
            metric_effective_vote_counts[metric].append(vote_count)

    # 4. Calculate coverage per metric (Average number of *effective* votes per applicable entity)
    metric_coverage = {}
    st.write("--- Metric Coverage Calculation ---") # DEBUG
    for metric in all_metrics:
        effective_counts_list = metric_effective_vote_counts.get(metric, [])
        num_entities = 0 # Total applicable entities

        # Determine total applicable entities for this metric
        if metric in PAIRWISE_METRICS:
            num_entities = sum(1 for sid in pair_metrics_map if metric in pair_metrics_map[sid])
        else:
            num_entities = sum(1 for sid in sample_metrics_map if metric in sample_metrics_map[sid])

        if num_entities > 0:
            total_effective_votes = sum(effective_counts_list)
            # Ensure list length matches entity count (should match if logic is correct)
            if len(effective_counts_list) != num_entities:
                 st.warning(f"Mismatch count for metric {metric}: {len(effective_counts_list)} vote counts vs {num_entities} entities. Check logic.")
                 # Fallback or adjust? Let's use num_entities as the divisor.
            average_votes = total_effective_votes / num_entities
            metric_coverage[metric] = average_votes # Store average votes directly
            st.write(f"Metric: {metric}, Entities: {num_entities}, EffectiveVotes: {total_effective_votes}, AvgVotes: {average_votes:.2f}") # DEBUG
        else:
            metric_coverage[metric] = float('inf') # Assign high coverage if no entities apply (won't be selected)
            st.write(f"Metric: {metric}, Entities: 0, Coverage: INF (skipped)") # DEBUG


    # 5. Find the metric with the lowest average effective votes
    # Filter out metrics with infinite coverage (no applicable entities)
    applicable_metrics_coverage = {
        m: cov for m, cov in metric_coverage.items() if cov != float('inf')
    }

    if not applicable_metrics_coverage:
        st.warning("No applicable metrics found with samples/pairs to rate.")
        # Fallback: Maybe pick any metric defined? Or return None.
        defined_metrics_list = list(all_metrics)
        return random.choice(defined_metrics_list) if defined_metrics_list else None

    min_coverage_value = min(applicable_metrics_coverage.values())
    lowest_coverage_metrics = [m for m, cov in applicable_metrics_coverage.items() if cov == min_coverage_value]

    st.write(f"Min Avg Votes: {min_coverage_value:.2f}") # DEBUG
    st.write(f"Metrics with lowest avg votes: {lowest_coverage_metrics}") # DEBUG
    st.write("--- End Debug ---") # DEBUG

    # Choose randomly among metrics with the same lowest coverage
    chosen_metric = random.choice(lowest_coverage_metrics)
    st.info(f"Selected metric: {chosen_metric}") # User Info
    return chosen_metric


# --- MODIFIED get_samples_for_metric ---
def get_samples_for_metric(metric, validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    entities_with_votes = [] # Store tuples of (entity, effective_vote_count)
    if ratings_key not in ratings: ratings[ratings_key] = {}
    user_ratings = ratings.get(ratings_key, {})

    metric_config = evaluation_templates.get(metric, {})
    required_attributes = metric_config.get('required_attributes', [])
    is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
    metric_requires_history = 'history' in required_attributes

    # --- Step 1: Collect relevant entities and their *effective* vote counts ---
    if metric in PAIRWISE_METRICS:
        # (Keep existing logic to find common_ids and iterate)
        samples_a_dict = {s['id']: s for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
        samples_b_dict = {s['id']: s for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
        common_ids = set(samples_a_dict.keys()).intersection(samples_b_dict.keys())

        for sample_id in common_ids:
            # (Keep logic to find sample_a, sample_b, turn_type_a, category_a)
            sample_a = samples_a_dict.get(sample_id)
            sample_b = samples_b_dict.get(sample_id)
            if not sample_a or not sample_b: continue # Should not happen with intersection, but safety check

            turn_type_a, category_a = None, None
            for tt in ["singleturn", "multiturn"]:
                 if tt in validation_data_a:
                     for cat, samples in validation_data_a[tt].items():
                         if any(s.get("id") == sample_id for s in samples):
                             turn_type_a, category_a = tt, cat
                             break
                 if turn_type_a: break

            if turn_type_a and category_a:
                 category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type_a, {}).get(category_a, [])
                 if metric in category_metrics:
                     # (Keep checks for history/context requirements)
                     history_a = sample_a.get('history')
                     pair_has_history = isinstance(history_a, list) and len(history_a) > 0
                     if metric_requires_history and not pair_has_history: continue

                     # Calculate *standard* vote count for pairwise
                     ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
                     vote_count = len(ratings_data.get("votes", []))
                     entities_with_votes.append(((sample_a, sample_b), vote_count))

    else: # Single/Multi-turn metric
        # (Keep existing logic to iterate through samples)
        for turn_type in ["singleturn", "multiturn"]:
            if turn_type in validation_data_a:
                for category, samples in validation_data_a[turn_type].items():
                    category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                    if metric in category_metrics:
                        for sample in samples:
                            sample_id = sample.get("id")
                            if sample_id:
                                # (Keep checks for history/context requirements)
                                retrieved_contexts = sample.get('retrieved_contexts')
                                has_empty_context = not retrieved_contexts
                                history = sample.get('history')
                                has_history = isinstance(history, list) and len(history) > 0

                                if is_context_relevance_metric and has_empty_context: continue
                                if metric_requires_history and not has_history: continue

                                # Calculate *effective* vote count using the helper
                                ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
                                effective_vote_count = get_effective_vote_count(sample_id, metric, ratings_data, validation_data_a)
                                entities_with_votes.append((sample, effective_vote_count))

    # --- Step 2 & 3: Categorize and Order based on TARGET_VOTES (using effective counts) ---
    if not entities_with_votes:
        return []

    zero_votes = []
    under_target = []
    at_or_over_target = []

    for entity, effective_count in entities_with_votes:
        if effective_count == 0:
            zero_votes.append((entity, effective_count))
        elif 0 < effective_count < TARGET_VOTES:
            under_target.append((entity, effective_count))
        else: # effective_count >= TARGET_VOTES
            at_or_over_target.append((entity, effective_count))

    # Sort entities with 1 to TARGET_VOTES-1 votes (ascending by count)
    under_target.sort(key=lambda x: x[1])

    # Shuffle the other groups randomly
    random.shuffle(zero_votes)
    random.shuffle(at_or_over_target)

    # Combine the lists in the prioritized order
    combined_list = under_target + zero_votes + at_or_over_target

    # --- Step 4: Return only the entities ---
    final_ordered_entities = [entity for entity, _ in combined_list]
    return final_ordered_entities


# --- format_chat_history, format_contexts_as_accordion ---
# ... (Keep these functions as they are) ...
def format_chat_history(history_list):
    """Formats a list of chat turns into a readable string for Markdown."""
    formatted_lines = []
    if not isinstance(history_list, list): return "" # Handle invalid input
    for turn in history_list:
        if not isinstance(turn, dict): continue # Skip invalid turns
        role = turn.get('role', 'unknown').lower()
        content = turn.get('content', '[missing content]')

        prefix = "Unbekannt"
        if role == 'user': prefix = "Nutzer"
        elif role in ('assistant', 'bot'): prefix = "Bot"
        elif role == 'system': prefix = "System" # Or decide to skip

        if prefix != "Unbekannt" or role == 'unknown':
            formatted_lines.append(f"**{prefix}:** {html.escape(str(content))}") # Escape content here

    return "\n\n".join(formatted_lines)

def format_contexts_as_accordion(sources, full_contexts):
    """Formats lists of context sources and full texts into an HTML accordion."""
    if not sources or not full_contexts or not isinstance(sources, list) or not isinstance(full_contexts, list) or len(sources) != len(full_contexts):
        # Don't show error to user, just return informative placeholder
        return '<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4><p><em>[Kontextdaten fehlen oder sind inkonsistent.]</em></p></div>'

    accordion_html = '<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4>'
    if not sources: # Handle case where lists are empty
         accordion_html += '<p><em>[Kein Kontext abgerufen.]</em></p>'
    else:
        for i in range(len(sources)):
            source = sources[i] if sources[i] is not None else "[Quelle fehlt]"
            full_context = full_contexts[i] if full_contexts[i] is not None else "[Kontext fehlt]"

            escaped_source = html.escape(str(source))
            escaped_content = html.escape(str(full_context))

            accordion_html += f"""
            <details class="context-details">
                <summary class="context-summary">Kontext {i+1}: {escaped_source}</summary>
                <div class="context-content-body">
                    <p style="white-space: pre-wrap; word-wrap: break-word;">{escaped_content}</p>
                </div>
            </details>
            """

    accordion_html += '</div>'
    return accordion_html


# --- generate_prompt ---
# ... (Keep this function as is, ensuring it uses markdown.markdown correctly) ...
# Minor tweak: Ensure md_converter handles None input gracefully
def generate_prompt(entity, metric, swap_options=False):
    if metric not in evaluation_templates:
        st.error(f"Metric '{metric}' not found in prompt templates.")
        return "[Fehler: Metrik nicht definiert]", []

    template_config = evaluation_templates[metric]
    instruction_text = template_config["prompt"] # Base instruction text
    required_attributes = template_config.get("required_attributes", []) # Use .get for safety
    rating_scale = template_config["rating_scale"]

    missing_attrs = []
    html_parts = [] # List to build HTML sections

    is_pairwise = isinstance(entity, tuple) and len(entity) == 2
    primary_sample = entity[0] if is_pairwise else entity
    if not isinstance(primary_sample, dict): # Safety check
         st.error("Internal Error: Invalid entity structure passed to generate_prompt.")
         return "[Fehler: Ungültige Datenstruktur]", []
    sample_id = primary_sample.get("id", "N/A")

    # --- Determine requirements ---
    requires_history = "history" in required_attributes
    # Check if the metric explicitly requires the 'query' attribute for separate display
    requires_separate_query_display = "query" in required_attributes

    # --- Markdown Conversion Helper ---
    def md_converter(text):
        # ... (keep existing md_converter logic) ...
        if text is None: return "" # Handle None input
        try:
            md_html = markdown.markdown(str(text), extensions=['extra', 'nl2br'])
            return md_html
        except Exception as e:
             st.warning(f"Markdown conversion failed: {e}. Falling back to escaped text.")
             return f"<p>{html.escape(str(text))}</p>" # Fallback to simple paragraph

    # --- SPECIAL HANDLING FOR CONTEXT RELEVANCE METRICS ---
    # (This section already handles query display based on its own logic, no changes needed here)
    if metric in ["context_relevance", "multiturn_context_relevance"]:
        # ... (Keep existing logic for context relevance) ...
        sample = entity # It's not a pair for this metric
        query = sample.get("query")
        sources_list = sample.get('retrieved_contexts')
        full_contexts_list = sample.get('retrieved_contexts_full')
        history_list = sample.get("history") if metric == "multiturn_context_relevance" else None

        # Basic Data Checks
        if not query: missing_attrs.append('query')
        if sources_list is not None and not isinstance(sources_list, list): missing_attrs.append('retrieved_contexts (invalid type)')
        if full_contexts_list is not None and not isinstance(full_contexts_list, list): missing_attrs.append('retrieved_contexts_full (invalid type)')
        if isinstance(sources_list, list) and isinstance(full_contexts_list, list) and len(sources_list) != len(full_contexts_list):
             missing_attrs.append('context lists length mismatch')
             st.warning(f"Context list length mismatch for {sample_id}")

        if metric == "multiturn_context_relevance":
            if not history_list or not isinstance(history_list, list):
                 missing_attrs.append('history (required but missing/invalid)')
                 st.warning(f"History missing/invalid for {sample_id} and metric '{metric}'.")

        # --- Build HTML for Context Relevance Rating ---
        if history_list and isinstance(history_list, list):
            raw_formatted_history = format_chat_history(history_list) # Get raw markdown
            history_html = md_converter(raw_formatted_history) # Convert to HTML
            html_parts.append(f'<div class="history-section"><h4>Gesprächsverlauf:</h4><div class="markdown-content">{history_html}</div></div>') # Insert HTML
        elif metric == "multiturn_context_relevance":
             # Display placeholder if history was required but missing/invalid
             html_parts.append('<div class="history-section"><h4>Gesprächsverlauf:</h4><p><i>[Verlauf fehlt oder ist ungültig]</i></p></div>')


        query_text = html.escape(str(query)) if query else "[Frage fehlt]"
        html_parts.append(f'<div class="query-section"><h4>Frage:</h4><p>{query_text}</p></div>')

        if sources_list is None or full_contexts_list is None or not isinstance(sources_list, list) or not isinstance(full_contexts_list, list):
             html_parts.append("<div class='context-accordion-container'><h4>Verfügbarer Kontext:</h4><p><i>[Kontextdaten fehlen oder sind ungültig.]</i></p></div>")
             rating_scale = []
        elif not sources_list:
             html_parts.append("<div class='context-accordion-container'><h4>Verfügbarer Kontext:</h4><p><i>Für diese Anfrage wurden keine Kontexte abgerufen.</i></p></div>")
             rating_scale = []
        elif len(sources_list) != len(full_contexts_list):
             html_parts.append("<div class='context-accordion-container'><h4>Verfügbarer Kontext:</h4><p><i>[Fehler: Kontextlisten stimmen nicht überein.]</i></p></div>")
             rating_scale = []
        else:
            context_accordion_html = format_contexts_as_accordion(sources_list, full_contexts_list)
            html_parts.append(context_accordion_html)

        styles = """
        <style>
        /* Instruction handled separately */
        /* White background and border for sections */
        .history-section, .query-section, .context-accordion-container, .reference-answer-section, .answer-section {
            margin-bottom: 15px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #ffffff;
        }
        /* Headers */
        .history-section h4, .query-section h4, .reference-answer-section h4, .answer-section h4, .context-accordion-container h4 {
            margin-top: 0; margin-bottom: 8px; font-size: 1.05em; color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px;
        }
        /* Query plain text */
        .query-section p { margin-bottom: 0; line-height: 1.5; word-wrap: break-word; } /* Added word-wrap */

        /* Markdown Content Styling */
        .markdown-content { line-height: 1.5; word-wrap: break-word; }
        .markdown-content p:last-child { margin-bottom: 0; }
        .markdown-content p { margin-bottom: 0.75em; }
        .markdown-content ul, .markdown-content ol { padding-left: 25px; margin-bottom: 0.75em; }
        .markdown-content li { margin-bottom: 0.25em; }
        .markdown-content code { background-color: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; word-wrap: break-word; }
        .markdown-content pre { background-color: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #e9ecef; overflow-x: auto; }
        .markdown-content pre code { background-color: transparent; padding: 0; border-radius: 0; font-size: 0.95em; white-space: pre; }
        .markdown-content blockquote { border-left: 3px solid #ccc; padding-left: 10px; margin-left: 0; color: #666; }
        .markdown-content table { border-collapse: collapse; margin-bottom: 1em; width: auto; }
        .markdown-content th, .markdown-content td { border: 1px solid #ddd; padding: 6px 10px; }
        .markdown-content th { background-color: #f2f2f2; font-weight: bold; }
        /* Ensure links are styled visibly */
        .markdown-content a { color: #007bff; text-decoration: underline; }
        .markdown-content a:hover { color: #0056b3; }

        /* Context Accordion Specific Styles */
        /* .context-accordion-container already styled above */
        .context-details { border: 1px solid #ddd; border-radius: 4px; margin-bottom: 8px; background-color: #fff; }
        .context-summary { cursor: pointer; padding: 10px 15px 10px 35px; font-weight: 500; color: #007bff; list-style: none; position: relative; background-color: #f8f9fa; border-bottom: 1px solid #ddd; }
        .context-details[open] > .context-summary { border-bottom: 1px solid #007bff; }
        .context-summary::-webkit-details-marker { display: none; }
        .context-summary::before { content: '+'; position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-weight: bold; color: #6c757d; margin-right: 8px; font-size: 1.1em; transition: transform 0.2s; }
        .context-details[open] > .context-summary::before { content: '−'; transform: translateY(-50%) rotate(45deg); }
        .context-content-body { padding: 15px 15px 15px 35px; font-size: 0.95em; color: #333; line-height: 1.6; border-top: 1px solid #eee; }
        .context-content-body p { margin-top: 0; margin-bottom: 10px; white-space: pre-wrap; word-wrap: break-word; } /* Keep pre-wrap for context */
        .context-rating-placeholder { margin-top: 15px; border-top: 1px dashed #ccc; padding-top: 15px; }
        </style>
        """
        formatted_prompt = styles + "\n".join(html_parts)
        return formatted_prompt, rating_scale
        # --- END SPECIAL HANDLING FOR CONTEXT RELEVANCE ---


    # --- GENERAL LOGIC FOR OTHER METRICS ---

    # 1. Format History (if required)
    raw_formatted_history = "" # Store the raw markdown string first
    if requires_history:
        history_list = primary_sample.get("history")
        if history_list and isinstance(history_list, list):
            # *** FIX: Assign the result correctly to raw_formatted_history ***
            raw_formatted_history = format_chat_history(history_list) # Get raw markdown
        else:
            # Add placeholder only if history was actually required but missing
            raw_formatted_history = '<i>(Kein Verlauf vorhanden oder zutreffend)</i>' # Use italics tag directly
            if not history_list or not isinstance(history_list, list):
                 missing_attrs.append("history (required but missing/invalid)")

    # 2. *** Conditionally Append Current Query to Formatted History ***
    # Append to the raw markdown string
    if raw_formatted_history and not requires_separate_query_display:
        current_query = primary_sample.get("query")
        if current_query:
            # Append the current query formatted as the last user turn (raw markdown)
            separator = "\n\n" if raw_formatted_history and not raw_formatted_history.startswith('<i>') else ""
            raw_formatted_history += f"{separator}**Nutzer:** {html.escape(str(current_query))}" # Escape query content
        elif "query" in required_attributes:
             missing_attrs.append("query (needed for history context but missing)")

    # 3. *** Convert the final history markdown to HTML and Add History Section ***
    if requires_history:
         # Convert the potentially augmented raw_formatted_history string to HTML
         history_html = md_converter(raw_formatted_history)
         # Insert the generated HTML directly into the div
         html_parts.append(f'<div class="history-section"><h4>Gesprächsverlauf:</h4><div class="markdown-content">{history_html}</div></div>')

    # 4. Add Context Accordion (if required and available)
    # ... (rest of the function remains the same) ...
    metric_displays_context = 'retrieved_contexts' in required_attributes or 'retrieved_contexts_full' in required_attributes
    if metric_displays_context:
        sources_list = primary_sample.get('retrieved_contexts')
        full_contexts_list = primary_sample.get('retrieved_contexts_full')
        valid_context = (
            isinstance(sources_list, list) and
            isinstance(full_contexts_list, list) and
            len(sources_list) == len(full_contexts_list)
        )
        if valid_context:
            context_accordion_html = format_contexts_as_accordion(sources_list, full_contexts_list)
            html_parts.append(context_accordion_html)
        else:
             html_parts.append('<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4><p><em>[Kontextdaten fehlen oder sind ungültig.]</em></p></div>')
             metric_strictly_requires_context = evaluation_templates.get(metric, {}).get("strictly_requires_context", False)
             if metric_strictly_requires_context:
                 missing_attrs.append('context (required but missing/invalid)')


    # 5. Add Separate Query Section (ONLY if required by metric)
    if requires_separate_query_display:
        query = primary_sample.get("query")
        query_text = html.escape(str(query)) if query else "[Frage nicht gefunden]"
        html_parts.append(f'<div class="query-section"><h4>Frage:</h4><p>{query_text}</p></div>')
        if not query: missing_attrs.append("query") # Still note if missing when required

    # 6. Add Reference Answer (if required) - Render as Markdown
    # ... (keep existing reference answer logic) ...
    if "reference_answer" in required_attributes:
        reference_answer = primary_sample.get("reference_answer")
        if reference_answer is not None:
            ref_answer_html = md_converter(reference_answer)
            html_parts.append(f'<div class="reference-answer-section"><h4>Referenzantwort:</h4><div class="markdown-content">{ref_answer_html}</div></div>')
        else:
            html_parts.append('<div class="reference-answer-section"><h4>Referenzantwort:</h4><p>[Referenzantwort nicht gefunden]</p></div>')
            missing_attrs.append("reference_answer")

    # 7. Add Answer(s) - Pairwise or Single - Render as Markdown
    # ... (keep existing answer logic) ...
    if is_pairwise:
        # ... (pairwise logic using md_converter) ...
        sample_a_orig, sample_b_orig = entity
        if not isinstance(sample_a_orig, dict) or not isinstance(sample_b_orig, dict):
             st.error("Internal Error: Invalid pairwise entity structure.")
             return "[Fehler: Ungültige Paar-Datenstruktur]", []

        sample_for_a = sample_b_orig if swap_options else sample_a_orig
        sample_for_b = sample_a_orig if swap_options else sample_b_orig

        attr_a_name = next((attr for attr in required_attributes if attr.endswith('_a')), None)
        attr_b_name = next((attr for attr in required_attributes if attr.endswith('_b')), None)
        base_attr_a = attr_a_name[:-2] if attr_a_name else "answer"
        base_attr_b = attr_b_name[:-2] if attr_b_name else "answer"

        content_a_html = f"[Attribut '{base_attr_a}' nicht gefunden in Sample A]"
        raw_content_a = sample_for_a.get(base_attr_a)
        if raw_content_a is not None: content_a_html = md_converter(raw_content_a)
        else: missing_attrs.append(f"{base_attr_a} (for A)")

        content_b_html = f"[Attribut '{base_attr_b}' nicht gefunden in Sample B]"
        raw_content_b = sample_for_b.get(base_attr_b)
        if raw_content_b is not None: content_b_html = md_converter(raw_content_b)
        else: missing_attrs.append(f"{base_attr_b} (for B)")

        pairwise_html = f"""
        <div class="column-container">
            <div class="column">
                <h3>Antwort A</h3>
                <div class="column-content markdown-content">{content_a_html}</div>
            </div>
            <div class="column">
                <h3>Antwort B</h3>
                <div class="column-content markdown-content">{content_b_html}</div>
            </div>
        </div>
        """
        html_parts.append(pairwise_html)


    elif "answer" in required_attributes: # Handle single answer
        answer = primary_sample.get("answer")
        if answer is not None:
            answer_html = md_converter(answer)
            html_parts.append(f'<div class="answer-section"><h4>Antwort:</h4><div class="markdown-content">{answer_html}</div></div>')
        else:
            html_parts.append('<div class="answer-section"><h4>Antwort:</h4><p>[Antwort nicht gefunden]</p></div>')
            missing_attrs.append("answer")


    # --- Combine HTML Parts and Add Styles ---
    styles = """
    <style>
    /* Instruction handled separately */
    /* White background and border for sections */
    .history-section, .query-section, .context-accordion-container, .reference-answer-section, .answer-section {
        margin-bottom: 15px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #ffffff;
    }
    /* Headers */
    .history-section h4, .query-section h4, .reference-answer-section h4, .answer-section h4, .context-accordion-container h4 {
        margin-top: 0; margin-bottom: 8px; font-size: 1.05em; color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px;
    }
    /* Query plain text */
    .query-section p { margin-bottom: 0; line-height: 1.5; word-wrap: break-word; } /* Added word-wrap */

    /* Markdown Content Styling */
    .markdown-content { line-height: 1.5; word-wrap: break-word; }
    .markdown-content p:last-child { margin-bottom: 0; }
    .markdown-content p { margin-bottom: 0.75em; }
    .markdown-content ul, .markdown-content ol { padding-left: 25px; margin-bottom: 0.75em; }
    .markdown-content li { margin-bottom: 0.25em; }
    .markdown-content code { background-color: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; word-wrap: break-word; }
    .markdown-content pre { background-color: #f8f9fa; padding: 10px; border-radius: 4px; border: 1px solid #e9ecef; overflow-x: auto; }
    .markdown-content pre code { background-color: transparent; padding: 0; border-radius: 0; font-size: 0.95em; white-space: pre; }
    .markdown-content blockquote { border-left: 3px solid #ccc; padding-left: 10px; margin-left: 0; color: #666; }
    .markdown-content table { border-collapse: collapse; margin-bottom: 1em; width: auto; }
    .markdown-content th, .markdown-content td { border: 1px solid #ddd; padding: 6px 10px; }
    .markdown-content th { background-color: #f2f2f2; font-weight: bold; }
    /* Ensure links are styled visibly */
    .markdown-content a { color: #007bff; text-decoration: underline; }
    .markdown-content a:hover { color: #0056b3; }

    /* Context Accordion Specific Styles */
    /* .context-accordion-container already styled above */
    .context-details { border-bottom: 1px solid #eee; margin-bottom: 5px; padding-bottom: 5px; background-color: #fff; border-radius: 4px; }
    .context-details:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
    .context-summary { cursor: pointer; padding: 10px 15px 10px 35px; font-weight: 500; color: #007bff; list-style: none; position: relative; }
    .context-summary::-webkit-details-marker { display: none; }
    .context-summary::before { content: '+'; position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-weight: bold; color: #6c757d; margin-right: 8px; font-size: 1.1em; }
    .context-details[open] > .context-summary::before { content: '−'; }
    .context-summary:hover { color: #0056b3; background-color: #f0f0f0; border-radius: 4px 4px 0 0; }
    .context-content-body { padding: 15px 15px 15px 35px; font-size: 0.95em; color: #333; line-height: 1.6; border-top: 1px solid #eee; }
    .context-content-body p { margin-top: 0; margin-bottom: 10px; white-space: pre-wrap; word-wrap: break-word; } /* Keep pre-wrap for context */
    .context-accordion-container > p { font-style: italic; color: #666; margin-top: 5px; } /* For 'no context' message */

    /* Pairwise Columns */
    .column-container { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 20px; justify-content: center; }
    .column { flex: 1 1 300px; min-width: 280px; border: 1px solid #dcdcdc; padding: 20px; border-radius: 12px; background-color: #ffffff; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05); transition: box-shadow 0.3s ease; }
    .column:hover { box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }
    .column h3 { margin-top: 0; margin-bottom: 15px; font-size: 1.2rem; border-bottom: 1px solid #eee; padding-bottom: 8px; color: #333; }
    .column-content { font-size: 1rem; color: #444; }
    </style>
    """
    formatted_prompt = styles + "\n".join(html_parts)

    # --- Report Missing Attributes ---
    if missing_attrs:
        st.warning(f"Missing/invalid attributes for metric '{metric}' in sample/pair {sample_id}: {', '.join(missing_attrs)}")

    return formatted_prompt, rating_scale


# --- save_rating ---
# (Keep this function as is - it correctly saves the tuple for local context relevance)
def save_rating(sample_id, metric, rating, it_background, is_pairwise, swap_options, supabase_client=None, context_index=None): # Added context_index
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    # Ensure base structure exists in session state BEFORE attempting to append
    if ratings_key not in st.session_state.ratings: st.session_state.ratings[ratings_key] = {}
    if sample_id not in st.session_state.ratings[ratings_key]: st.session_state.ratings[ratings_key][sample_id] = {}
    if metric not in st.session_state.ratings[ratings_key][sample_id]:
        st.session_state.ratings[ratings_key][sample_id][metric] = {"votes": [], "swap_history": []}
    # Ensure lists exist if metric entry was already there but malformed
    if "votes" not in st.session_state.ratings[ratings_key][sample_id][metric] or not isinstance(st.session_state.ratings[ratings_key][sample_id][metric]["votes"], list):
        st.session_state.ratings[ratings_key][sample_id][metric]["votes"] = []
    if "swap_history" not in st.session_state.ratings[ratings_key][sample_id][metric] or not isinstance(st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"], list):
        st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"] = []


    # --- Local JSON Handling ---
    if MODE == "local":
        vote_to_save = rating
        # --- Store context relevance vote as tuple (rating, index) ---
        if metric in ["context_relevance", "multiturn_context_relevance"] and context_index is not None:
            vote_to_save = (rating, context_index)
        # --- END MODIFIED CONDITION ---

        st.session_state.ratings[ratings_key][sample_id][metric]["votes"].append(vote_to_save)

        # Append swap option, ensuring list lengths stay synchronized
        swap_val = swap_options if is_pairwise else None
        st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"].append(swap_val)

        # Ensure lengths match after append (should always match if logic is correct)
        if len(st.session_state.ratings[ratings_key][sample_id][metric]["votes"]) != len(st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"]):
             st.error(f"INTERNAL ERROR after appending vote: Votes/Swap mismatch for {sample_id}/{metric}. Resetting swap history for this vote.")
             # Attempt recovery: Pad swap history to match votes length
             votes_len = len(st.session_state.ratings[ratings_key][sample_id][metric]["votes"])
             st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"] = st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"][:votes_len-1] + [swap_val]


        # Save immediately in local mode
        save_ratings(st.session_state.ratings, None) # Pass None for supabase_client

    # --- Supabase Handling ---
    # This saves ONE rating event (either a single vote or one context vote)
    elif MODE == "supabase" and supabase_client:
        try:
            insert_data = {
                "rater_type": ratings_key,
                "sample_id": sample_id,
                "metric": metric,
                "vote": rating, # Store the raw rating value
                "swap_positions": swap_options if is_pairwise else None,
                "context_index": context_index # Add the context index (will be null if not context relevance)
                # Add rater_id here if available, e.g., "rater_id": st.session_state.user_id
            }

            response = supabase_client.schema("api").table("ratings").insert(insert_data).execute() # Adjust schema if needed

            # --- DEBUGGING: Print response ---
            st.sidebar.write(f"Supabase Response: {response}")
            # --- END DEBUGGING ---

            # Check for errors in the response
            if hasattr(response, 'error') and response.error:
                 st.error(f"Supabase insert error: {response.error}")
                 st.sidebar.error(f"Supabase insert error: {response.error}")
            # Check for data length, indicating success (might vary based on Supabase client version)
            # elif not hasattr(response, 'data') or not response.data:
            #      st.warning("Supabase insert seemed to succeed, but no data returned in response.")

        except Exception as e:
            st.error(f"Error saving rating to Supabase: {e}")
            st.exception(e) # Log full traceback


# --- start_new_round ---
# ... (Keep this function as is, it uses the updated get_lowest_coverage_metric and get_samples_for_metric) ...
def start_new_round():
    # Ensure validation_sets are loaded (assuming they are loaded globally)
    if 'validation_sets' not in globals() or not validation_sets.get('a') or not validation_sets.get('b'):
         st.error("Validation sets not loaded correctly.")
         st.stop()

    validation_data_a = validation_sets['a']
    validation_data_b = validation_sets['b']

    # Add a check for ratings structure before passing
    if not isinstance(st.session_state.get("ratings"), dict):
        st.error("Ratings data is corrupted or not initialized. Reloading.")
        st.session_state.ratings = load_ratings(supabase_client if MODE == "supabase" else None)
        # Add another check after reload
        if not isinstance(st.session_state.get("ratings"), dict):
             st.error("Failed to load valid ratings data. Cannot proceed.")
             st.stop()


    chosen_metric = get_lowest_coverage_metric(
        validation_data_a, validation_data_b, st.session_state.ratings, st.session_state.it_background
    )
    if chosen_metric is None:
        # This might happen if no metrics have applicable samples
        st.info("No suitable metric found with remaining samples to rate. All tasks might be complete.")
        st.session_state.round_over = True # Treat as round over
        # Optionally, display completion message here or let the main loop handle it
        return # Stop processing this round start

    if chosen_metric not in evaluation_templates:
        st.error(f"Error: Chosen metric '{chosen_metric}' not defined in evaluation_templates.")
        st.stop() # Stop if the chosen metric is invalid

    st.session_state.current_metric = chosen_metric
    st.session_state.is_pairwise = chosen_metric in PAIRWISE_METRICS

    # Get samples or pairs for the chosen metric, ordered by the new logic
    st.session_state.entities = get_samples_for_metric(
        st.session_state.current_metric, validation_data_a, validation_data_b, st.session_state.ratings, st.session_state.it_background
    )

    if not st.session_state.entities:
         # This case should ideally be handled by get_lowest_coverage_metric returning None,
         # but as a fallback:
         st.warning(f"No samples/pairs found for the chosen metric '{chosen_metric}', although the metric was selected. Trying next round.")
         # To avoid infinite loops if coverage logic has issues, maybe stop or force a different metric?
         # For now, just mark round over.
         st.session_state.round_over = True
         return

    # Determine number of entities (samples/pairs) for this round
    st.session_state.num_entities_this_round = min(SAMPLES_PER_ROUND, len(st.session_state.entities))
    # Take a sublist for the round - make sure to copy or slice correctly
    st.session_state.entities_this_round = list(st.session_state.entities[:st.session_state.num_entities_this_round]) # Create a copy

    st.session_state.entity_count = 0 # Reset count for the new round
    st.session_state.round_over = False
    st.session_state.round_count += 1
    st.session_state.current_sample = None # Reset single sample view
    st.session_state.current_sample_pair = None # Reset pair view
    st.session_state.user_rating = None # Reset potential leftover rating
    st.session_state.context_ratings = {} # Reset context ratings

    # Load the first entity for the round
    if st.session_state.entities_this_round:
        # Pop from the round's list, not the main list
        current_entity = st.session_state.entities_this_round.pop(0)
        if st.session_state.is_pairwise:
            st.session_state.current_sample_pair = current_entity
            st.session_state.swap_options = random.choice([True, False]) # Randomize swap for pairs
        else:
            st.session_state.current_sample = current_entity
            st.session_state.swap_options = False # No swap for single samples
    else:
        # This shouldn't happen if num_entities_this_round > 0, but safety check
        st.warning("Started round but no entities loaded into the round batch.")
        st.session_state.round_over = True


# --- Load Validation Sets ---
# ... (Keep existing validation set loading and version check logic) ...
validation_sets = {}
try:
    with open(VALIDATION_FILE_A, "r", encoding="utf-8") as f:
        validation_sets['a'] = json.load(f)
    with open(VALIDATION_FILE_B, "r", encoding="utf-8") as f:
        validation_sets['b'] = json.load(f)

    version_a = validation_sets['a'].get("version")
    version_b = validation_sets['b'].get("version")

    if version_a is None or version_b is None:
        st.error(f"Fehler: 'version'-Schlüssel fehlt...")
        st.stop()
    if version_a != version_b:
        st.error(f"Fehler: Versionskonflikt...")
        st.stop()

except FileNotFoundError as e:
    st.error(f"Error: Validation file not found: {e.filename}...")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Error: Failed to decode JSON from validation file: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred loading validation files: {e}")
    st.stop()


# --- Initialize session state ---
# Use setdefault for robustness
st.session_state.setdefault("ratings", {}) # Loaded below
st.session_state.setdefault("entities", [])
st.session_state.setdefault("entities_this_round", [])
st.session_state.setdefault("current_sample", None)
st.session_state.setdefault("current_sample_pair", None)
st.session_state.setdefault("current_metric", None)
st.session_state.setdefault("entity_count", 0)
st.session_state.setdefault("round_over", True)
st.session_state.setdefault("user_rating", None)
st.session_state.setdefault("context_ratings", {}) # For context relevance
st.session_state.setdefault("current_rating_dict_key", None) # Key for context ratings reset
st.session_state.setdefault("num_entities_this_round", 0)
st.session_state.setdefault("app_started", False)
st.session_state.setdefault("it_background", None)
st.session_state.setdefault("round_count", 0)
st.session_state.setdefault("swap_options", False)
st.session_state.setdefault("is_pairwise", False)

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
        st.error(f"An unexpected error during Supabase setup: {e}")
        st.stop()
elif MODE == "local":
    st.session_state.ratings = load_ratings(None)
else:
    st.error(f"Invalid MODE '{MODE}'. Use 'local' or 'supabase'.")
    st.stop()

# --- Main Function ---
def main():
    st.title("RAG Answer Rating App")

    # --- Initial User Setup ---
    if not st.session_state.app_started:
        # ... (keep existing setup logic) ...
        st.write("Willkommen! Bitte geben Sie Ihren IT-Hintergrund an.")
        it_background_choice = st.radio("IT-Hintergrund:", ("Ja", "Nein"), key="it_bg_radio", horizontal=True, index=None)
        if st.button("Start", key="start_btn"):
            if it_background_choice is None:
                st.warning("Bitte wählen Sie eine Option.")
            else:
                st.session_state.it_background = it_background_choice
                st.session_state.app_started = True
                # Trigger the first round immediately after setup
                start_new_round()
                st.rerun() # Rerun to display the first item
        return # Stop execution until setup is complete

    # --- Round Management & Completion Check ---
    if st.session_state.round_over:
        if st.session_state.round_count > 0:
             st.success(f"Runde {st.session_state.round_count} abgeschlossen!")
        # Check if there's potentially more work before showing the button
        # We can try to start a new round silently to see if a metric is found
        # Note: This might be slightly slow if coverage calculation is heavy
        peek_metric = get_lowest_coverage_metric(
            validation_sets['a'], validation_sets['b'], st.session_state.ratings, st.session_state.it_background
        )
        if peek_metric is None:
             st.info("Alle verfügbaren Bewertungsaufgaben sind abgeschlossen. Vielen Dank!")
             # Optionally add a final save action here if needed
             # save_ratings(st.session_state.ratings, supabase_client if MODE == "supabase" else None)
             return # Stop execution, all done
        else:
             # There are more tasks, show the button
             if st.button("Nächste Runde starten", key="next_round_btn"):
                 start_new_round()
                 # Check if start_new_round actually found tasks (it might hit edge cases)
                 if not st.session_state.round_over:
                      st.rerun()
                 # If start_new_round marked it as over again, the rerun won't happen,
                 # and the 'All done' message might show on the next cycle.
        return # Stop execution until button is pressed or all done


    # --- Display Current Item ---
    # Determine the current entity being displayed
    current_entity_for_display = None
    if st.session_state.is_pairwise:
        current_entity_for_display = st.session_state.current_sample_pair
    else:
        current_entity_for_display = st.session_state.current_sample

    # State Check: Ensure we have an entity to display
    if current_entity_for_display is None:
         st.warning("Zustandsfehler: Kein aktuelles Sample/Paar zum Anzeigen. Versuche, neue Runde zu starten.")
         # Log details for debugging
         st.write(f"State: is_pairwise={st.session_state.is_pairwise}, current_sample={st.session_state.current_sample}, current_pair={st.session_state.current_sample_pair}")
         start_new_round()
         # Check if the new round fixed it
         if not st.session_state.round_over:
             st.rerun()
         else:
             # If starting a new round didn't help, likely no more tasks
             st.info("Keine weiteren Aufgaben verfügbar.")
             return # Stop

    # --- Display Header ---
    # ... (Keep existing header display logic) ...
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; text-align: center; width: 100%;">
                Item {st.session_state.entity_count + 1} / {st.session_state.num_entities_this_round} (Runde {st.session_state.round_count}) | Metric: {st.session_state.current_metric}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Display the general intro + specific instruction
    st.markdown(f"**Aufgabe:** {general_intro_prompt}")
    instruction = evaluation_templates.get(st.session_state.current_metric, {}).get("prompt", "[Anweisung fehlt]")
    st.markdown(f'<div class="instruction-section" style="background-color: #eef; padding: 10px; border-radius: 5px; margin-bottom: 15px;">{instruction}</div>', unsafe_allow_html=True)


    # --- Generate and Display the Structured Prompt ---
    try:
        prompt_html, rating_scale = generate_prompt(
            current_entity_for_display,
            st.session_state.current_metric,
            st.session_state.swap_options
        )
        st.write("---") # Separator
        st.html(prompt_html) # Use st.html to render the generated HTML
        st.write("---") # Separator

        # --- Handle cases with no rating needed (e.g., context relevance with no contexts) ---
        # This check is now implicitly handled within generate_prompt for context relevance
        # If rating_scale is empty, it means no rating is possible/needed.
        if not rating_scale:
             st.info("Keine Bewertung für dieses Item erforderlich (z.B. keine Kontexte vorhanden oder Datenfehler).")
             if st.button("Weiter zum nächsten Item", key="skip_no_rating"):
                 # --- Advance Logic (duplicated below, consider function) ---
                 st.session_state.entity_count += 1
                 # Check if round batch is finished
                 if st.session_state.entity_count >= st.session_state.num_entities_this_round or not st.session_state.entities_this_round:
                     st.session_state.round_over = True
                     st.session_state.current_sample = None
                     st.session_state.current_sample_pair = None
                     st.session_state.entities_this_round = [] # Clear the batch list
                 else:
                     # Load next entity from the current round's batch
                     next_entity = st.session_state.entities_this_round.pop(0)
                     if st.session_state.is_pairwise: # Check based on the *metric*, not the entity type
                         st.session_state.current_sample_pair = next_entity
                         st.session_state.current_sample = None
                         st.session_state.swap_options = random.choice([True, False])
                     else:
                         st.session_state.current_sample = next_entity
                         st.session_state.current_sample_pair = None
                         st.session_state.swap_options = False
                     # Reset ratings for the new item
                     st.session_state.user_rating = None
                     st.session_state.context_ratings = {}
                 st.rerun()
             return # Stop further processing for this item

    except Exception as e:
         st.error(f"Unerwarteter Fehler bei der Prompt-Generierung für Metric '{st.session_state.current_metric}': {e}")
         st.exception(e) # Show traceback
         # Provide a way to skip the problematic item
         if st.button("Problem melden und dieses Item überspringen", key="skip_error_item"):
              # Advance logic (same as above)
              st.session_state.entity_count += 1
              if st.session_state.entity_count >= st.session_state.num_entities_this_round or not st.session_state.entities_this_round:
                  st.session_state.round_over = True
                  st.session_state.current_sample = None
                  st.session_state.current_sample_pair = None
                  st.session_state.entities_this_round = []
              else:
                  next_entity = st.session_state.entities_this_round.pop(0)
                  if st.session_state.is_pairwise:
                      st.session_state.current_sample_pair = next_entity
                      st.session_state.current_sample = None
                      st.session_state.swap_options = random.choice([True, False])
                  else:
                      st.session_state.current_sample = next_entity
                      st.session_state.current_sample_pair = None
                      st.session_state.swap_options = False
                  st.session_state.user_rating = None
                  st.session_state.context_ratings = {}
              st.rerun()
         return # Stop further processing


    # --- Rating Input ---
    # Get sample ID robustly
    current_id = None
    if st.session_state.is_pairwise and isinstance(current_entity_for_display, tuple) and current_entity_for_display:
        current_id = current_entity_for_display[0].get('id')
    elif not st.session_state.is_pairwise and isinstance(current_entity_for_display, dict):
        current_id = current_entity_for_display.get('id')

    if not current_id:
        st.error("Konnte Sample ID nicht ermitteln. Überspringe dieses Item.")
        # Add skip logic here if needed, similar to above error handling
        return


    # Special input for context relevance
    if st.session_state.current_metric in ["context_relevance", "multiturn_context_relevance"]:
        sample = current_entity_for_display # Should be a dict
        sources_list = sample.get('retrieved_contexts', [])
        num_contexts = len(sources_list) if isinstance(sources_list, list) else 0

        # Use a unique key for the rating dictionary per item shown
        # Combine round, entity index within round, and sample ID for uniqueness
        rating_dict_key = f"ctx_{st.session_state.round_count}_{st.session_state.entity_count}_{current_id}"

        # Reset context ratings if the key changes (i.e., new item displayed)
        if st.session_state.get('current_rating_dict_key') != rating_dict_key:
            st.session_state.context_ratings = {} # Reset dictionary
            st.session_state.current_rating_dict_key = rating_dict_key # Store the new key

        st.write("**Bewerten Sie jeden Kontext:**")
        all_context_radios_rendered = True
        if num_contexts > 0:
            for i in range(num_contexts):
                radio_key = f"context_rating_{rating_dict_key}_{i}" # Unique key per radio
                # Default to None if not already rated in this session state view
                current_selection = st.session_state.context_ratings.get(i, None)
                index_to_select = None
                str_rating_scale = [str(item) for item in rating_scale] # Ensure string comparison
                if current_selection is not None:
                    try:
                        index_to_select = str_rating_scale.index(str(current_selection))
                    except ValueError:
                        index_to_select = None # Selection not in scale? Reset.

                # Use columns for better layout
                col1, col2 = st.columns([1, 3])
                with col1:
                     st.markdown(f"**Kontext #{i+1}:**") # Display index clearly
                with col2:
                     # Store the selection back into the dictionary using the index 'i' as the key
                     st.session_state.context_ratings[i] = st.radio(
                         f"Relevanz Kontext {i+1}", # Label for screen readers etc.
                         str_rating_scale, # Use string version for radio
                         key=radio_key,
                         horizontal=True,
                         index=index_to_select,
                         label_visibility="collapsed" # Hide label visually
                     )
        else:
             # This case should have been caught earlier (empty rating_scale), but as safety:
             st.info("Keine Kontexte zum Bewerten vorhanden.")
             all_context_radios_rendered = False


    else: # Standard single rating input
        radio_key = f"user_rating_{st.session_state.round_count}_{st.session_state.entity_count}_{current_id}"
        # Default to None if not already rated
        current_selection = st.session_state.get("user_rating", None)
        index_to_select = None
        str_rating_scale = [str(item) for item in rating_scale]
        if current_selection is not None:
             try:
                 index_to_select = str_rating_scale.index(str(current_selection))
             except ValueError:
                 index_to_select = None

        st.session_state.user_rating = st.radio(
            "Ihre Bewertung:",
            str_rating_scale,
            key=radio_key,
            horizontal=True,
            index=index_to_select
        )

    # --- Next Button Logic ---
    if st.button("Weiter", key=f"next_btn_{st.session_state.round_count}_{st.session_state.entity_count}_{current_id}"):
        # --- Validation and Saving ---
        ready_to_advance = False
        try:
            if st.session_state.current_metric in ["context_relevance", "multiturn_context_relevance"]:
                num_contexts_expected = len(current_entity_for_display.get('retrieved_contexts', []))
                # Check if all expected contexts have a rating in the dictionary
                all_rated = True
                if len(st.session_state.context_ratings) != num_contexts_expected:
                    all_rated = False
                else:
                    for i in range(num_contexts_expected):
                         if st.session_state.context_ratings.get(i) is None: # Check for None explicitly
                              all_rated = False
                              break

                if not all_rated and num_contexts_expected > 0: # Only warn if contexts were expected
                    st.warning("Bitte bewerten Sie die Relevanz *aller* angezeigten Kontexte.")
                else:
                    # Save each context rating individually
                    sample_id_to_save = current_id
                    for index, rating in st.session_state.context_ratings.items():
                        # Convert rating back if needed (e.g., if scale was numeric)
                        # Find original value from rating_scale based on string version if necessary
                        original_rating = rating # Assume it's already correct type or string is fine
                        try:
                             # Attempt to find the original type from the template's scale
                             original_scale = evaluation_templates[st.session_state.current_metric]["rating_scale"]
                             original_rating = next(item for item in original_scale if str(item) == str(rating))
                        except (StopIteration, KeyError):
                             st.warning(f"Could not map rating '{rating}' back to original scale type. Saving as string.")
                             original_rating = str(rating) # Save as string if mapping fails


                        save_rating(
                            sample_id=sample_id_to_save,
                            metric=st.session_state.current_metric,
                            rating=original_rating, # Save the potentially type-converted rating
                            it_background=st.session_state.it_background,
                            is_pairwise=False, # Context relevance is never pairwise
                            swap_options=False,
                            supabase_client=supabase_client if MODE == "supabase" else None,
                            context_index=index # Pass the context index
                        )
                    ready_to_advance = True

            else: # Handle single/pairwise ratings
                current_rating = st.session_state.user_rating
                if current_rating is None:
                    st.warning("Bitte wählen Sie eine Bewertung aus.")
                else:
                    sample_id_to_save = current_id
                    # Convert rating back to original type if needed
                    original_rating = current_rating
                    try:
                        original_scale = evaluation_templates[st.session_state.current_metric]["rating_scale"]
                        original_rating = next(item for item in original_scale if str(item) == str(current_rating))
                    except (StopIteration, KeyError):
                        st.warning(f"Could not map rating '{current_rating}' back to original scale type. Saving as string.")
                        original_rating = str(current_rating)

                    save_rating(
                        sample_id_to_save,
                        st.session_state.current_metric,
                        original_rating, # Save potentially type-converted rating
                        st.session_state.it_background,
                        st.session_state.is_pairwise,
                        st.session_state.swap_options,
                        supabase_client if MODE == "supabase" else None,
                        context_index=None # No context index for these metrics
                    )
                    ready_to_advance = True

            # --- Advance to next entity or end round ---
            if ready_to_advance:
                st.session_state.entity_count += 1
                # Check if round batch is finished
                if st.session_state.entity_count >= st.session_state.num_entities_this_round or not st.session_state.entities_this_round:
                    st.session_state.round_over = True
                    st.session_state.current_sample = None
                    st.session_state.current_sample_pair = None
                    st.session_state.entities_this_round = [] # Clear the batch list
                else:
                    # Load next entity from the current round's batch
                    next_entity = st.session_state.entities_this_round.pop(0)
                    # Determine if the *metric* is pairwise to set state correctly
                    if st.session_state.current_metric in PAIRWISE_METRICS:
                        st.session_state.is_pairwise = True # Ensure state is correct
                        st.session_state.current_sample_pair = next_entity
                        st.session_state.current_sample = None
                        st.session_state.swap_options = random.choice([True, False])
                    else:
                        st.session_state.is_pairwise = False # Ensure state is correct
                        st.session_state.current_sample = next_entity
                        st.session_state.current_sample_pair = None
                        st.session_state.swap_options = False
                    # Reset ratings for the new item
                    st.session_state.user_rating = None
                    st.session_state.context_ratings = {}

                st.rerun() # Rerun to display next item or round end message

        except Exception as e:
             st.error(f"Fehler beim Speichern oder Fortfahren: {e}")
             st.exception(e)
             # Consider ending the round on error to prevent loops
             st.session_state.round_over = True
             st.rerun()


if __name__ == "__main__":
    main()

# /home/dns/repos/streamlit-rating-app/rating_app.py
import streamlit as st
import json
import random
import os
import supabase
import html
import markdown
from collections import defaultdict, Counter # Added Counter
from prompt_templates import evaluation_templates

# ... (Keep other constants: DATA_FILE, VALIDATION_FILE_A/B, MODE, TARGET_VOTES, SAMPLES_PER_ROUND, PAIRWISE_METRICS) ...
DATA_FILE = "data/ratings_data.json"
VALIDATION_FILE_A = "data/validationset-test.json"
VALIDATION_FILE_B = "data/validationset-test.json"
MODE = "supabase"
TARGET_VOTES = 3
SAMPLES_PER_ROUND = 3
PAIRWISE_METRICS = {"quality_pairwise", "multiturn_quality_pairwise"}

@st.cache_data # Cache the result of this function
def load_validation_sets(file_a, file_b):
    """Loads validation sets from JSON files and performs version check."""
    validation_sets = {}
    try:
        with open(file_a, "r", encoding="utf-8") as f:
            validation_sets['a'] = json.load(f)
        with open(file_b, "r", encoding="utf-8") as f:
            validation_sets['b'] = json.load(f)

        version_a = validation_sets['a'].get("version")
        version_b = validation_sets['b'].get("version")

        if version_a is None or version_b is None:
            raise ValueError(f"Fehler: 'version'-Schlüssel fehlt in {file_a} oder {file_b}.")
        if version_a != version_b:
            raise ValueError(f"Fehler: Versionskonflikt zwischen {file_a} (v{version_a}) und {file_b} (v{version_b}).")

        # Basic structure validation (optional but recommended)
        for key in ['a', 'b']:
            if not isinstance(validation_sets[key].get("evaluation_criteria"), dict):
                 st.warning(f"Validation set '{key}' missing or has invalid 'evaluation_criteria'.")
            # Add more checks as needed

        st.success(f"Validation sets (Version {version_a}) loaded successfully.") # Log success
        return validation_sets

    except FileNotFoundError as e:
        st.error(f"Error: Validation file not found: {e.filename}. Please ensure '{file_a}' and '{file_b}' exist.")
        st.stop() # Stop execution if files are missing
    except json.JSONDecodeError as e:
        st.error(f"Error: Failed to decode JSON from validation file: {e}. Check file content.")
        st.stop()
    except ValueError as e: # Catch our custom version errors
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading validation files: {e}")
        st.exception(e) # Show traceback for unexpected errors
        st.stop()


# --- Cached Function: Calculate Metric Mappings ---
@st.cache_data # Cache the mapping based on the validation data content
def calculate_metric_mappings(_validation_data_a, _validation_data_b):
    """
    Calculates which metrics apply to which samples or pairs based on validation data.
    Takes copies (_var) as input for caching based on content.
    """
    sample_metrics_map = defaultdict(list)
    pair_metrics_map = defaultdict(list)

    # --- Process validation_data_a for single/multi-turn metrics ---
    for turn_type in ["singleturn", "multiturn"]:
        if turn_type in _validation_data_a:
            for category, samples in _validation_data_a[turn_type].items():
                category_metrics = _validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                for sample in samples:
                    sample_id = sample.get("id")
                    if sample_id:
                        retrieved_contexts = sample.get('retrieved_contexts')
                        # Check if retrieved_contexts is explicitly None or an empty list
                        has_empty_context = retrieved_contexts is None or (isinstance(retrieved_contexts, list) and not retrieved_contexts)
                        history = sample.get('history')
                        has_history = isinstance(history, list) and len(history) > 0

                        for metric in category_metrics:
                            if metric in PAIRWISE_METRICS: continue # Skip pairwise here

                            metric_config = evaluation_templates.get(metric, {})
                            metric_req_attrs = metric_config.get('required_attributes', [])
                            is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
                            metric_requires_history = 'history' in metric_req_attrs
                            metric_strictly_requires_context = metric_config.get("strictly_requires_context", False)

                            # Skip checks
                            if is_context_relevance_metric and has_empty_context: continue
                            if metric_strictly_requires_context and has_empty_context: continue # Skip if context is strictly needed but missing
                            if metric_requires_history and not has_history: continue

                            sample_metrics_map[sample_id].append(metric)

    # --- Identify valid pairs and map pairwise metrics ---
    ids_a = {s['id'] for turn_type in _validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in _validation_data_a[turn_type] for s in _validation_data_a[turn_type][cat]}
    ids_b = {s['id'] for turn_type in _validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in _validation_data_b[turn_type] for s in _validation_data_b[turn_type][cat]}
    common_ids = ids_a.intersection(ids_b)

    samples_a_dict = {s['id']: s for turn_type in _validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in _validation_data_a[turn_type] for s in _validation_data_a[turn_type][cat]}

    for sample_id in common_ids:
        sample_a = samples_a_dict.get(sample_id)
        if not sample_a: continue

        turn_type_a, category_a = None, None
        # Find category for sample_a
        for tt in ["singleturn", "multiturn"]:
             if tt in _validation_data_a:
                 for cat, samples in _validation_data_a[tt].items():
                     if any(s.get("id") == sample_id for s in samples):
                         turn_type_a, category_a = tt, cat
                         break
             if turn_type_a: break

        if turn_type_a and category_a:
             category_metrics = _validation_data_a.get("evaluation_criteria", {}).get(turn_type_a, {}).get(category_a, [])
             applicable_pairwise = [m for m in category_metrics if m in PAIRWISE_METRICS]

             if applicable_pairwise:
                 history_a = sample_a.get('history')
                 pair_has_history = isinstance(history_a, list) and len(history_a) > 0
                 # Add context check if pairwise metrics can depend on context
                 # retrieved_contexts_a = sample_a.get('retrieved_contexts')
                 # pair_has_context = retrieved_contexts_a is not None and (isinstance(retrieved_contexts_a, list) and retrieved_contexts_a)

                 for metric in applicable_pairwise:
                     metric_config = evaluation_templates.get(metric, {})
                     metric_req_attrs = metric_config.get('required_attributes', [])
                     metric_requires_history = 'history' in metric_req_attrs
                     # metric_strictly_requires_context = metric_config.get("strictly_requires_context", False) # Example if needed

                     if metric_requires_history and not pair_has_history: continue
                     # if metric_strictly_requires_context and not pair_has_context: continue

                     pair_metrics_map[sample_id].append(metric)
                     # st.write(f"Pairwise metric {metric} applicable to sample ID {sample_id}") # DEBUG (keep low verbosity in cached func)

    return sample_metrics_map, pair_metrics_map

# --- NEW Helper Function: Get Effective Vote Count ---
def get_effective_vote_count(sample_id, metric, ratings_data_for_sample_metric, validation_data_a):
    """
    Calculates the effective number of votes for a sample/metric,
    handling the special aggregation for context relevance metrics in local mode.
    """
    is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
    votes_list = ratings_data_for_sample_metric.get("votes", [])

    if not is_context_relevance_metric:
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

# --- Supabase Initialization (Consider caching the client resource) ---
@st.cache_resource # Cache the Supabase client object
def init_supabase():
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or Key not found in secrets")
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client

# --- Load/Save Functions (Ensure they handle the potential tuple format for context votes locally) ---
# Modify load_ratings slightly to ensure structure even if file is empty/new
def load_ratings(supabase_client): # Pass the client
    ratings = {"Experten": {}, "Crowd": {}}
    if MODE == "supabase":
        if not supabase_client:
             st.error("Supabase client not available for loading ratings.")
             return ratings # Return empty if client failed
        try:
            # ... (rest of Supabase load logic using supabase_client) ...
            response = supabase_client.schema("api").table("ratings").select("*").execute()
            # ... (processing logic) ...
            data = response.data

            for row in data:
                rater_type = row["rater_type"]
                sample_id = row["sample_id"] # This ID represents the sample or sample pair
                metric = row["metric"]
                vote = row["vote"]
                swap_positions = row.get("swap_positions")
                context_index = row.get("context_index") # Fetch if needed

                if rater_type not in ratings: ratings[rater_type] = {} # Ensure rater_type exists
                if sample_id not in ratings[rater_type]:
                    ratings[rater_type][sample_id] = {}
                if metric not in ratings[rater_type][sample_id]:
                    ratings[rater_type][sample_id][metric] = {"votes": [], "swap_history": []}

                # Store vote as (vote, context_index) tuple for context relevance metrics
                is_context_relevance = metric in ["context_relevance", "multiturn_context_relevance"]
                if is_context_relevance and context_index is not None:
                    vote_to_store = (vote, context_index)
                else:
                    # Store raw vote for other metrics or if context_index is missing
                    vote_to_store = vote
                    if is_context_relevance and context_index is None:
                         st.warning(f"Context index missing for context relevance metric '{metric}' in record: {row}. Storing raw vote.")

                ratings[rater_type][sample_id][metric]["votes"].append(vote_to_store)
                

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
    else: # local mode
        # ... (Keep existing local load logic) ...
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f: # Ensure correct encoding
                    loaded_data = json.load(f)
                    # More robust check for top-level structure
                    if isinstance(loaded_data, dict):
                        ratings["Experten"] = loaded_data.get("Experten", {})
                        ratings["Crowd"] = loaded_data.get("Crowd", {})
                    else:
                        st.warning(f"Local data file {DATA_FILE} does not contain a valid dictionary. Starting fresh.")
                        ratings = {"Experten": {}, "Crowd": {}}

                    # Deep validation and structure correction
                    for group, group_data in ratings.items():
                        if not isinstance(group_data, dict):
                            ratings[group] = {}
                            continue
                        for sample_id, sample_data in list(group_data.items()): # Iterate over copy
                            if not isinstance(sample_data, dict):
                                ratings[group][sample_id] = {}
                                continue
                            for metric, metric_data in list(sample_data.items()): # Iterate over copy
                                if not isinstance(metric_data, dict):
                                    ratings[group][sample_id][metric] = {"votes": [], "swap_history": []}
                                    continue

                                votes = metric_data.get("votes", [])
                                swaps = metric_data.get("swap_history", [])

                                if not isinstance(votes, list): votes = []
                                if not isinstance(swaps, list): swaps = []

                                # Ensure swap_history length matches votes length
                                votes_len = len(votes)
                                swap_len = len(swaps)
                                if swap_len < votes_len:
                                    swaps.extend([None] * (votes_len - swap_len))
                                elif swap_len > votes_len:
                                     swaps = swaps[:votes_len] # Truncate extra swaps

                                ratings[group][sample_id][metric] = {"votes": votes, "swap_history": swaps}

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


def save_ratings(ratings, supabase_client): # supabase_client is unused here now
    # ... (Keep consistency check logic) ...
    ratings_copy = json.loads(json.dumps(ratings)) # Deep copy

    for group in ratings_copy:
        for sample_id in ratings_copy[group]:
            for metric in ratings_copy[group][sample_id]:
                # Ensure keys exist before accessing
                metric_data = ratings_copy[group][sample_id][metric]
                votes = metric_data.get("votes", [])
                swaps = metric_data.get("swap_history", [])

                if not isinstance(votes, list): votes = []
                if not isinstance(swaps, list): swaps = []

                if len(votes) != len(swaps):
                     st.warning(f"SAVE WARNING: Mismatch votes ({len(votes)}) / swap_history ({len(swaps)}) for {sample_id}/{metric}. Attempting fix.")
                     swap_len = len(swaps)
                     votes_len = len(votes)
                     if swap_len < votes_len:
                         swaps.extend([None] * (votes_len - swap_len))
                     else:
                         swaps = swaps[:votes_len]
                     # Update the copy
                     ratings_copy[group][sample_id][metric]["swap_history"] = swaps
                     # Re-check
                     if len(ratings_copy[group][sample_id][metric]["votes"]) != len(ratings_copy[group][sample_id][metric]["swap_history"]):
                          st.error(f"--> FIX FAILED for {sample_id}/{metric}. Data might be inconsistent.")
                          # Decide whether to skip saving this metric or save the potentially fixed version

    if MODE == "supabase":
        # Individual saves happen in save_rating, nothing to do here.
        pass
    else: # local mode
        try:
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(ratings_copy, f, indent=4, ensure_ascii=False)
        except IOError as e:
            st.error(f"Error saving ratings locally to {DATA_FILE}: {e}")
        except Exception as e:
             st.error(f"Unexpected error saving ratings locally: {e}")


# --- MODIFIED get_lowest_coverage_metric ---
def get_lowest_coverage_metric(validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    # st.write("--- Debug: Calculating Lowest Coverage Metric ---") # Keep debug low

    # 1. Get cached metric mappings
    # Pass the actual validation data to the cached function
    sample_metrics_map, pair_metrics_map = calculate_metric_mappings(validation_data_a, validation_data_b)

    # 2. Get all possible metrics from templates
    all_metrics = set(evaluation_templates.keys())
    if not all_metrics:
        st.warning("No evaluation templates found.")
        return None

    # 3. Calculate effective vote counts per metric using the helper function
    metric_effective_vote_counts = defaultdict(list)
    if ratings_key not in ratings: ratings[ratings_key] = {}
    user_ratings = ratings.get(ratings_key, {})

    # Calculate for single/multi-turn samples using the map
    for sample_id, metrics in sample_metrics_map.items():
        for metric in metrics:
            ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
            # Pass validation_data_a needed by the helper for context relevance logic
            effective_count = get_effective_vote_count(sample_id, metric, ratings_data, validation_data_a)
            metric_effective_vote_counts[metric].append(effective_count)

    # Calculate for pairwise samples using the map
    for sample_id, metrics in pair_metrics_map.items():
        for metric in metrics:
            ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
            vote_count = len(ratings_data.get("votes", [])) # Standard count for pairwise
            metric_effective_vote_counts[metric].append(vote_count)

    # 4. Calculate coverage per metric
    metric_coverage = {}
    # st.write("--- Metric Coverage Calculation ---") # Keep debug low
    for metric in all_metrics:
        effective_counts_list = metric_effective_vote_counts.get(metric, [])
        num_entities = 0 # Total applicable entities based on maps

        if metric in PAIRWISE_METRICS:
            # Count entities from the pair_metrics_map
            num_entities = sum(1 for sid in pair_metrics_map if metric in pair_metrics_map[sid])
        else:
            # Count entities from the sample_metrics_map
            num_entities = sum(1 for sid in sample_metrics_map if metric in sample_metrics_map[sid])

        if num_entities > 0:
            total_effective_votes = sum(effective_counts_list)
            # Sanity check: list length should ideally match entity count from maps
            if len(effective_counts_list) != num_entities:
                 st.warning(f"Coverage Calc Warning: Mismatch count for metric {metric}: {len(effective_counts_list)} vote counts vs {num_entities} entities from map. Using map count.")
            average_votes = total_effective_votes / num_entities
            metric_coverage[metric] = average_votes
            st.write(f"Metric: {metric}, Entities: {num_entities}, AvgVotes: {average_votes:.2f}") # DEBUG
        else:
            metric_coverage[metric] = float('inf') # Assign high coverage if no entities apply
            # st.write(f"Metric: {metric}, Entities: 0, Coverage: INF") # DEBUG

    # 5. Find the metric with the lowest average effective votes
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

    # st.write(f"Min Avg Votes: {min_coverage_value:.2f}") # DEBUG
    # st.write(f"Metrics with lowest avg votes: {lowest_coverage_metrics}") # DEBUG
    # st.write("--- End Debug ---") # DEBUG

    chosen_metric = random.choice(lowest_coverage_metrics)
    return chosen_metric


# --- MODIFIED get_samples_for_metric ---
def get_samples_for_metric(metric, validation_data_a, validation_data_b, ratings, it_background):
    # (Keep the existing logic, it seems correct for ordering based on votes)
    # Ensure it uses the validation_data_a passed in, which comes from the cached source.
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    entities_with_votes = []
    if ratings_key not in ratings: ratings[ratings_key] = {}
    user_ratings = ratings.get(ratings_key, {})

    metric_config = evaluation_templates.get(metric, {})
    required_attributes = metric_config.get('required_attributes', [])
    is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
    metric_requires_history = 'history' in required_attributes
    metric_strictly_requires_context = metric_config.get("strictly_requires_context", False) # Added check

    # --- Step 1: Collect relevant entities and their *effective* vote counts ---
    if metric in PAIRWISE_METRICS:
        samples_a_dict = {s['id']: s for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
        samples_b_dict = {s['id']: s for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
        common_ids = set(samples_a_dict.keys()).intersection(samples_b_dict.keys())

        for sample_id in common_ids:
            sample_a = samples_a_dict.get(sample_id)
            sample_b = samples_b_dict.get(sample_id)
            if not sample_a or not sample_b: continue

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
                     history_a = sample_a.get('history')
                     pair_has_history = isinstance(history_a, list) and len(history_a) > 0
                     # Add context check if needed for pairwise
                     # retrieved_contexts_a = sample_a.get('retrieved_contexts')
                     # pair_has_context = retrieved_contexts_a is not None and (isinstance(retrieved_contexts_a, list) and retrieved_contexts_a)

                     if metric_requires_history and not pair_has_history: continue
                     # if metric_strictly_requires_context and not pair_has_context: continue # Example

                     ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
                     vote_count = len(ratings_data.get("votes", [])) # Standard count for pairwise
                     entities_with_votes.append(((sample_a, sample_b), vote_count))

    else: # Single/Multi-turn metric
        for turn_type in ["singleturn", "multiturn"]:
            if turn_type in validation_data_a:
                for category, samples in validation_data_a[turn_type].items():
                    category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                    if metric in category_metrics:
                        for sample in samples:
                            sample_id = sample.get("id")
                            if sample_id:
                                retrieved_contexts = sample.get('retrieved_contexts')
                                has_empty_context = retrieved_contexts is None or (isinstance(retrieved_contexts, list) and not retrieved_contexts)
                                history = sample.get('history')
                                has_history = isinstance(history, list) and len(history) > 0

                                if is_context_relevance_metric and has_empty_context: continue
                                if metric_strictly_requires_context and has_empty_context: continue # Check strict requirement
                                if metric_requires_history and not has_history: continue

                                ratings_data = user_ratings.get(sample_id, {}).get(metric, {})
                                # Use validation_data_a here for context relevance calculation
                                effective_vote_count = get_effective_vote_count(sample_id, metric, ratings_data, validation_data_a)
                                entities_with_votes.append((sample, effective_vote_count))

    # --- Step 2 & 3: Categorize and Order ---
    # (Keep the sorting/shuffling logic)
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

    under_target.sort(key=lambda x: x[1])
    random.shuffle(zero_votes)
    random.shuffle(at_or_over_target)
    combined_list = under_target + zero_votes + at_or_over_target

    # --- Step 4: Return only the entities ---
    final_ordered_entities = [entity for entity, _ in combined_list]
    return final_ordered_entities


# --- format_chat_history, format_contexts_as_accordion ---
# (Keep these functions as they are)
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
            # Escape content before adding prefix for safety
            escaped_content = html.escape(str(content))
            formatted_lines.append(f"**{prefix}:** {escaped_content}")

    return "\n\n".join(formatted_lines)


def format_contexts_as_accordion(sources, full_contexts):
    """Formats lists of context sources and full texts into an HTML accordion."""
    if not sources or not full_contexts or not isinstance(sources, list) or not isinstance(full_contexts, list) or len(sources) != len(full_contexts):
        # Don't show error to user, just return informative placeholder
        return '<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4><p><em>[Kein Kontext vorhanden.]</em></p></div>'

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
    # ... (initial setup: template_config, required_attributes, rating_scale, missing_attrs, html_parts, is_pairwise, primary_sample, sample_id) ...
    if metric not in evaluation_templates:
        st.error(f"Metric '{metric}' not found in prompt templates.")
        return "[Fehler: Metrik nicht definiert]", []

    template_config = evaluation_templates[metric]
    instruction_text = template_config["prompt"] # Base instruction text
    required_attributes = template_config.get("required_attributes", []) # Use .get for safety
    rating_scale = template_config["rating_scale"] # Keep rating scale needed by main loop

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
    requires_separate_query_display = "query" in required_attributes
    metric_displays_context = 'retrieved_contexts' in required_attributes or 'retrieved_contexts_full' in required_attributes
    is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"] # Check if it's the special metric

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

    # --- GENERAL LOGIC FOR ALL METRICS ---

    # 1. Format History (if required)
    # ... (keep existing history formatting logic) ...
    raw_formatted_history = ""
    if requires_history:
        history_list = primary_sample.get("history")
        if history_list and isinstance(history_list, list):
            raw_formatted_history = format_chat_history(history_list)
        else:
            raw_formatted_history = '<i>(Kein Verlauf vorhanden oder zutreffend)</i>'
            if not history_list or not isinstance(history_list, list):
                 missing_attrs.append("history (required but missing/invalid)")

    # 2. Conditionally Append Current Query to Formatted History
    # ... (keep existing logic) ...
    if raw_formatted_history and not requires_separate_query_display:
        current_query = primary_sample.get("query")
        if current_query:
            separator = "\n\n" if raw_formatted_history and not raw_formatted_history.startswith('<i>') else ""
            raw_formatted_history += f"{separator}**Nutzer:** {html.escape(str(current_query))}"
        elif "query" in required_attributes:
             missing_attrs.append("query (needed for history context but missing)")

    # 3. Convert History to HTML and Add Section
    # ... (keep existing logic) ...
    if requires_history:
         history_html = md_converter(raw_formatted_history)
         html_parts.append(f'<div class="history-section"><h4>Gesprächsverlauf:</h4><div class="markdown-content">{history_html}</div></div>')


    # 4. Add Context Accordion (if required AND NOT context_relevance metric)
    if metric_displays_context:
        sources_list = primary_sample.get('retrieved_contexts')
        full_contexts_list = primary_sample.get('retrieved_contexts_full')
        valid_context = (
            isinstance(sources_list, list) and
            isinstance(full_contexts_list, list) and
            len(sources_list) == len(full_contexts_list)
        )

        # --- MODIFICATION START ---
        # Only render the accordion if it's NOT a context relevance metric
        if not is_context_relevance_metric:
            if valid_context:
                # Use the standard function which generates the desired accordion style
                context_accordion_html = format_contexts_as_accordion(sources_list, full_contexts_list)
                html_parts.append(context_accordion_html)
            else:
                 # Display consistent placeholder/error message for other metrics
                 html_parts.append('<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4><p><em>[Kontextdaten fehlen oder sind ungültig.]</em></p></div>')
                 # Check if context was strictly required
                 metric_strictly_requires_context = template_config.get("strictly_requires_context", False)
                 if metric_strictly_requires_context:
                     missing_attrs.append('context (required but missing/invalid)')
        # --- MODIFICATION END ---

        # --- Keep this check for context relevance metrics, even if not rendering accordion ---
        # If context is invalid/missing for context_relevance, rating is impossible
        if is_context_relevance_metric and not valid_context:
            missing_attrs.append('context (required for rating but missing/invalid)')
            rating_scale = [] # Disable rating by clearing the scale


    # 5. Add Separate Query Section (ONLY if required by metric)
    # ... (keep existing logic) ...
    if requires_separate_query_display:
        query = primary_sample.get("query")
        query_text = html.escape(str(query)) if query else "[Frage nicht gefunden]"
        html_parts.append(f'<div class="query-section"><h4>Frage:</h4><p>{query_text}</p></div>')
        if not query: missing_attrs.append("query")

    # 6. Add Reference Answer (if required)
    # ... (keep existing logic) ...
    if "reference_answer" in required_attributes:
        reference_answer = primary_sample.get("reference_answer")
        if reference_answer is not None:
            ref_answer_html = md_converter(reference_answer)
            html_parts.append(f'<div class="reference-answer-section"><h4>Referenzantwort:</h4><div class="markdown-content">{ref_answer_html}</div></div>')
        else:
            html_parts.append('<div class="reference-answer-section"><h4>Referenzantwort:</h4><p>[Referenzantwort nicht gefunden]</p></div>')
            missing_attrs.append("reference_answer")

    # 7. Add Answer(s) - Pairwise or Single
    # ... (keep existing logic) ...
    if is_pairwise:
        # ... (pairwise HTML generation) ...
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
    # Styles remain the same, including context accordion styles for other metrics
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
    .query-section p { margin-bottom: 0; line-height: 1.5; word-wrap: break-word; }

    /* Markdown Content Styling */
    /* ... (keep existing markdown styles) ... */
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
    .markdown-content a { color: #007bff; text-decoration: underline; }
    .markdown-content a:hover { color: #0056b3; }

    /* Context Accordion Specific Styles (still needed for other metrics) */
    /* ... (keep existing context accordion styles) ... */
    .context-details { border-bottom: 1px solid #eee; margin-bottom: 5px; padding-bottom: 5px; background-color: #fff; border-radius: 4px; }
    .context-details:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
    .context-summary { cursor: pointer; padding: 10px 15px 10px 35px; font-weight: 500; color: #007bff; list-style: none; position: relative; }
    .context-summary::-webkit-details-marker { display: none; }
    .context-summary::before { content: '+'; position: absolute; left: 10px; top: 50%; transform: translateY(-50%); font-weight: bold; color: #6c757d; margin-right: 8px; font-size: 1.1em; }
    .context-details[open] > .context-summary::before { content: '−'; }
    .context-summary:hover { color: #0056b3; background-color: #f0f0f0; border-radius: 4px 4px 0 0; }
    .context-content-body { padding: 15px 15px 15px 35px; font-size: 0.95em; color: #333; line-height: 1.6; border-top: 1px solid #eee; }
    .context-content-body p { margin-top: 0; margin-bottom: 10px; white-space: pre-wrap; word-wrap: break-word; }
    .context-accordion-container > p { font-style: italic; color: #666; margin-top: 5px; }

    /* Pairwise Columns */
    /* ... (keep existing pairwise styles) ... */
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

    # Return the HTML (without context accordion for context_relevance) and the rating scale
    return formatted_prompt, rating_scale


# --- save_rating ---
def save_rating(sample_id, metric, rating, it_background, is_pairwise, swap_options, supabase_client=None, context_index=None):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    # Ensure base structure exists in session state
    # (Keep existing session state update logic)
    if ratings_key not in st.session_state.ratings: st.session_state.ratings[ratings_key] = {}
    if sample_id not in st.session_state.ratings[ratings_key]: st.session_state.ratings[ratings_key][sample_id] = {}
    if metric not in st.session_state.ratings[ratings_key][sample_id]:
        st.session_state.ratings[ratings_key][sample_id][metric] = {"votes": [], "swap_history": []}
    if "votes" not in st.session_state.ratings[ratings_key][sample_id][metric] or not isinstance(st.session_state.ratings[ratings_key][sample_id][metric]["votes"], list):
        st.session_state.ratings[ratings_key][sample_id][metric]["votes"] = []
    if "swap_history" not in st.session_state.ratings[ratings_key][sample_id][metric] or not isinstance(st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"], list):
        st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"] = []


    # --- Local JSON Handling ---
    if MODE == "local":
        vote_to_save = rating
        if metric in ["context_relevance", "multiturn_context_relevance"] and context_index is not None:
            vote_to_save = (rating, context_index)

        st.session_state.ratings[ratings_key][sample_id][metric]["votes"].append(vote_to_save)
        swap_val = swap_options if is_pairwise else None
        st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"].append(swap_val)

        # (Keep consistency check and fix attempt)
        if len(st.session_state.ratings[ratings_key][sample_id][metric]["votes"]) != len(st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"]):
             st.error(f"INTERNAL ERROR after appending vote: Votes/Swap mismatch for {sample_id}/{metric}. Resetting swap history for this vote.")
             votes_len = len(st.session_state.ratings[ratings_key][sample_id][metric]["votes"])
             current_swaps = st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"]
             st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"] = current_swaps[:votes_len-1] + [swap_val]

        save_ratings(st.session_state.ratings, None) # Save immediately

    # --- Supabase Handling ---
    elif MODE == "supabase" and supabase_client: # Check if client is valid
        try:
            insert_data = {
                "rater_type": ratings_key,
                "sample_id": sample_id,
                "metric": metric,
                "vote": rating,
                "swap_positions": swap_options if is_pairwise else None,
                "context_index": context_index
                # "rater_id": st.session_state.get("user_id") # Example if you add user tracking
            }
            response = supabase_client.schema("api").table("ratings").insert(insert_data).execute()

            # More robust error checking based on Supabase client library structure
            if hasattr(response, 'error') and response.error:
                 st.error(f"Supabase insert error: {response.error}")
            # elif not hasattr(response, 'data') or not response.data:
                 # Some versions might return empty data on success, adjust if needed
                 # st.warning("Supabase insert successful, but no data returned in response.")
            # else:
                 # st.sidebar.write(f"Supabase insert successful: {response.data}") # Optional success log

        except Exception as e:
            st.error(f"Error saving rating to Supabase: {e}")
            st.exception(e)
    elif MODE == "supabase" and not supabase_client:
         st.error("Cannot save rating: Supabase client not initialized.")




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

# --- Callback function for context relevance radio buttons ---
def update_context_rating(context_idx, radio_widget_key):
    """
    Callback to update the central context_ratings dictionary
    when an individual context radio button changes.
    """
    # Read the value directly from the widget's state using its key
    new_value = st.session_state.get(radio_widget_key)
    if new_value is not None:
        # Update the central dictionary using the context index
        st.session_state.context_ratings[context_idx] = new_value
    # else: Optionally handle if the widget state is None, though unlikely for radio


# --- Main Function ---
def main():
    st.title("Bewertung von Chatbot-Antworten")

    # --- Initial User Setup ---
    if not st.session_state.app_started:
        # ... (keep existing setup logic) ...
        st.markdown("""
        ### 🎓 Hintergrund der Studie

        Diese Umfrage wird im Rahmen einer **Masterarbeit** im Studiengang Data Science durchgeführt. Ziel ist es die **Qualität eines Chatbots** zu untersuchen. Dabei wird gemessen, wie stark die Einschätzungen von Menschen mit denen großer Sprachmodelle  übereinstimmen.
                    
        ---

        ### 🤖 Zum Chatbot

        Der Chatbot wurde für einen Anbieter von **IT-Seminaren** entwickelt und beantwortet Fragen rund um das Seminarprogramm.

        Technisch basiert der Bot auf dem **RAG-Prinzip (Retrieval-Augmented Generation)**. Das bedeutet, er sucht zuerst passende Informationen aus einer Wissensdatenbank (Kontext) und erstellt daraufhin eine Antwort.

        ---

        ### 📝 Ablauf der Bewertung

        Sie bewerten **ein bestimmtes Kriterium** anhand von **10 Beispielen**.
        Diese Beispiele sind unterschiedlich umfangreich und stammen teils aus realen Anfragen, teils sind sie zu Testzwecken erstellt.

        ---

        ### 🧑‍💻 Vor dem Start:

        Bitte geben Sie an, ob Sie über einen **IT-Hintergrund** verfügen (z. B. durch Ausbildung, Studium oder Beruf).
        """)

        it_background_choice = st.radio(
            "IT-Hintergrund:",
            ("Ja", "Nein"),
            key="it_bg_radio",
            horizontal=True,
            index=None
        )
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
             st.success(f"Danke für die Teilnahme!")
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
                Beispiel {st.session_state.entity_count + 1} / {st.session_state.num_entities_this_round}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Display the general intro + specific instruction
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
    primary_entity = current_entity_for_display[0] if st.session_state.is_pairwise else current_entity_for_display
    if isinstance(primary_entity, dict):
        current_id = primary_entity.get('id')

    if not current_id:
        # ... (keep existing ID error handling) ...
        st.error("Konnte Sample ID nicht ermitteln. Überspringe dieses Item.")
        if st.button("Item überspringen (ID Fehler)", key="skip_id_error"):
             # ... (advance logic) ...
             st.rerun()
        return

    # --- Context Relevance Rating Input ---
    if st.session_state.current_metric in ["context_relevance", "multiturn_context_relevance"]:
        # ... (Keep existing context relevance logic with st.expander and on_change callback) ...
        sample = primary_entity
        sources_list = sample.get('retrieved_contexts', [])
        full_contexts_list = sample.get('retrieved_contexts_full', [])
        num_contexts = len(sources_list)

        rating_dict_key = f"ctx_{st.session_state.round_count}_{st.session_state.entity_count}_{current_id}"
        if st.session_state.get('current_rating_dict_key') != rating_dict_key:
            st.session_state.context_ratings = {}
            st.session_state.current_rating_dict_key = rating_dict_key
        
        metric_config = evaluation_templates.get(st.session_state.current_metric, {})
        final_question = metric_config.get("final_question", "Bitte bewerten:") # Fallback text

        # --- REMOVED: st.markdown(f"**{final_question}**") ---
        st.subheader("Bewerten Sie jeden einzelnen Kontext:")

        all_context_radios_rendered = True

        if num_contexts > 0:
            if not rating_scale:
                 st.error("Interner Fehler: Rating Scale fehlt für Context Relevance.")
                 return

            str_rating_scale_ctx = [str(item) for item in rating_scale] # Use a different name just in case

            for i in range(num_contexts):
                # ... (expander setup) ...
                expander_label = f"Kontext {i+1}: {html.escape(str(sources_list[i] if sources_list[i] is not None else '[Quelle fehlt]'))}"
                with st.expander(expander_label, expanded=False):
                    full_context = full_contexts_list[i] if full_contexts_list[i] is not None else "[Kontext fehlt]"
                    st.text(f"{full_context}")

                    st.markdown(f"**{final_question}**") # Display the question here

                    radio_key = f"context_rating_{rating_dict_key}_{i}"
                    current_selection = st.session_state.context_ratings.get(i, None)
                    index_to_select = None
                    if current_selection is not None:
                        try:
                            index_to_select = str_rating_scale_ctx.index(str(current_selection))
                        except ValueError:
                            index_to_select = None


                    st.radio(
                         f"Relevanz Kontext {i+1}",
                         str_rating_scale_ctx, # Use the specific variable
                         key=radio_key,
                         horizontal=True,
                         index=index_to_select,
                         label_visibility="collapsed",
                         on_change=update_context_rating,
                         kwargs=dict(context_idx=i, radio_widget_key=radio_key)
                     )
        else:
             st.info("Keine Kontexte zum Bewerten vorhanden für dieses Item.")
             all_context_radios_rendered = False


    # --- Standard single/pairwise rating input (for other metrics like quality_pairwise) ---
    else:
        radio_key = f"user_rating_{st.session_state.round_count}_{st.session_state.entity_count}_{current_id}"
        current_selection = st.session_state.get("user_rating", None)
        index_to_select = None
        str_rating_scale = None # Initialize to None

        # Check if rating_scale is valid before proceeding
        if not rating_scale: # rating_scale comes from generate_prompt
             st.error(f"Interner Fehler: Rating Scale fehlt für Metric {st.session_state.current_metric}.")
             # If rating_scale is missing, we cannot display the radio button, so return.
             # Add a skip button?
             if st.button("Item überspringen (Scale Fehler)", key="skip_scale_error"):
                  # ... (advance logic) ...
                  st.session_state.entity_count += 1
                  # ... (rest of advance logic) ...
                  st.rerun()
             return # Stop if scale is missing
        else:
             # If rating_scale is valid, create the string version
             try:
                str_rating_scale = [str(item) for item in rating_scale]
             except TypeError:
                 st.error(f"Interner Fehler: Rating Scale für Metric {st.session_state.current_metric} ist kein gültiger Iterable.")
                 # Add skip button?
                 if st.button("Item überspringen (Scale Fehler)", key="skip_scale_error_type"):
                      # ... (advance logic) ...
                      st.session_state.entity_count += 1
                      # ... (rest of advance logic) ...
                      st.rerun()
                 return # Stop if scale is invalid type

        # Now, str_rating_scale should be a list. Proceed to find the index.
        if current_selection is not None:
             try:
                 # We know str_rating_scale is a list here
                 index_to_select = str_rating_scale.index(str(current_selection))
             except ValueError:
                 # The current selection is not in the expected scale (maybe from old data?)
                 index_to_select = None
                 st.warning(f"Vorherige Auswahl '{current_selection}' nicht in aktueller Skala gefunden.")


        # Display the radio button - str_rating_scale must be valid list here
        st.session_state.user_rating = st.radio(
            f"Bitte bewerte: {evaluation_templates[st.session_state.current_metric]['final_question']}",
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

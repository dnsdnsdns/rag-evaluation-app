import streamlit as st
import json
import random
import os
import supabase
import html
import markdown
from collections import defaultdict
from prompt_templates import evaluation_templates, general_intro_prompt

DATA_FILE = "data/ratings_data.json"
VALIDATION_FILE_A = "data/validationset.json"
# --- Added second validation file constant ---
VALIDATION_FILE_B = "data/validationset-b.json"
MODE = "supabase"  # supported modes: "local", "supabase"

# --- Define Pairwise Metrics ---
PAIRWISE_METRICS = {"quality_pairwise", "multiturn_quality_pairwise"}
#PAIRWISE_METRICS = {}

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
                swap_positions = row.get("swap_positions")
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
# --- Adapted get_lowest_coverage_metric ---
def get_lowest_coverage_metric(validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    # 1. Get all possible metrics (No change needed)
    all_metrics = set()
    for turn_type, criteria in validation_data_a.get("evaluation_criteria", {}).items():
        for category, metrics in criteria.items():
            all_metrics.update(metrics)
    all_metrics.update(PAIRWISE_METRICS)

    if not all_metrics:
        st.warning("No evaluation criteria found in validation set.")
        return None

    # 2. Map samples/pairs to their applicable metrics (No change needed)
    sample_metrics_map = defaultdict(list)
    pair_metrics_map = defaultdict(list)

    # Process validation_data_a for single/multi-turn metrics
    for turn_type in ["singleturn", "multiturn"]:
        if turn_type in validation_data_a:
            for category, samples in validation_data_a[turn_type].items():
                category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                for sample in samples:
                    sample_id = sample.get("id")
                    if sample_id:
                        retrieved_contexts = sample.get('retrieved_contexts')
                        has_empty_context = not retrieved_contexts # True if None or empty list
                        history = sample.get('history') # Check for history
                        has_history = isinstance(history, list) and len(history) > 0

                        for metric in category_metrics:
                            if metric in PAIRWISE_METRICS:
                                continue

                            metric_config = evaluation_templates.get(metric, {})
                            metric_req_attrs = metric_config.get('required_attributes', [])

                            is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
                            metric_requires_history = 'history' in metric_req_attrs

                            # Skip if context relevance metric requires context but sample has none
                            if is_context_relevance_metric and has_empty_context:
                                continue
                            # Skip if metric requires history but sample has none
                            if metric_requires_history and not has_history:
                                continue

                            sample_metrics_map[sample_id].append(metric)

    # Identify valid pairs and map pairwise metrics (No change needed)
    ids_a = {s['id'] for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
    ids_b = {s['id'] for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
    common_ids = ids_a.intersection(ids_b)

    samples_a_dict = {s['id']: s for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}

    for sample_id in common_ids:
        sample_a = samples_a_dict.get(sample_id)
        if not sample_a: continue

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
             applicable_pairwise = [m for m in category_metrics if m in PAIRWISE_METRICS]

             if applicable_pairwise:
                 retrieved_contexts_a = sample_a.get('retrieved_contexts')
                 pair_has_empty_context = not retrieved_contexts_a
                 history_a = sample_a.get('history')
                 pair_has_history = isinstance(history_a, list) and len(history_a) > 0

                 for metric in applicable_pairwise:
                     metric_config = evaluation_templates.get(metric, {})
                     metric_req_attrs = metric_config.get('required_attributes', [])
                     metric_requires_history = 'history' in metric_req_attrs

                     if metric_requires_history and not pair_has_history:
                         continue
                     # Add context skip logic if needed for pairwise

                     pair_metrics_map[sample_id].append(metric)

    # 3. Calculate vote counts per metric (No change needed here, we use this info below)
    metric_vote_counts = defaultdict(list) # Stores a list of vote counts (one per applicable entity)
    if ratings_key not in ratings: ratings[ratings_key] = {}

    for sample_id, metrics in sample_metrics_map.items():
        for metric in metrics:
            votes_data = ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", [])
            # For context relevance, count each individual rating (tuple in local, row in supabase)
            # For others, count each rating entry
            vote_count = len(votes_data) # This is the number of votes for THIS sample/metric
            metric_vote_counts[metric].append(vote_count)

    for sample_id, metrics in pair_metrics_map.items():
        for metric in metrics:
            vote_count = len(ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", []))
            metric_vote_counts[metric].append(vote_count)

    # --- MODIFIED SECTION: Calculate coverage based on total votes ---
    # 4. Calculate coverage per metric (Average number of votes per applicable entity)
    metric_coverage = {}
    for metric in all_metrics:
        vote_counts_list = metric_vote_counts.get(metric, []) # List of vote counts for this metric (e.g., [2, 0, 1, 3])
        num_entities = 0 # Total number of entities (samples/pairs) applicable to this metric

        # Determine total applicable entities for this metric
        if metric in PAIRWISE_METRICS:
            num_entities = sum(1 for sample_id in pair_metrics_map if metric in pair_metrics_map[sample_id])
        else:
            num_entities = sum(1 for sample_id in sample_metrics_map if metric in sample_metrics_map[sample_id])

        if num_entities > 0:
            # Sum the total number of votes across all applicable entities
            total_votes = sum(vote_counts_list)

            # Coverage = (Total Votes / Total Applicable Entities) * 100
            # This represents the average number of votes per entity, expressed as a percentage.
            # It can exceed 100 if entities are rated multiple times on average.
            metric_coverage[metric] = (total_votes / num_entities) * 100.0
        else:
            # If no entities are applicable for this metric, coverage is 0
            metric_coverage[metric] = 0

    # --- END MODIFIED SECTION ---

    # 5. Find the metric with the lowest coverage (No change needed in this logic)
    # Filter out metrics that have no applicable samples/pairs before finding the minimum
    applicable_metrics_with_coverage = {
        m: cov for m, cov in metric_coverage.items()
        if (m in PAIRWISE_METRICS and any(m in metrics for metrics in pair_metrics_map.values())) or \
           (m not in PAIRWISE_METRICS and any(m in metrics for metrics in sample_metrics_map.values()))
    }

    if not applicable_metrics_with_coverage:
        st.warning("No applicable metrics found with samples/pairs to rate.")
        # Fallback: Maybe pick any metric defined? Or return None.
        all_defined_metrics = list(all_metrics)
        return random.choice(all_defined_metrics) if all_defined_metrics else None

    min_coverage_value = min(applicable_metrics_with_coverage.values())
    lowest_coverage_metrics = [m for m, cov in applicable_metrics_with_coverage.items() if cov == min_coverage_value]

    return random.choice(lowest_coverage_metrics) if lowest_coverage_metrics else None # Should always find one if applicable_metrics_with_coverage is not empty



# --- Adapted get_samples_for_metric ---
def get_samples_for_metric(metric, validation_data_a, validation_data_b, ratings, it_background):
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"
    relevant_entities = []
    if ratings_key not in ratings: ratings[ratings_key] = {}

    metric_config = evaluation_templates.get(metric, {})
    required_attributes = metric_config.get('required_attributes', [])
    # --- MODIFIED CHECK ---
    is_context_relevance_metric = metric in ["context_relevance", "multiturn_context_relevance"]
    metric_requires_history = 'history' in required_attributes
    # --- END MODIFIED CHECK ---

    if metric in PAIRWISE_METRICS:
        samples_a_dict = {s['id']: s for turn_type in validation_data_a if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_a[turn_type] for s in validation_data_a[turn_type][cat]}
        samples_b_dict = {s['id']: s for turn_type in validation_data_b if turn_type in ['singleturn', 'multiturn'] for cat in validation_data_b[turn_type] for s in validation_data_b[turn_type][cat]}
        common_ids = set(samples_a_dict.keys()).intersection(samples_b_dict.keys())

        for sample_id in common_ids:
            sample_a = samples_a_dict[sample_id]
            sample_b = samples_b_dict[sample_id]

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
                     # --- MODIFIED CHECK ---
                     retrieved_contexts_a = sample_a.get('retrieved_contexts')
                     pair_has_empty_context = not retrieved_contexts_a
                     history_a = sample_a.get('history')
                     pair_has_history = isinstance(history_a, list) and len(history_a) > 0

                     # Skip if context relevance metric requires context but pair lacks it (if applicable to pairwise)
                     # if is_context_relevance_metric and pair_has_empty_context: continue
                     # Skip if metric requires history but pair lacks it
                     if metric_requires_history and not pair_has_history:
                         continue
                     # --- END MODIFIED CHECK ---

                     vote_count = len(ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", []))
                     relevant_entities.append(((sample_a, sample_b), vote_count))

    else: # Single/Multi-turn metric
        for turn_type in ["singleturn", "multiturn"]:
            if turn_type in validation_data_a:
                for category, samples in validation_data_a[turn_type].items():
                    category_metrics = validation_data_a.get("evaluation_criteria", {}).get(turn_type, {}).get(category, [])
                    if metric in category_metrics:
                        for sample in samples:
                            sample_id = sample.get("id")
                            if sample_id:
                                # --- MODIFIED CHECK ---
                                retrieved_contexts = sample.get('retrieved_contexts')
                                has_empty_context = not retrieved_contexts
                                history = sample.get('history')
                                has_history = isinstance(history, list) and len(history) > 0

                                if is_context_relevance_metric and has_empty_context:
                                    continue
                                if metric_requires_history and not has_history:
                                    continue
                                # --- END MODIFIED CHECK ---

                                # Handle potential tuple format for context relevance votes in local mode
                                votes_data = ratings[ratings_key].get(sample_id, {}).get(metric, {}).get("votes", [])
                                if metric in ["context_relevance", "multiturn_context_relevance"] and MODE == "local":
                                     # Use the number of context ratings (tuples) as the sort key
                                     vote_count = len([v for v in votes_data if isinstance(v, (tuple, list))])
                                else:
                                     vote_count = len(votes_data)

                                relevant_entities.append((sample, vote_count))

    relevant_entities.sort(key=lambda x: x[1])
    return [entity for entity, _ in relevant_entities]

def format_chat_history(history_list):
    """Formats a list of chat turns into a readable string for Markdown."""
    formatted_lines = []
    for turn in history_list:
        role = turn.get('role', 'unknown').lower()
        # Get raw content, Markdown conversion will happen later
        content = turn.get('content', '[missing content]')

        prefix = "Unbekannt"
        if role == 'user':
            prefix = "Nutzer"
        elif role == 'assistant' or role == 'bot':
            prefix = "Bot"
        elif role == 'system':
             prefix = "System" # Or decide to skip system messages

        if prefix != "Unbekannt" or role == 'unknown':
            # Use Markdown bold for the prefix
            formatted_lines.append(f"**{prefix}:** {content}")

    # Join lines with double newlines for paragraph breaks in Markdown
    return "\n\n".join(formatted_lines)

# --- Helper function for formatting contexts (Keep using html.escape and pre-wrap) ---
def format_contexts_as_accordion(sources, full_contexts):
    """Formats lists of context sources and full texts into an HTML accordion."""
    if not sources or not full_contexts or len(sources) != len(full_contexts):
        st.error("Internal Error: format_contexts_as_accordion called with invalid/empty data.")
        return "<p><em>[Interner Fehler bei der Anzeige des Kontexts.]</em></p>"

    accordion_html = '<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4>'

    for i in range(len(sources)):
        source = sources[i]
        full_context = full_contexts[i]

        escaped_source = html.escape(str(source))
        # Escape content but keep newlines for CSS pre-wrap
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

# --- Adapted generate_prompt ---
def generate_prompt(entity, metric, swap_options=False):
    # --- (Initial checks remain the same) ---
    if metric not in evaluation_templates:
        st.error(f"Metric '{metric}' not found in prompt templates.")
        return "[Fehler: Metrik nicht definiert]", []

    template_config = evaluation_templates[metric]
    instruction_text = template_config["prompt"] # Base instruction text
    required_attributes = template_config["required_attributes"]
    rating_scale = template_config["rating_scale"]

    missing_attrs = []
    html_parts = [] # List to build HTML sections

    is_pairwise = isinstance(entity, tuple) and len(entity) == 2
    primary_sample = entity[0] if is_pairwise else entity
    sample_id = primary_sample.get("id", "N/A")

    # --- Markdown Conversion Helper ---
    # Use 'extra' which includes nl2br, fenced_code, tables, auto-linking, etc.
    # Explicitly add 'linkify=True' if using a version where 'extra' doesn't force it
    # Or use a dedicated extension like 'markdown_linkify.LinkifyExtension' if needed,
    # but 'extra' usually handles standard URLs.
    md_converter = lambda text: markdown.markdown(str(text), extensions=['extra']) # Ensure input is string

    # --- SPECIAL HANDLING FOR CONTEXT RELEVANCE METRICS ---
    if metric in ["context_relevance", "multiturn_context_relevance"]:
        sample = entity
        query = sample.get("query")
        sources_list = sample.get('retrieved_contexts')
        full_contexts_list = sample.get('retrieved_contexts_full')
        history_list = sample.get("history", []) if metric == "multiturn_context_relevance" else None

        # Basic Data Checks (keep as is)
        if not query: missing_attrs.append('query')
        if not isinstance(sources_list, list) or not isinstance(full_contexts_list, list) or len(sources_list) != len(full_contexts_list):
            st.error(f"Invalid or missing context data for sample {sample_id} and metric '{metric}'.")
            return "[Fehler: Ungültige Kontextdaten]", []
        if metric == "multiturn_context_relevance" and not history_list:
             st.error(f"Internal Error: History missing for sample {sample_id} and metric '{metric}'.")
             missing_attrs.append('history (required but missing)')
             return "[Fehler: Verlauf fehlt]", []

        # --- Build HTML for Context Relevance Rating ---
        # Add History (if applicable) - Render as Markdown
        if history_list:
            raw_history = format_chat_history(history_list)
            history_html = md_converter(raw_history)
            html_parts.append(f'<div class="history-section"><h4>Gesprächsverlauf:</h4><div class="markdown-content">{history_html}</div></div>')

        # Add Query - Keep as plain text escaped
        if query:
             html_parts.append(f'<div class="query-section"><h4>Frage:</h4><p>{html.escape(str(query))}</p></div>') # Ensure string
        else:
             html_parts.append('<div class="query-section"><h4>Frage:</h4><p>[Frage fehlt]</p></div>')

        # Add Context Accordion (Keep using pre-wrap for context content)
        if not sources_list:
             html_parts.append("<p><i>Für diese Anfrage wurden keine Kontexte abgerufen. Es gibt nichts zu bewerten.</i></p>")
             rating_scale = []
        else:
            context_accordion_html = format_contexts_as_accordion(sources_list, full_contexts_list)
            html_parts.append(context_accordion_html)

        # --- Combine and Return for Context Relevance ---
        # ***** CSS (Ensure box styles are present) *****
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
        # ***** END CSS *****
        formatted_prompt = styles + "\n".join(html_parts)
        return formatted_prompt, rating_scale
        # --- END SPECIAL HANDLING FOR CONTEXT RELEVANCE ---


    # --- GENERAL LOGIC FOR OTHER METRICS ---

    # Instruction is handled outside this function now

    # 2. Add History (if required) - Render as Markdown
    if "history" in required_attributes:
        history_list = primary_sample.get("history", [])
        if history_list:
            raw_history = format_chat_history(history_list)
            history_html = md_converter(raw_history)
            html_parts.append(f'<div class="history-section"><h4>Gesprächsverlauf:</h4><div class="markdown-content">{history_html}</div></div>')
        else:
            # Keep placeholder for missing history
            html_parts.append('<div class="history-section"><h4>Gesprächsverlauf:</h4><p><i>(Kein Verlauf vorhanden oder zutreffend)</i></p></div>')
            if not any(isinstance(t, dict) and t.get('role') and t.get('content') for t in history_list):
                 missing_attrs.append("history (required but missing/invalid)")

    # 3. Add Context Accordion (if required and available) - Keep using pre-wrap
    metric_displays_context = 'retrieved_contexts' in required_attributes or 'retrieved_contexts_full' in required_attributes
    if metric_displays_context:
        sources_list = primary_sample.get('retrieved_contexts')
        full_contexts_list = primary_sample.get('retrieved_contexts_full')
        context_keys_present = isinstance(sources_list, list) and isinstance(full_contexts_list, list)
        context_lists_match = context_keys_present and len(sources_list) == len(full_contexts_list)
        context_is_empty = not sources_list if context_keys_present else True

        if context_lists_match and not context_is_empty:
            context_accordion_html = format_contexts_as_accordion(sources_list, full_contexts_list)
            html_parts.append(context_accordion_html)
        elif context_is_empty:
             # Keep placeholder for empty context
             html_parts.append('<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4><p><em>Für diese Anfrage steht kein Kontext zur Verfügung.</em></p></div>')
             metric_strictly_requires_context = evaluation_templates.get(metric, {}).get("strictly_requires_context", False)
             if metric_strictly_requires_context: missing_attrs.append('context (required but missing)')
        else:
            # Keep warning for inconsistent context
            st.warning(f"Context data invalid/inconsistent for sample {sample_id}. Check 'retrieved_contexts' and 'retrieved_contexts_full'.")
            html_parts.append('<div class="context-accordion-container"><h4>Verfügbarer Kontext:</h4><p><em>[Fehler bei der Anzeige des Kontexts: Inkonsistente Daten.]</em></p></div>')
            metric_strictly_requires_context = evaluation_templates.get(metric, {}).get("strictly_requires_context", False)
            if metric_strictly_requires_context: missing_attrs.append('context (inconsistent data)')

    # 4. Add Query (if required) - Keep as plain text escaped
    if "query" in required_attributes:
        query = primary_sample.get("query")
        if query:
            html_parts.append(f'<div class="query-section"><h4>Frage:</h4><p>{html.escape(str(query))}</p></div>') # Ensure string
        else:
            html_parts.append('<div class="query-section"><h4>Frage:</h4><p>[Frage nicht gefunden]</p></div>')
            missing_attrs.append("query")

    # 5. Add Reference Answer (if required) - Render as Markdown
    if "reference_answer" in required_attributes:
        reference_answer = primary_sample.get("reference_answer")
        if reference_answer is not None:
            ref_answer_html = md_converter(reference_answer)
            html_parts.append(f'<div class="reference-answer-section"><h4>Referenzantwort:</h4><div class="markdown-content">{ref_answer_html}</div></div>')
        else:
            html_parts.append('<div class="reference-answer-section"><h4>Referenzantwort:</h4><p>[Referenzantwort nicht gefunden]</p></div>')
            missing_attrs.append("reference_answer")

    # 6. Add Answer(s) - Pairwise or Single - Render as Markdown
    if is_pairwise:
        sample_a_orig, sample_b_orig = entity
        sample_for_a = sample_b_orig if swap_options else sample_a_orig
        sample_for_b = sample_a_orig if swap_options else sample_b_orig

        attr_a_name = next((attr for attr in required_attributes if attr.endswith('_a')), 'answer_a')
        attr_b_name = next((attr for attr in required_attributes if attr.endswith('_b')), 'answer_b')
        base_attr_a = attr_a_name[:-2]
        base_attr_b = attr_b_name[:-2]

        content_a_html = f"[Attribut '{base_attr_a}' nicht gefunden]"
        if base_attr_a in sample_for_a:
            raw_content_a = sample_for_a[base_attr_a]
            content_a_html = md_converter(raw_content_a)
        else:
            missing_attrs.append(attr_a_name)

        content_b_html = f"[Attribut '{base_attr_b}' nicht gefunden]"
        if base_attr_b in sample_for_b:
            raw_content_b = sample_for_b[base_attr_b]
            content_b_html = md_converter(raw_content_b)
        else:
            missing_attrs.append(attr_b_name)

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
    # ***** CSS (Ensure box styles are present) *****
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
    # ***** END CSS *****
    formatted_prompt = styles + "\n".join(html_parts)

    # --- Report Missing Attributes ---
    if missing_attrs:
        display_missing = [attr for attr in missing_attrs if not attr.startswith(('context (', 'history ('))]
        if display_missing:
            st.warning(f"Missing required attributes {display_missing} for metric '{metric}' in sample/pair {sample_id}.")

    return formatted_prompt, rating_scale

# --- Adapted save_rating ---
def save_rating(sample_id, metric, rating, it_background, is_pairwise, swap_options, supabase_client=None, context_index=None): # Added context_index
    ratings_key = "Experten" if it_background == "Ja" else "Crowd"

    # --- Local JSON Handling ---
    if MODE == "local":
        if ratings_key not in st.session_state.ratings: st.session_state.ratings[ratings_key] = {}
        if sample_id not in st.session_state.ratings[ratings_key]: st.session_state.ratings[ratings_key][sample_id] = {}
        if metric not in st.session_state.ratings[ratings_key][sample_id]:
            st.session_state.ratings[ratings_key][sample_id][metric] = {"votes": [], "swap_history": []}
        elif "votes" not in st.session_state.ratings[ratings_key][sample_id][metric]:
             st.session_state.ratings[ratings_key][sample_id][metric]["votes"] = []
        if "swap_history" not in st.session_state.ratings[ratings_key][sample_id][metric]:
             st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"] = []

        # Store rating, potentially with context index for context_relevance metrics
        vote_to_save = rating
        # --- MODIFIED CONDITION ---
        if metric in ["context_relevance", "multiturn_context_relevance"] and context_index is not None:
            # Store as a tuple (rating, context_index)
            vote_to_save = (rating, context_index)
            # Optional: Keep the warning or remove if expected behavior
            # st.warning(f"Local Mode: Saving {metric} rating for index {context_index} as a tuple in 'votes'. Ensure downstream processing handles this.")
        # --- END MODIFIED CONDITION ---

        st.session_state.ratings[ratings_key][sample_id][metric]["votes"].append(vote_to_save)
        st.session_state.ratings[ratings_key][sample_id][metric]["swap_history"].append(swap_options if is_pairwise else None)

        # Save immediately in local mode
        save_ratings(st.session_state.ratings, None) # Pass None for supabase_client

    # --- Supabase Handling ---
    elif MODE == "supabase" and supabase_client:
        try:
            insert_data = {
                "rater_type": ratings_key,
                "sample_id": sample_id,
                "metric": metric,
                "vote": rating, # Store the raw rating value
                "swap_positions": swap_options if is_pairwise else None,
                "context_index": context_index # Add the context index
            }
            response = supabase_client.schema("api").table("ratings").insert(insert_data).execute()
            if hasattr(response, 'error') and response.error:
                 st.error(f"Supabase insert error: {response.error}")
        except Exception as e:
            st.error(f"Error saving rating to Supabase: {e}")


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
        # ... (keep existing setup logic) ...
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
                start_new_round()
                st.rerun()
        return

    # --- Round Management ---
    if st.session_state.round_over:
        # ... (keep existing round end/start logic) ...
        if st.session_state.round_count > 0:
             st.success(f"Runde {st.session_state.round_count} abgeschlossen. Danke für Ihre Bewertungen!")
        if st.button("Nächste Runde starten", key="next_round_button"):
            start_new_round()
            if not st.session_state.round_over:
                 st.rerun()
        return

    # --- State Check ---
    current_entity_for_display = st.session_state.current_sample_pair if st.session_state.is_pairwise else st.session_state.current_sample
    if not current_entity_for_display and not st.session_state.round_over:
         st.warning("Zustandsfehler: Kein aktuelles Sample/Paar. Starte neue Runde.")
         start_new_round()
         if not st.session_state.round_over: st.rerun()
         else:
             st.info("Keine weiteren Aufgaben verfügbar.")
             return

    # --- Display Header ---
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

    # Display the general intro prompt
    st.write(general_intro_prompt)
    # '<div class="instruction-section">{html.escape(instruction_text)}</div>'
    # st.write(f"{evaluation_templates[st.session_state.current_metric]['prompt']}")
    # display above two lines as a single line
    st.markdown(f'<div class="instruction-section">{evaluation_templates[st.session_state.current_metric]["prompt"]}</div>', unsafe_allow_html=True)

    # --- Generate and Display the Structured Prompt ---
    try:
        prompt_html, rating_scale = generate_prompt(
            current_entity_for_display,
            st.session_state.current_metric,
            st.session_state.swap_options
        )
        st.write("---") # Separator before the main content
        st.html(prompt_html) # Use st.html to render the generated HTML structure
        st.write("---") # Separator after the main content

        # --- Handle cases with no rating needed (e.g., context relevance with no contexts) ---
        if not rating_scale and st.session_state.current_metric in ["context_relevance", "multiturn_context_relevance"]:
             st.info("Keine Bewertung für dieses Item erforderlich (z.B. keine Kontexte vorhanden).")
             if st.button("Weiter zum nächsten Item", key="skip_no_rating"):
                 # ... (skip logic remains the same) ...
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
                 st.rerun()
             return # Stop further processing for this item

    except ValueError as e:
        st.error(f"Fehler beim Generieren des Prompts: {e}")
        st.button("Problem melden und nächste Runde starten", on_click=lambda: setattr(st.session_state, 'round_over', True))
        return # Stop further processing for this item
    except Exception as e:
         st.error(f"Unerwarteter Fehler bei der Prompt-Generierung: {e}")
         st.exception(e) # Show traceback for unexpected errors
         st.button("Problem melden und nächste Runde starten", on_click=lambda: setattr(st.session_state, 'round_over', True))
         return # Stop further processing for this item


    # --- Rating Input ---
    # (Keep the existing rating input logic, including the special handling for context_relevance)
    current_id = current_entity_for_display[0]['id'] if st.session_state.is_pairwise else current_entity_for_display['id']

    if st.session_state.current_metric in ["context_relevance", "multiturn_context_relevance"]:
        # ... (Keep context relevance rating input logic) ...
        sample = current_entity_for_display
        sources_list = sample.get('retrieved_contexts', [])
        num_contexts = len(sources_list)
        st.session_state.setdefault('context_ratings', {})

        rating_dict_key = f"{st.session_state.round_count}_{current_id}_{st.session_state.entity_count}"
        if st.session_state.get('current_rating_dict_key') != rating_dict_key:
            st.session_state.context_ratings = {}
            st.session_state.current_rating_dict_key = rating_dict_key

        st.write("**Bewerten Sie jeden Kontext:**")
        for i in range(num_contexts):
            radio_key = f"context_rating_{rating_dict_key}_{i}"
            col1, col2 = st.columns([1, 3])
            with col1:
                 st.markdown(f"**Kontext #{i+1} Bewertung:**")
            with col2:
                 current_selection = st.session_state.context_ratings.get(i)
                 index_to_select = None
                 if current_selection is not None:
                     try:
                         str_rating_scale = [str(item) for item in rating_scale]
                         index_to_select = str_rating_scale.index(str(current_selection))
                     except ValueError:
                         index_to_select = None

                 st.session_state.context_ratings[i] = st.radio(
                     f"Relevanz Kontext {i+1}",
                     rating_scale,
                     key=radio_key,
                     horizontal=True,
                     index=index_to_select,
                     label_visibility="collapsed"
                 )

    else:
        # Existing logic for single rating
        radio_key = f"user_rating_{st.session_state.round_count}_{current_id}_{st.session_state.entity_count}"
        st.session_state.user_rating = st.radio(
            "Ihre Bewertung:",
            rating_scale,
            key=radio_key,
            horizontal=True,
            index=None # Or restore index if needed
        )

    # --- Next Button Logic ---
    # (Keep the existing "Weiter" button logic for saving ratings and advancing)
    if st.button("Weiter", key="next_entity_button"):
        # ... (Keep the saving logic for both context_relevance and other metrics) ...
        if st.session_state.current_metric in ["context_relevance", "multiturn_context_relevance"]:
            num_contexts_expected = len(current_entity_for_display.get('retrieved_contexts', []))
            all_rated = True
            if len(st.session_state.context_ratings) != num_contexts_expected:
                 all_rated = False
            else:
                 for i in range(num_contexts_expected):
                      if st.session_state.context_ratings.get(i) is None:
                           all_rated = False
                           break

            if not all_rated:
                st.warning("Bitte bewerten Sie die Relevanz *aller* angezeigten Kontexte.")
            else:
                try:
                    sample_id_to_save = current_id
                    for index, rating in st.session_state.context_ratings.items():
                        save_rating(
                            sample_id=sample_id_to_save,
                            metric=st.session_state.current_metric,
                            rating=rating,
                            it_background=st.session_state.it_background,
                            is_pairwise=False,
                            swap_options=False,
                            supabase_client=supabase_client,
                            context_index=index
                        )

                    st.session_state.entity_count += 1
                    st.session_state.context_ratings = {} # Clear ratings for the next item

                    # Advance to next entity or end round
                    if st.session_state.entity_count >= st.session_state.num_entities_this_round or not st.session_state.entities_this_round:
                        st.session_state.round_over = True
                        st.session_state.current_sample = None
                        st.session_state.current_sample_pair = None
                        st.session_state.entities_this_round = []
                    else:
                        next_entity = st.session_state.entities_this_round.pop(0)
                        st.session_state.current_sample = next_entity
                        st.session_state.current_sample_pair = None
                        st.session_state.swap_options = False

                    st.rerun()

                except Exception as e:
                     st.error(f"Fehler beim Speichern der Kontext-Bewertungen: {e}")
                     st.session_state.round_over = True # End round on save error
                     st.rerun()

        else: # Handle single/pairwise ratings
            current_rating = st.session_state.user_rating
            if current_rating is None:
                st.warning("Bitte wählen Sie eine Bewertung aus, bevor Sie fortfahren.")
            else:
                try:
                    sample_id_to_save = current_id
                    save_rating(
                        sample_id_to_save,
                        st.session_state.current_metric,
                        current_rating,
                        st.session_state.it_background,
                        st.session_state.is_pairwise,
                        st.session_state.swap_options,
                        supabase_client,
                        context_index=None
                    )
                    st.session_state.entity_count += 1

                    # Advance to next entity or end round
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

                    st.session_state.user_rating = None # Clear rating for the next item
                    st.rerun()

                except KeyError as e:
                     st.error(f"Fehler beim Speichern: Fehlender Schlüssel {e}. Entity ID: {current_id}")
                     st.session_state.round_over = True
                     st.rerun()
                except Exception as e:
                     st.error(f"Fehler beim Speichern der Bewertung: {e}")
                     st.exception(e)
                     st.session_state.round_over = True
                     st.rerun()


if __name__ == "__main__":
    # --- Ensure validation sets are loaded before main() if needed globally ---
    # It seems they are loaded globally already in your original code.
    # If not, load them here or pass them appropriately.
    main()

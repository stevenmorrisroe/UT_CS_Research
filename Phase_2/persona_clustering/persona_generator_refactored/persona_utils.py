# Persona prompt generation and saving functions
import os
import pandas as pd # Needed for aggregating purchases
import calendar # Needed for month names
from collections import Counter
import logging # Add logging import

# Set up logger for this module
logger = logging.getLogger(__name__)

# Import constants
from .constants import N_TOPICS, N_WORDS_FOR_PROMPT, OUTPUT_FILES

# Constants for thresholds (could be moved to constants.py)
VALUE_LOW_THRESHOLD = 0.75
VALUE_HIGH_THRESHOLD = 1.25
VALUE_VERY_HIGH_THRESHOLD = 1.75
FREQ_INFREQUENT_THRESHOLD = 1.5
FREQ_OCCASIONAL_THRESHOLD = 3.5
FREQ_MODERATE_THRESHOLD = 6.0
FREQ_HIGH_THRESHOLD = 10.0 # Added a higher threshold

def _infer_shopping_goal(primary_focus, top_keywords):
    """Helper function to infer shopping goal based on keywords."""
    # Prioritize category keywords
    category_keywords = [kw for kw in top_keywords[:10] if kw.startswith('category_')]
    if category_keywords:
        top_cats = [cat.replace('category_', '').replace('_', ' ') for cat in category_keywords[:2]]
        if len(top_cats) == 1:
            return f"primarily shopping within the **{top_cats[0]}** category"
        elif len(top_cats) == 2:
            return f"primarily shopping within the **{top_cats[0]}** and **{top_cats[1]}** categories"

    # Fallback to keyword heuristics (less reliable)
    if any(kw in primary_focus for kw in ['coffee', 'tea', 'pods']): return "stocking up on beverages"
    if any(kw in primary_focus for kw in ['cat_food', 'dog_food', 'pet_treat']): return "buying supplies for their pets"
    if any(kw in primary_focus for kw in ['organic', 'gluten_free', 'cheese', 'chocolate']): return "shopping for groceries, possibly focusing on specialty items"
    if any(kw in primary_focus for kw in ['usb_c', 'hdmi_cable', 'phone_case', 'power_adapter']): return "looking for electronics accessories"
    if any(kw in primary_focus for kw in ['baby', 'toddler', 'kids', 'diaper']): return "buying items for children or babies"
    if any(kw in primary_focus for kw in ['skin_care', 'shampoo', 'makeup', 'face_wash']): return "shopping for personal care and beauty products"
    if any(kw in primary_focus for kw in ['storage', 'kitchen', 'led', 'decor']): return "purchasing items for home improvement or decor"

    return "shopping online for various goods" # Default fallback

def _get_value_descriptor(topic_id, topic_value_metrics):
    """Helper function to generate a more nuanced value descriptor based on
    average purchase value relative to the overall average.
    """
    if not topic_value_metrics or topic_id not in topic_value_metrics:
        return "Value sensitivity information unavailable." # Return neutral statement

    avg_value = topic_value_metrics[topic_id].get('AverageValuePerPurchase', None)
    overall_avg = topic_value_metrics.get('overall_avg_value', None)

    if avg_value is None or overall_avg is None or overall_avg <= 0:
        return "Value sensitivity could not be determined." # Handle missing or invalid data

    relative_value = avg_value / overall_avg

    if relative_value < VALUE_LOW_THRESHOLD:
        return "Appears **highly price-sensitive**, typically purchasing items significantly below the average value."
    elif relative_value < 1 / VALUE_HIGH_THRESHOLD: # Between low and slightly below average
         return "Tends towards **budget-consciousness**, often purchasing items slightly below the average value."
    elif relative_value <= VALUE_HIGH_THRESHOLD: # Around average
        return "Purchases items around the **average value point**, suggesting balanced value consideration."
    elif relative_value <= VALUE_VERY_HIGH_THRESHOLD: # Moderately high value
        return "Often selects **premium or higher-value items**, spending moderately above the average."
    else: # Significantly high value
        return "Strongly prefers **high-end or luxury items**, with purchases significantly exceeding the average value."

def _format_purchase_frequency(insights):
    """Helper function to format purchase frequency insights into more descriptive
    shopping style text, grounded by average purchase counts.
    """
    avg_purchases = insights.get('avg_purchases_per_customer', 0)
    # max_purchases = insights.get('max_purchases', 0) # Consider using max for high-end description

    if avg_purchases <= FREQ_INFREQUENT_THRESHOLD:
        return "Acts as an **Infrequent Buyer**, making only 1-2 purchases on average, suggesting targeted or needs-based shopping."
    elif avg_purchases <= FREQ_OCCASIONAL_THRESHOLD:
        return "Shops **Occasionally**, averaging 2-3 purchases, possibly for specific events or less frequent needs."
    elif avg_purchases <= FREQ_MODERATE_THRESHOLD:
        return "Is a **Regular Shopper**, making 4-6 purchases on average, indicating consistent engagement or routine buying."
    elif avg_purchases <= FREQ_HIGH_THRESHOLD:
         return "Behaves as a **Frequent Purchaser**, averaging 7-10 purchases, suggesting strong loyalty or diverse needs."
    else: # avg_purchases > FREQ_HIGH_THRESHOLD
        # Consider adding nuance based on max_purchases if available and significant
        return "Is a **Highly Engaged Shopper**, making over 10 purchases on average, demonstrating very high frequency and likely broad interaction."

def _format_purchase_seasonality(insights):
    """Helper function to format purchase seasonality insights into text."""
    monthly_purchases = insights.get('purchase_by_month', {})
    if not monthly_purchases:
        return "" # No seasonality data

    total_purchases = sum(monthly_purchases.values())
    if total_purchases == 0:
        return "" # Avoid division by zero

    # Find peak months (e.g., months contributing > 1.5x the average monthly purchase rate)
    avg_monthly_rate = total_purchases / 12
    peak_months = [
        calendar.month_name[month] for month, count in monthly_purchases.items()
        if count > 1.5 * avg_monthly_rate and count > 1 # Avoid trivial peaks
    ]

    if not peak_months:
        return " Shopping occurs **relatively evenly throughout the year**."
    elif len(peak_months) == 1:
        return f" Show a potential **peak shopping season around {peak_months[0]}**."
    elif len(peak_months) == 2:
        return f" Show potential **peak shopping seasons around {peak_months[0]} and {peak_months[1]}**."
    else:
        # List top 3 for brevity
        top_months = sorted(monthly_purchases, key=monthly_purchases.get, reverse=True)[:3]
        month_names = [calendar.month_name[m] for m in top_months]
        return f" Shopping activity seems **highest around {', '.join(month_names[:-1])}, and {month_names[-1]}**."

def _format_repeat_behavior(insights):
    """Helper function to format repeat purchase behavior into text."""
    pct_repeats = insights.get('pct_customers_with_repeats', 0) * 100
    if pct_repeats == 0:
        return "Appear to primarily make **one-time purchases** of specific items."
    elif pct_repeats < 15:
        return "Show **low repeat purchase behavior** (less than 15% buy the same item again)."
    elif pct_repeats < 40:
        return "Exhibit **moderate repeat purchase behavior** (15-40% buy the same item again)."
    else:
        return "Demonstrate **high loyalty or need for specific items**, with significant repeat purchasing (over 40% buy the same item again)."

def _format_demographics(insights):
    """Helper function to format aggregated demographic insights into text."""
    if not insights:
        return "" # No demographic data available

    parts = []

    # Age (if available)
    age_desc = insights.get('typical_age_range', None)
    if age_desc:
        parts.append(f"Typically falls within the **{age_desc}** age group.")

    # Gender (if available)
    gender_dist = insights.get('gender_distribution', None)
    if gender_dist:
        dominant_gender = max(gender_dist, key=gender_dist.get)
        dominant_pct = gender_dist[dominant_gender] * 100
        if dominant_pct > 60: # Simple heuristic for skew
            parts.append(f"Predominantly identifies as **{dominant_gender}** ({dominant_pct:.0f}%).")
        else:
            gender_str = ", ".join([f"{g}: {p*100:.0f}%" for g, p in gender_dist.items()])
            parts.append(f"Gender distribution includes: **{gender_str}**.")

    # Income (if available)
    income_desc = insights.get('typical_income_range', None)
    if income_desc:
        parts.append(f"Commonly reports an income in the **{income_desc}** range.")

    # Education (if available)
    edu_desc = insights.get('typical_education', None)
    if edu_desc:
        parts.append(f"Most frequently reports education level as **{edu_desc}**.")

    # Location (if available)
    locations = insights.get('top_locations', [])
    if locations:
        if len(locations) == 1:
            parts.append(f"Primarily located in **{locations[0]}**.")
        elif len(locations) <= 3:
            loc_str = ", ".join([f"**{loc}**" for loc in locations[:-1]]) + f" and **{locations[-1]}**"
            parts.append(f"Commonly located in {loc_str}.")
        else: # Show top 3
            loc_str = ", ".join([f"**{loc}**" for loc in locations[:3]])
            parts.append(f"Top locations include {loc_str}, among others.")

    if not parts:
        return " Demographic information is limited for this group."

    # Combine parts into a paragraph-like structure
    return "\n- " + "\n- ".join(parts)

def _clean_and_get_mode(series):
    """Cleans string data and returns the mode (most frequent value), handling NaNs."""
    cleaned_series = series.dropna().astype(str).str.strip()
    if cleaned_series.empty:
        return None
    mode_val = cleaned_series.mode()
    # .mode() can return multiple values if tied; return the first one
    return mode_val[0] if not mode_val.empty else None

def _get_distribution(series):
    """Calculates the value distribution (percentage) for a categorical series."""
    cleaned_series = series.dropna().astype(str).str.strip()
    if cleaned_series.empty:
        return {}
    value_counts = cleaned_series.value_counts(normalize=True)
    # Optional: filter small percentages or limit categories
    return value_counts.to_dict()

def extract_behavioral_signals(topic_id, purchase_df, topic_assignments):
    """
    Extract additional behavioral signals for improved persona development

    Parameters:
    -----------
    topic_id : int
        Topic identifier
    purchase_df : DataFrame
        Purchase level data (needs 'Survey_ResponseID', 'Order Date', 'ASIN_ISBN')
    topic_assignments : DataFrame
        Mapping customers to topics (needs 'Survey_ResponseID', 'Topic')

    Returns:
    --------
    dict : Behavioral insights for this topic
    """
    # Ensure required columns exist
    required_purchase_cols = ['Survey_ResponseID']
    required_assignment_cols = ['Survey_ResponseID', 'Topic']
    optional_purchase_cols = ['Order Date', 'ASIN_ISBN']

    if not all(col in purchase_df.columns for col in required_purchase_cols):
        logger.error(f"purchase_df is missing one or more required columns: {required_purchase_cols}")
        return {}
    if not all(col in topic_assignments.columns for col in required_assignment_cols):
        logger.error(f"topic_assignments is missing one or more required columns: {required_assignment_cols}")
        return {}

    # Get customers in this topic
    topic_customers = topic_assignments[topic_assignments['Topic'] == topic_id]['Survey_ResponseID']

    if topic_customers.empty:
        return { # Return default structure even if no customers
            'avg_purchases_per_customer': 0,
            'max_purchases': 0,
            'purchase_by_month': {},
            'pct_customers_with_repeats': 0
        }

    # Filter purchases to these customers
    topic_purchases = purchase_df[purchase_df['Survey_ResponseID'].isin(topic_customers)].copy() # Use .copy() to avoid SettingWithCopyWarning

    insights = {}

    # Purchase frequency
    if not topic_purchases.empty:
        purchase_counts = topic_purchases.groupby('Survey_ResponseID').size()
        insights['avg_purchases_per_customer'] = purchase_counts.mean() if not purchase_counts.empty else 0
        insights['max_purchases'] = purchase_counts.max() if not purchase_counts.empty else 0
    else:
        insights['avg_purchases_per_customer'] = 0
        insights['max_purchases'] = 0

    # Temporal patterns (if date data available)
    if 'Order Date' in topic_purchases.columns:
        try:
            topic_purchases['Order_Date'] = pd.to_datetime(topic_purchases['Order Date'], errors='coerce')
            topic_purchases.dropna(subset=['Order_Date'], inplace=True)
            insights['purchase_by_month'] = topic_purchases['Order_Date'].dt.month.value_counts().to_dict()
        except Exception as e:
            logger.warning(f"Could not parse Order Date for seasonality: {e}")
            insights['purchase_by_month'] = {}
    else:
        insights['purchase_by_month'] = {}

    # Repeat purchase behavior (if ASIN available)
    if 'ASIN_ISBN' in topic_purchases.columns and not topic_purchases.empty:
        # Count purchases per customer per ASIN
        cust_asin_counts = topic_purchases.groupby(['Survey_ResponseID', 'ASIN_ISBN']).size()
        # Find customers who bought *any* ASIN more than once
        repeating_customers = cust_asin_counts[cust_asin_counts > 1].reset_index()['Survey_ResponseID'].nunique()
        total_customers_in_topic = topic_customers.nunique()
        insights['pct_customers_with_repeats'] = (repeating_customers / total_customers_in_topic) if total_customers_in_topic > 0 else 0
    else:
        insights['pct_customers_with_repeats'] = 0

    return insights

def aggregate_demographics_for_topic(topic_id, topic_assignments, demographics_df):
    """Aggregates demographic information for customers within a specific topic.

    Args:
        topic_id (int): The ID of the topic.
        topic_assignments (pd.DataFrame): DataFrame mapping 'Survey_ResponseID' to 'Topic'.
        demographics_df (pd.DataFrame): DataFrame containing demographic data with 'Survey_ResponseID' and other demographic columns (e.g., 'Age', 'Gender', 'Income').

    Returns:
        dict: Aggregated demographic insights for the topic.
    """
    # Check required columns
    required_assignment_cols = ['Survey_ResponseID', 'Topic']
    if not all(col in topic_assignments.columns for col in required_assignment_cols):
        logger.error(f"topic_assignments is missing one or more required columns: {required_assignment_cols}")
        return {}
    if not demographics_df.empty and 'Survey_ResponseID' not in demographics_df.columns:
        logger.error("demographics_df is missing required column: 'Survey_ResponseID'")
        return {}

    # Get customers in this topic
    topic_customers = topic_assignments[topic_assignments['Topic'] == topic_id]['Survey_ResponseID']

    if topic_customers.empty:
        return {} # No customers, no demographics

    # Filter demographics to these customers
    # Ensure ID types match (convert both to string for safety)
    topic_customers_str = topic_customers.astype(str)
    demographics_df['Survey_ResponseID'] = demographics_df['Survey_ResponseID'].astype(str)
    topic_demographics = demographics_df[demographics_df['Survey_ResponseID'].isin(topic_customers_str)]

    if topic_demographics.empty:
        logger.info(f"No demographic data found for customers in Topic {topic_id}")
        return {}

    insights = {}

    # Calculate typical values (mode) for categorical data
    if 'Age' in topic_demographics.columns:
        insights['typical_age_range'] = _clean_and_get_mode(topic_demographics['Age'])
    if 'Gender' in topic_demographics.columns:
        # Get distribution instead of just mode for gender
        insights['gender_distribution'] = _get_distribution(topic_demographics['Gender'])
    if 'Income' in topic_demographics.columns:
        insights['typical_income_range'] = _clean_and_get_mode(topic_demographics['Income'])
    if 'Education' in topic_demographics.columns:
        insights['typical_education'] = _clean_and_get_mode(topic_demographics['Education'])

    # Calculate top locations (handle potential multiple values per customer if needed)
    if 'State' in topic_demographics.columns:
         # Simple mode for now, assumes one state per respondent in this dataset
         top_locations = topic_demographics['State'].dropna().astype(str).str.strip().value_counts().head(3).index.tolist()
         insights['top_locations'] = top_locations

    # Add other calculations as needed (e.g., race distribution, hispanic percentage)

    return insights

def generate_persona_prompt(topic_id, all_topic_keywords, topic_value_metrics, behavioral_insights, demographic_insights):
    """Generates a detailed text prompt for an LLM to create a persona description.

    Args:
        topic_id (int): The ID of the topic.
        all_topic_keywords (dict): Dictionary mapping topic_id to list of top keywords.
        topic_value_metrics (dict): Dictionary containing average purchase value metrics.
        behavioral_insights (dict): Dictionary containing behavioral signals.
        demographic_insights (dict): Dictionary containing aggregated demographic signals.

    Returns:
        str: The generated system prompt.
    """
    if topic_id not in all_topic_keywords:
        logger.warning(f"Keywords not found for topic {topic_id}. Cannot generate prompt.")
        return ""

    keywords = all_topic_keywords[topic_id]
    primary_focus = ", ".join(keywords[:5])
    secondary_focus = ", ".join(keywords[5:N_WORDS_FOR_PROMPT])

    inferred_goal = _infer_shopping_goal(primary_focus, keywords)
    value_descriptor = _get_value_descriptor(topic_id, topic_value_metrics)

    # Format behavioral insights using refined functions
    frequency_desc = _format_purchase_frequency(behavioral_insights)
    seasonality_desc = _format_purchase_seasonality(behavioral_insights)
    repeat_desc = _format_repeat_behavior(behavioral_insights)

    # Format demographic insights
    demographic_desc = _format_demographics(demographic_insights)

    # Construct the prompt
    prompt = f"""Create a descriptive persona based on the following data for Topic {topic_id}.

**Topic Core:**
- Primary Keywords: {primary_focus}
- Secondary Keywords: {secondary_focus}
- Inferred Goal: {inferred_goal}

**Value Sensitivity:**
- {value_descriptor}

**Behavioral Signals:**
- Shopping Style: {frequency_desc} {seasonality_desc}
- Item Loyalty: {repeat_desc}

**Demographics (if available):** {demographic_desc}

**Instructions:**
Synthesize the information above into a concise, narrative persona description. Give the persona a plausible name. Describe their likely motivations, shopping habits, lifestyle, and attitude based *only* on the provided data. Avoid making assumptions beyond the data. Focus on what the data tells us about their online purchasing behavior and related characteristics.
"""
    return prompt

def aggregate_top_purchases_by_frequency(customer_topics_df, original_purchase_df, top_n=100):
    """Aggregates top N purchased items (ASIN & Title) by frequency for each topic.

    Args:
        customer_topics_df (pd.DataFrame): DataFrame with 'Survey_ResponseID' and 'Topic'.
        original_purchase_df (pd.DataFrame): Original DataFrame with all purchase records,
                                            including 'Survey_ResponseID', 'ASIN_ISBN', 'Title'.
        top_n (int): The number of top items to return per topic.

    Returns:
        dict: Dictionary mapping topic_id (int) to a DataFrame with columns
              ['Rank', 'ASIN_ISBN', 'Title', 'Frequency'], sorted by Frequency.
              Returns an empty DataFrame for topics with no customers or purchases.
    """
    logger.info(f"Aggregating top {top_n} purchased items per topic...")
    top_purchases_per_topic = {}

    # Check for necessary data
    if customer_topics_df is None or original_purchase_df is None:
        logger.warning("Missing customer topics or purchase data for aggregation. Returning empty results.")
        # Return structure with empty dataframes
        for topic_id in range(N_TOPICS): # Assuming N_TOPICS is accessible or defined elsewhere
            top_purchases_per_topic[topic_id] = pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency'])
        return top_purchases_per_topic

    # Ensure required columns exist
    if not all(col in original_purchase_df.columns for col in ['Survey_ResponseID', 'ASIN_ISBN', 'Title']):
        logger.error("original_purchase_df missing required columns ('Survey_ResponseID', 'ASIN_ISBN', 'Title'). Cannot aggregate.")
        # Return structure with empty dataframes
        for topic_id in range(N_TOPICS): # Assuming N_TOPICS is accessible or defined elsewhere
            top_purchases_per_topic[topic_id] = pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency'])
        return top_purchases_per_topic

    try:
        # Ensure ID types match for merging (convert both to string for safety)
        customer_topics_df['Survey_ResponseID'] = customer_topics_df['Survey_ResponseID'].astype(str)
        original_purchase_df['Survey_ResponseID'] = original_purchase_df['Survey_ResponseID'].astype(str)

        # Merge topic assignments with purchase data
        purchase_df_with_topics = pd.merge(
            original_purchase_df[['Survey_ResponseID', 'ASIN_ISBN', 'Title']],
            customer_topics_df,
            on='Survey_ResponseID',
            how='inner' # Only include purchases from customers assigned a topic
        )

        if purchase_df_with_topics.empty:
            logger.warning("No purchases found matching assigned topics. Cannot aggregate.")
            for topic_id in range(N_TOPICS):
                top_purchases_per_topic[topic_id] = pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency'])
            return top_purchases_per_topic

        # Group by topic, then count frequency of ASIN/Title pairs
        grouped = purchase_df_with_topics.groupby('Topic')

        for topic_id, group in grouped:
            # Count frequency of each ASIN/Title combination within the topic
            # Ensure ASIN and Title are not NaN before grouping
            group = group.dropna(subset=['ASIN_ISBN', 'Title'])
            if group.empty:
                top_purchases_per_topic[topic_id] = pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency'])
                continue

            item_counts = group.groupby(['ASIN_ISBN', 'Title']).size().reset_index(name='Frequency')

            # Sort by frequency and take top N
            top_items = item_counts.sort_values(by='Frequency', ascending=False).head(top_n)

            # Add rank
            top_items['Rank'] = range(1, len(top_items) + 1)

            # Select and order columns
            top_purchases_per_topic[topic_id] = top_items[['Rank', 'ASIN_ISBN', 'Title', 'Frequency']]

        # Fill in empty DFs for topics with no purchases
        all_topics = set(range(N_TOPICS)) # Assuming N_TOPICS covers all expected topic IDs
        topics_with_data = set(top_purchases_per_topic.keys())
        for topic_id in all_topics - topics_with_data:
            logger.info(f"No purchases found for Topic {topic_id}, creating empty top purchases list.")
            top_purchases_per_topic[topic_id] = pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency'])

        logger.info(f"Finished aggregating top {top_n} purchases for {len(top_purchases_per_topic)} topics.")

    except Exception as e:
        logger.error(f"Error aggregating top purchases: {e}")
        # Return structure with empty dataframes in case of error
        for topic_id in range(N_TOPICS):
            top_purchases_per_topic[topic_id] = pd.DataFrame(columns=['Rank', 'ASIN_ISBN', 'Title', 'Frequency'])

    return top_purchases_per_topic

def save_personas_and_purchases(personas, aggregated_top_purchases, output_dir):
    """Saves generated persona prompts and top purchases to files.

    Args:
        personas (dict): Dictionary mapping topic_id to persona prompt string.
        aggregated_top_purchases (dict): Dictionary mapping topic_id to DataFrame
                                        of top purchases (Rank, ASIN_ISBN, Title, Frequency).
        output_dir (str): Directory to save the files.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving persona prompts and top purchases to: {output_dir}")

    # Save persona prompts
    if personas:
        for topic_id, prompt in personas.items():
            filename = OUTPUT_FILES["persona_prompt_template"].format(topic_id)
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(prompt)
            except Exception as e:
                logger.error(f"Error saving persona prompt for topic {topic_id} to {filepath}: {e}")
        logger.info(f"Saved {len(personas)} persona prompts.")
    else:
        logger.warning("No persona prompts generated to save.")

    # Save aggregated top purchases
    if aggregated_top_purchases:
        saved_count = 0
        for topic_id, df in aggregated_top_purchases.items():
            filename = OUTPUT_FILES["persona_top_purchases_template"].format(topic_id)
            filepath = os.path.join(output_dir, filename)
            try:
                if not df.empty:
                    df.to_csv(filepath, index=False)
                    saved_count += 1
                else:
                    # Optionally, create an empty file or log that it was empty
                    # open(filepath, 'a').close() # Create empty file
                    logger.info(f"Skipping save for Topic {topic_id} as top purchases list is empty.")
            except Exception as e:
                logger.error(f"Error saving top purchases for topic {topic_id} to {filepath}: {e}")
        logger.info(f"Saved top purchase lists for {saved_count} topics.")
    else:
        logger.warning("No aggregated purchases found to save.") 
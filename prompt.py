"""
This module provides functions for generating structured prompts for financial analysis tasks.
It includes functions to:
- Create introductory prompts for companies and cryptocurrencies.
- Format weekly data (price movements, news, basics) into prompt segments.
- Sample news items randomly.
- Map numerical bin labels to descriptive text.
- Assemble complete prompts by combining historical data, company/crypto info,
  and a specific task instruction for an LLM.
"""
import os
import json
import random
import finnhub
import yfinance as yf # Used for fetching crypto info via yf.Ticker
import pandas as pd
from indices import CRYPTO

# --- API Key Configuration ---
FINNHUB_API_KEY = "d0per69r01qr8ds39n80d0per69r01qr8ds39n8g"

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def get_company_prompt(symbol):
    """
    Generates an introductory text snippet for a given company using Finnhub data.

    Args:
        symbol (str): The stock ticker symbol.

    Returns:
        str: A formatted string containing company introduction details.
             Returns an error message string if profile fetching fails.
    """
    try:
        profile = finnhub_client.company_profile2(symbol=symbol)
        if not profile: # Check if profile is empty or None
            return f"[Company Introduction]:\n\nInformation for {symbol} could not be retrieved or is not available."
    except Exception as e:
        print(f"Error fetching company profile for {symbol}: {e}")
        return f"[Company Introduction]:\n\nError retrieving information for {symbol}."

    # Template for company introduction
    company_template = (
        "[Company Introduction]:\n\n"
        "{name} is a leading entity in the {finnhubIndustry} sector. "
        "Incorporated and publicly traded since {ipo}, the company has established "
        "its reputation as one of the key players in the market. As of today, "
        "{name} has a market capitalization of {marketCapitalization:.2f} in {currency}, "
        "with {shareOutstanding:.2f} shares outstanding.\n\n"
        "{name} operates primarily in the {country}, trading under the ticker {ticker} "
        "on the {exchange}. As a dominant force in the {finnhubIndustry} space, "
        "the company continues to innovate and drive progress within the industry."
    )

    # Fill the template with profile data. Use .get() for safer access to dict keys.
    formatted_str = company_template.format(
        name=profile.get('name', symbol), # Fallback to symbol if name is missing
        finnhubIndustry=profile.get('finnhubIndustry', 'N/A'),
        ipo=profile.get('ipo', 'N/A'),
        marketCapitalization=profile.get('marketCapitalization', 0.0),
        currency=profile.get('currency', 'USD'),
        shareOutstanding=profile.get('shareOutstanding', 0.0),
        country=profile.get('country', 'N/A'),
        ticker=profile.get('ticker', symbol),
        exchange=profile.get('exchange', 'N/A')
    )
    
    return formatted_str


def get_crypto_prompt(symbol):
    """
    Generates an introductory text snippet for a given cryptocurrency using yfinance.

    Args:
        symbol (str): The cryptocurrency ticker symbol (e.g., "BTC-USD").

    Returns:
        str: A formatted string containing cryptocurrency introduction details.
             Returns an error message string if info fetching fails.
    """
    try:
        ticker_info = yf.Ticker(symbol).info
        if not ticker_info or 'description' not in ticker_info or 'marketCap' not in ticker_info:
             return f"[Cryptocurrency Introduction]: Description or market cap for {symbol} is not available via yfinance."
    except Exception as e:
        print(f"Error fetching crypto info for {symbol}: {e}")
        return f"[Cryptocurrency Introduction]: Error retrieving information for {symbol}."

    # Template for crypto introduction
    # Using .get for safer access to dictionary keys from yf.Ticker().info
    crypto_template = "[Cryptocurrency Introduction]: {description} It has a market capitalization of {marketCap}."
    
    formatted_str = crypto_template.format(
        description=ticker_info.get('description', 'No description available.'),
        marketCap=ticker_info.get('marketCap', 'N/A')
    )
    
    return formatted_str


def get_prompt_by_row(symbol, row):
    """
    Formats a single week's data (price movement, news, basics) into a prompt segment for companies.

    Args:
        symbol (str): The stock ticker symbol.
        row (pd.Series): A row from a DataFrame containing weekly data 
                         ('Start Date', 'End Date', 'Start Price', 'End Price', 'News', 'Basics').

    Returns:
        tuple: (str, list, str)
            - head_str (str): Text describing price movement for the week.
            - news_list (list): A list of formatted news strings for the week.
            - basics_str (str): Formatted string of basic financials for the week.
    """
    # Ensure dates are strings
    start_date_str = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date_str = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    
    # Determine if price increased or decreased
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    
    # Create the header string describing price movement
    head_str = (
        f"From {start_date_str} to {end_date_str}, {symbol}'s stock price {term} "
        f"from {row['Start Price']:.2f} to {row['End Price']:.2f}. "
        f"News during this period are listed below:\n\n"
    )
    
    # Load and filter news
    try:
        news_data = json.loads(row["News"]) # News is stored as a JSON string
        news_list_formatted = [
            f"[Headline]: {n.get('headline', 'N/A')}\n[Summary]: {n.get('summary', 'N/A')}\n"
            for n in news_data 
            if n.get('date', '99999999')[:8] <= end_date_str.replace('-', '') and \
               not str(n.get('summary', '')).startswith("Looking for stock market analysis and research with proves results?") # Filter out promotional content
        ]
    except (json.JSONDecodeError, TypeError):
        news_list_formatted = ["No valid news data found for this period."]

    # Load and format basic financials
    try:
        basics_data = json.loads(row['Basics']) # Basics stored as JSON string
        if basics_data and basics_data.get('period'): # Check if basics data is not empty and has a period
            basics_str = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
                symbol, basics_data['period']) + "\n".join(f"{k}: {v}" for k, v in basics_data.items() if k != 'period')
        else:
            basics_str = "[Basic Financials]:\n\nNo basic financial reported for this period."
    except (json.JSONDecodeError, TypeError):
        basics_str = "[Basic Financials]:\n\nError loading basic financials for this period."
            
    return head_str, news_list_formatted, basics_str


def get_crypto_prompt_by_row(symbol, row):
    """
    Formats a single week's data (price movement, news) into a prompt segment for cryptocurrencies.
    Similar to get_prompt_by_row but without basic financials.

    Args:
        symbol (str): The cryptocurrency ticker symbol.
        row (pd.Series): A row from a DataFrame containing weekly data.

    Returns:
        tuple: (str, list, None)
            - head_str (str): Text describing price movement.
            - news_list (list): A list of formatted news strings.
            - None: Placeholder for basics, as cryptos don't have them in this context.
    """
    start_date_str = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date_str = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    
    head_str = (
        f"From {start_date_str} to {end_date_str}, {symbol}'s price {term} "
        f"from {row['Start Price']:.2f} to {row['End Price']:.2f}. "
        f"News during this period are listed below:\n\n"
    )
    
    try:
        news_data = json.loads(row["News"])
        news_list_formatted = [
            f"[Headline]: {n.get('headline', 'N/A')}\n[Summary]: {n.get('summary', 'N/A')}\n"
            for n in news_data
            if n.get('date', '99999999')[:8] <= end_date_str.replace('-', '') and \
               not str(n.get('summary', '')).startswith("Looking for stock market analysis and research with proves results?")
        ]
    except (json.JSONDecodeError, TypeError):
        news_list_formatted = ["No valid news data found for this period."]

    return head_str, news_list_formatted, None # No basics for crypto in this function


def sample_news(news_list, k=5):
    """
    Randomly samples k news items from a list of news.
    If the list has fewer than k items, all items are returned.

    Args:
        news_list (list): A list of news items (strings).
        k (int): The number of news items to sample.

    Returns:
        list: A list containing k (or fewer) sampled news items.
    """
    if not news_list:
        return []
    if len(news_list) <= k:
        return news_list # Return all if k is larger or equal to list size
    return [news_list[i] for i in sorted(random.sample(range(len(news_list)), k))]


def map_bin_label(bin_label_str):
    """
    Converts a compact bin label (e.g., "U1", "D5+") into a more descriptive string.

    Args:
        bin_label_str (str): The compact bin label.

    Returns:
        str: A descriptive string for the bin label (e.g., "up by 0-1%", "down by more than 5%").
    """
    if not isinstance(bin_label_str, str): # Basic type check
        return "Invalid label"

    lb = bin_label_str.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    
    # Specific replacements for numerical parts
    if '1' in lb and not '1-' in lb : lb = lb.replace('1', '0-1%') # Avoid replacing '1' in '1-2%'
    if '2' in lb and not '2-' in lb : lb = lb.replace('2', '1-2%')
    if '3' in lb and not '3-' in lb : lb = lb.replace('3', '2-3%')
    if '4' in lb and not '4-' in lb : lb = lb.replace('4', '3-4%')
    
    if lb.endswith('+'): # Handles "5+"
        lb = lb.replace('5+', 'more than 5%')
    elif '5' in lb : # Handles "5" (without +)
         lb = lb.replace('5', '4-5%')
    
    return lb

# Templates for the final part of the prompt, instructing the LLM on the task.
PROMPT_END = {
    "company": (
        "\n\nBased on all the information before {start_date}, let's first analyze the positive "
        "developments and potential concerns for {symbol}. Come up with 2-4 most important factors "
        "respectively and keep them concise. Most factors should be inferred from company related news. "
        "Then let's assume your prediction for next week ({start_date} to {end_date}) is {prediction}. "
        "Provide a summary analysis to support your prediction. The prediction result need to be inferred "
        "from your analysis at the end, and thus not appearing as a foundational factor of your analysis."
    ),
    "crypto": (
        "\n\nBased on all the information before {start_date}, let's first analyze the positive "
        "developments and potential concerns for {symbol}. Come up with 2-4 most important factors "
        "respectively and keep them concise. Most factors should be inferred from cryptocurrencies related news. "
        "Then let's assume your prediction for next week ({start_date} to {end_date}) is {prediction}. "
        "Provide a summary analysis to support your prediction. The prediction result need to be inferred "
        "from your analysis at the end, and thus not appearing as a foundational factor of your analysis."
    )
}

def get_all_prompts(symbol, data_dir, start_date_overall, end_date_overall, min_past_weeks=1, max_past_weeks=3, with_basics=True):
    """
    Generates a list of comprehensive prompts for a given symbol by combining historical
    data (prices, news, basics) and a task-specific instruction.

    Args:
        symbol (str): The stock or crypto ticker symbol.
        data_dir (str): Directory where the symbol's processed data CSV is located.
        start_date_overall (str): The overall start date for the data period (used for filename).
        end_date_overall (str): The overall end date for the data period (used for filename).
        min_past_weeks (int): Minimum number of past weeks of data to include in each prompt.
        max_past_weeks (int): Maximum number of past weeks of data to include in each prompt.
        with_basics (bool): Whether to load data that includes basic financials (affects filename).

    Returns:
        list: A list of fully formatted prompt strings.
    """
    # Determine the input CSV filename
    filename_suffix = ".csv"
    if not with_basics:
        filename_suffix = "_nobasics.csv"
    
    csv_file_path = os.path.join(data_dir, f'{symbol}_{start_date_overall}_{end_date_overall}{filename_suffix}')
    
    try:
        df_symbol_data = pd.read_csv(csv_file_path)
        # Convert date strings to datetime objects if they are not already (essential for strftime later)
        if 'Start Date' in df_symbol_data.columns:
            df_symbol_data['Start Date'] = pd.to_datetime(df_symbol_data['Start Date'])
        if 'End Date' in df_symbol_data.columns:
            df_symbol_data['End Date'] = pd.to_datetime(df_symbol_data['End Date'])

    except FileNotFoundError:
        print(f"Data file not found for {symbol} at {csv_file_path}. Cannot generate prompts.")
        return []
    
    # Get the appropriate introductory prompt (company or crypto)
    if symbol in CRYPTO: # Assumes CRYPTO is imported from indices.py
        info_intro_prompt = get_crypto_prompt(symbol)
    else:
        info_intro_prompt = get_company_prompt(symbol)

    historical_rows_buffer = [] # Stores (head, news_list, basics_str) tuples for past weeks
    all_generated_prompts = []

    # Iterate through each week's data for the symbol
    for _, current_row_data in df_symbol_data.iterrows():
        current_prompt_historical_section = "" # Accumulates historical data for the current prompt
        
        # If enough historical weeks are buffered, select a random number of them to include
        if len(historical_rows_buffer) >= min_past_weeks:
            # Number of past weeks to include, chosen randomly between min_past_weeks and max_past_weeks (inclusive)
            # but not more than the number of available historical weeks.
            num_weeks_to_include = min(random.choice(range(min_past_weeks, max_past_weeks + 1)), len(historical_rows_buffer))
            
            # Select the most recent `num_weeks_to_include` from the buffer
            selected_past_weeks_data = historical_rows_buffer[-num_weeks_to_include:]
            
            for past_head, past_news_list, _ in selected_past_weeks_data: # Basics from past weeks not directly used in prompt here
                current_prompt_historical_section += "\n" + past_head # Price movement part
                
                # Sample and add news from that past week
                sampled_news_items = sample_news(past_news_list, min(5, len(past_news_list)))
                if sampled_news_items:
                    current_prompt_historical_section += "\n".join(sampled_news_items)
                else:
                    current_prompt_historical_section += "No relevant news reported for this period."

        # Process the current week's data to be added to the buffer for future prompts
        if symbol in CRYPTO:
            current_week_head, current_week_news_list, current_week_basics_str = get_crypto_prompt_by_row(symbol, current_row_data)
        else:
            current_week_head, current_week_news_list, current_week_basics_str = get_prompt_by_row(symbol, current_row_data)

        # Add current week's processed data to the buffer
        historical_rows_buffer.append((current_week_head, current_week_news_list, current_week_basics_str))
        # Maintain the buffer size
        if len(historical_rows_buffer) > max_past_weeks:
            historical_rows_buffer.pop(0)  # Remove the oldest week

        # Only generate a full prompt if we have historical data to include
        if not current_prompt_historical_section:
            continue # Skip if no historical section was built (e.g., first few weeks)

        # Get the prediction label for the current week (this is what the LLM is "told to assume")
        # The actual target for fine-tuning would be the LLM's generated analysis and this label.
        target_prediction_text = map_bin_label(current_row_data['Bin Label'])
        
        # Combine: Intro + Historical Data selected + Current Week's Basics + Task Instruction
        # Note: current_week_basics_str contains the basics relevant *for the week being predicted*.
        final_prompt = info_intro_prompt + '\n' + current_prompt_historical_section + '\n' + current_week_basics_str

        # Add the task-specific ending to the prompt
        prompt_type = 'crypto' if symbol in CRYPTO else 'company'
        final_prompt += PROMPT_END[prompt_type].format(
            start_date=current_row_data['Start Date'].strftime('%Y-%m-%d'), # Week for which prediction is made
            end_date=current_row_data['End Date'].strftime('%Y-%m-%d'),
            prediction=target_prediction_text,
            symbol=symbol
        )

        all_generated_prompts.append(final_prompt.strip())
    
    return all_generated_prompts
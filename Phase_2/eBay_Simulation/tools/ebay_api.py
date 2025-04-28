import os
import requests
import re # Import regex module
import html # Import html module
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
project_root = Path(__file__).parents[1]  # Go up one level to reach project root
env_path = project_root / '.env'
load_dotenv(env_path)

class EbayAPI:
    """
    eBay Browse API client for searching items.
    """
    
    # Define URLs for both environments
    PRODUCTION_BASE_URL = "https://api.ebay.com/buy/browse/v1"
    PRODUCTION_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SANDBOX_BASE_URL = "https://api.sandbox.ebay.com/buy/browse/v1"
    SANDBOX_TOKEN_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
    
    def __init__(self, use_sandbox=None):
        """
        Initialize the eBay API client.
        
        Args:
            use_sandbox: Override to force sandbox (True) or production (False).
                         If None, uses EBAY_USE_SANDBOX environment variable.
        """
        self.client_id = os.getenv("EBAY_CLIENT_ID")
        self.client_secret = os.getenv("EBAY_CLIENT_SECRET")
        self.access_token = None
        self.token_expiry = None
        
        # Determine if we should use sandbox or production
        if use_sandbox is None:
            # Check environment variable, default to production (False)
            self.use_sandbox = os.getenv("EBAY_USE_SANDBOX", "").lower() in ("true", "1", "yes")
        else:
            self.use_sandbox = use_sandbox
            
        # Set the appropriate URLs based on environment
        if self.use_sandbox:
            self.BASE_URL = self.SANDBOX_BASE_URL
            self.TOKEN_URL = self.SANDBOX_TOKEN_URL
            print("Using eBay Sandbox environment")
        else:
            self.BASE_URL = self.PRODUCTION_BASE_URL
            self.TOKEN_URL = self.PRODUCTION_TOKEN_URL
            print("Using eBay Production environment")
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "eBay API credentials not found. Please set EBAY_CLIENT_ID and "
                "EBAY_CLIENT_SECRET in your environment or .env file."
            )
    
    def _get_access_token(self) -> str:
        """
        Get OAuth access token using client credentials flow.
        Caches the token until it expires.
        """
        if (
            self.access_token 
            and self.token_expiry 
            and datetime.now() < self.token_expiry
        ):
            return self.access_token
            
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        
        data = {
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope"
        }
        
        response = requests.post(
            self.TOKEN_URL,
            headers=headers,
            data=data,
            auth=(self.client_id, self.client_secret)
        )
        
        response.raise_for_status()
        token_data = response.json()
        
        self.access_token = token_data["access_token"]
        self.token_expiry = datetime.now() + timedelta(
            seconds=token_data["expires_in"]
        )
        
        return self.access_token
    
    def search_items(
        self,
        query: str,
        limit: int = 10,
        category_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Search for items on eBay using the Browse API.
        
        Args:
            query: Search query string
            limit: Maximum number of items to return (default 10)
            category_ids: Optional list of category IDs to filter by
            filters: Optional dictionary of additional filters
            
        Returns:
            Dict containing search results with item summaries
        """
        access_token = self._get_access_token()
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",  # Default to US marketplace
            "Content-Type": "application/json"
        }
        
        params = {
            "q": query,
            "limit": limit
        }
        
        if category_ids:
            params["category_ids"] = ",".join(category_ids)
            
        if filters:
            filter_str = []
            for key, value in filters.items():
                filter_str.append(f"{key}:{value}")
            params["filter"] = ",".join(filter_str)
            
        response = requests.get(
            f"{self.BASE_URL}/item_summary/search",
            headers=headers,
            params=params
        )
        
        response.raise_for_status()
        return response.json()

    def get_item_details(self, item_id: str) -> Dict:
        """
        Get detailed information about a specific eBay item.
        
        Args:
            item_id: The eBay item ID
            
        Returns:
            Dict containing detailed item information
        """
        access_token = self._get_access_token()
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{self.BASE_URL}/item/{item_id}",
            headers=headers
        )
        
        response.raise_for_status()
        return response.json()

def search_ebay(query: str, min_price: Optional[float] = None, max_price: Optional[float] = None, use_sandbox: Optional[bool] = None) -> str:
    """
    Agent tool function to search eBay and find matching items. 
    Returns item details INCLUDING ITEM IDs which are required for the answer_item_question tool.
    
    Args:
        query: Search query string
        min_price: Optional minimum price to search for
        max_price: Optional maximum price to search for
        use_sandbox: Override to force sandbox (True) or production (False).
                     If None, uses EBAY_USE_SANDBOX environment variable.
        
    Returns:
        str: A formatted list of search results with item details and their IDs.
             Use these Item IDs with the answer_item_question tool to answer buyer questions.
    """
    try:
        api = EbayAPI(use_sandbox=use_sandbox)
        
        # Construct price filter if price range is provided
        filters = {}
        if min_price is not None or max_price is not None:
            price_filter = []
            if min_price is not None:
                price_filter.append(str(min_price))
            else:
                price_filter.append("")
            
            if max_price is not None:
                price_filter.append(str(max_price))
            
            filters["price"] = "[" + "..".join(price_filter) + "]"
        
        # Get up to 5 items to provide a better overview
        results = api.search_items(query, limit=5, filters=filters)
        items = results.get("itemSummaries", [])
        
        if not items:
            if min_price is not None or max_price is not None:
                price_range = ""
                if min_price is not None and max_price is not None:
                    price_range = f" between ${min_price} and ${max_price}"
                elif min_price is not None:
                    price_range = f" above ${min_price}"
                elif max_price is not None:
                    price_range = f" below ${max_price}"
                return f"No items found matching '{query}'{price_range}."
            return f"No items found matching '{query}'."
        
        # Construct response with item details including item IDs
        response_parts = [f"Found {len(items)} items matching '{query}':"]
        
        for item in items:
            title = item.get("title", "Untitled")
            price = item.get("price", {}).get("value", "N/A")
            currency = item.get("price", {}).get("currency", "USD")
            condition = item.get("condition", "N/A")
            item_id = item.get("itemId", "Unknown ID")
            
            response_parts.append(
                f"- {title} ({condition}): {currency} {price} [Item ID: {item_id}]"
            )
        
        response_parts.append("\nNOTE: To answer buyer questions about a specific item, use the answer_item_question tool with the Item ID from the list above.")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"Error searching eBay: {str(e)}"

def answer_item_question(item_id: str) -> str:
    """
    Retrieves details for a specific eBay item ID and returns a standardized summary.
    Use this when a buyer asks about a specific item mentioned previously.

    Args:
        item_id: The eBay Item ID (obtained from 'search_ebay' results).

    Returns:
        str: A markdown formatted summary of the item's details, or an error message.
             Includes a note about the completeness of the information based on the API response.
    """
    conclusion_message = "\n\n*This summary includes the main details available from the API.*"
    try:
        # Instantiate API (uses environment variable for sandbox/prod)
        api = EbayAPI()

        # Get the item details
        item_details = api.get_item_details(item_id)

        # Extract relevant information safely using .get()
        title = item_details.get("title", "N/A")
        price_data = item_details.get("price", {})
        price = price_data.get("value", "N/A")
        currency = price_data.get("currency", "")
        condition = item_details.get("condition", "N/A")
        short_description = item_details.get("shortDescription", "").strip()
        full_description_html = item_details.get("description", "").strip() # Get raw description (potentially HTML)
        # Extract primary image URL
        image_data = item_details.get("image")
        primary_image_url = image_data.get("imageUrl", "N/A") if image_data else "N/A"
        
        # Clean HTML tags and decode entities from full description
        if full_description_html:
            text_description = re.sub(r'<[^>]+>', '', full_description_html) # Strip HTML tags
            text_description = html.unescape(text_description).strip() # Decode HTML entities and strip whitespace
        else:
            text_description = "" # Ensure it's an empty string if no description

        seller_data = item_details.get("seller", {})
        seller = seller_data.get("username", "N/A")
        item_url = item_details.get("itemWebUrl", "N/A")
        location_data = item_details.get("itemLocation", {})
        location = f"{location_data.get('city', '')}, {location_data.get('stateOrProvince', '')}, {location_data.get('country', 'N/A')}".strip(', ')
        if location == "N/A": location = "N/A" # Clean up if only N/A
        
        # Get item specifics (often includes dimensions, material, etc.)
        item_specifics = item_details.get("localizedAspects", [])

        shipping_options = item_details.get("shippingOptions", [])
        return_policy = item_details.get("returnTerms", {})

        # --- Build Markdown Summary ---
        summary = f"**Summary for Item ID: {item_id}**\n"
        summary += f"- **Title:** {title}\n"
        summary += f"- **Price:** {currency} {price}\n"
        summary += f"- **Condition:** {condition}\n"
        summary += f"- **Image URL:** {primary_image_url}\n"
        if short_description:
            # summary += f"- **Short Description:** {short_description}\\n" # Removed as requested
            pass # Keep short description check for potential future use, but don't add to summary
        if text_description:
            # Truncate the cleaned text description to 500 chars
            truncated_description = (text_description[:500] + '...') if len(text_description) > 500 else text_description
            summary += f"- **Full Description (Excerpt):** {truncated_description}\\n" # Add truncated CLEANED description (500 chars)
        # summary += f"- **Seller:** {seller}\\n" # Removed as requested
        # summary += f"- **Location:** {location}\\n" # Removed as requested

        # Item Specifics (Focus on dimensions)
        dimension_specifics = []
        dimension_keywords = ["dimension", "size", "height", "width", "depth", "length"]
        if item_specifics:
            for specific in item_specifics:
                name = specific.get("name", "").lower()
                value = specific.get("value", "")
                # Check if the name contains any dimension keywords
                if any(keyword in name for keyword in dimension_keywords):
                    dimension_specifics.append(f"- {specific.get('name', 'N/A')}: {value}")
            
        if dimension_specifics:
            summary += "**Specifications (Dimensions):**\\n"
            summary += "\\n".join(dimension_specifics) + "\\n"

        # Shipping Summary
        if shipping_options:
            primary_shipping = shipping_options[0]
            ship_cost_data = primary_shipping.get("shippingCost", {})
            ship_cost = ship_cost_data.get("value", "N/A")
            ship_curr = ship_cost_data.get("currency", currency)
            ship_type = primary_shipping.get("shippingServiceCode", "Standard")
            summary += f"- **Shipping (Primary):** {ship_type} - {ship_curr} {ship_cost}\\n"
        else:
            summary += "- **Shipping:** Info not available\\n"

        # Return Policy Summary
        if return_policy:
            accepted = return_policy.get("returnsAccepted", False)
            period_val = return_policy.get("returnPeriod", {}).get("value")
            period_unit = return_policy.get("returnPeriod", {}).get("unit", "days").lower()
            if accepted and period_val:
                summary += f"- **Returns:** Accepted within {period_val} {period_unit}\\n"
            elif accepted:
                 summary += f"- **Returns:** Accepted (period unspecified)\\n"
            else:
                summary += "- **Returns:** Not Accepted\\n"
        else:
            summary += "- **Returns:** Policy not specified\\n"

        return summary + conclusion_message

    except requests.exceptions.HTTPError as http_err:
        # Handle specific API errors (like 404 Not Found)
        status_code = http_err.response.status_code
        if status_code == 404:
            return f"Error: Item with ID '{item_id}' not found. Please check the Item ID."
        else:
            return f"Error retrieving item details: The eBay API returned an error (Status Code: {status_code}). Please try again later."
    except Exception as e:
        # Catch other potential errors (network issues, parsing errors, etc.)
        print(f"Error in answer_item_question for ID {item_id}: {type(e).__name__} - {e}")
        return f"Error retrieving item details: An unexpected error occurred ({type(e).__name__}). Please try again."
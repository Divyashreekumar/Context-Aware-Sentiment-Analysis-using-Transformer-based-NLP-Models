import requests
import json
def get_comments_from_response(response):
    return [review["review_text"] for review in response["reviews"]]
def get_product_review(url,cookie,api,requested_review_count):
    base_url = 'https://data.unwrangle.com/api/getter/'
    params = {
        'platform': 'amazon_reviews',
        'url': url,
        'api_key': api,
        "page":1,
        'cookie': cookie
    }

    # Construct the URL with parameters
    url = f"{base_url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"
    first_response = requests.get(url)
    first_response = first_response.json()
    if first_response["success"] == False:
        return Exception("API failed to fetch the data")
    comments = get_comments_from_response(first_response)
    
    total_results = first_response["total_results"]
    result_count = first_response["result_count"]
    requested_review_count = requested_review_count - result_count
    while requested_review_count > 0 and result_count < total_results:
        params = {
            'platform': 'amazon_reviews',
            'url': url,
            'api_key': api,
            "page":2,
            'cookie': cookie
        }
        url = f"{base_url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"
        response = requests.get(url)
        response = response.json()

        comments.extend(get_comments_from_response(response))
        if response["success"] == False:
            return Exception("API failed to fetch the data")
        requested_review_count = requested_review_count - response["result_count"]  
    return comments  
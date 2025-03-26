import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

def print_response(response):
    """Pretty print API response"""
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print_response(response)

def test_explore_data():
    """Test data exploration"""
    print("\n=== Testing Data Exploration ===")
    response = requests.get(f"{BASE_URL}/explore")
    print_response(response)

def test_mongodb_query():
    """Test direct MongoDB query"""
    print("\n=== Testing MongoDB Query ===")
    query_data = {
        "collection": "smallCollection",  # Replace with your actual collection name
        "query": {},
        "limit": 5
    }
    response = requests.post(
        f"{BASE_URL}/mongodb/query",
        json=query_data
    )
    print_response(response)

def test_analyze_data():
    """Test data analysis"""
    print("\n=== Testing Data Analysis ===")
    analysis_data = {
        "query": "Show me the total profit in the North region",
        "streaming": True
    }
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/analyze",
        json=analysis_data
    )
    end_time = time.time()
    
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
    print("\nInsights:")
    result = response.json()
    if "insights" in result:
        print(result["insights"])
    else:
        print_response(response)

def test_vector_db_indexing():
    """Test vector DB indexing"""
    print("\n=== Testing Vector DB Indexing ===")
    index_data = {
        "collection": "sales_data",  # Replace with your actual collection name
        "text_fields": ["product_description", "customer_feedback"],  # Replace with actual text fields
        "sample": True
    }
    response = requests.post(
        f"{BASE_URL}/vectordb/index",
        json=index_data
    )
    print_response(response)

def test_vector_db_search():
    """Test vector DB search"""
    print("\n=== Testing Vector DB Search ===")
    search_data = {
        "query": "customer satisfaction with premium products",
        "k": 3
    }
    response = requests.post(
        f"{BASE_URL}/vectordb/search",
        json=search_data
    )
    print_response(response)

def test_llm_query():
    """Test LLM-based query generation and execution"""
    print("\n=== Testing LLM Query ===")
    query_data = {
        "prompt": "most revenue movies in US in both musical and drama categories, select only revenuem, title, year, Genre and then sort in descending order by revenue",
        "collection": "users",  
        "max_results": 5
    }
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/llm/query",
        json=query_data
    )
    end_time = time.time()
    
    print(f"Query completed in {end_time - start_time:.2f} seconds")
    print("\nResults:")
    print_response(response)

def run_all_tests():
    """Run all tests in sequence"""
    test_root_endpoint()
    # test_explore_data()
    # test_mongodb_query()
    # test_vector_db_indexing()
    # test_vector_db_search()
    # test_analyze_data()
    test_llm_query()

if __name__ == "__main__":
    run_all_tests()
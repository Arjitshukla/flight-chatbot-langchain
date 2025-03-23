from django.shortcuts import render
from django.http import JsonResponse
import requests
import re
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

RAPIDAPI_KEY = "810cc052e8mshb5c92ca94aa209ep16e511jsn254c65a2263b"
FLIGHT_API_URL = "https://flights-sky.p.rapidapi.com/flights/price-calendar-web"

# 🚀 Initialize Hugging Face NLP Model
chat_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)
chat_llm = HuggingFacePipeline(pipeline=chat_pipeline)

memory = ConversationSummaryMemory(llm=chat_llm, input_key="flight_data", memory_key="chat_history", max_token_limit=250)

template = """
You are an AI assistant trained specifically for helping users find flights. 
You should ONLY answer questions related to flights, such as:
- Finding flights between two locations
- Checking flight prices
- Travel dates and availability

Below is a list of flights. Please summarize them in a user-friendly and readable way:

{flight_data}

Provide a clear and concise response to the user, listing the available flights with their details.
"""

prompt = PromptTemplate(input_variables=["flight_data"], template=template)
llm_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory)

MAX_INPUT_TOKENS = 400

def truncate_text(text, max_tokens=MAX_INPUT_TOKENS):
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]) + "..."
    return text

def extract_entities(text):
    """Extract location and date details from user message."""
    result = {"from_location": None, "to_location": None, "date": None}

    date_match = re.search(r"\b(\d{4}-\d{2})\b", text)
    if date_match:
        result["date"] = date_match.group(1)

    words = text.split()
    airport_codes = [word.upper() for word in words if len(word) == 3 and word.isalpha()]
    
    if len(airport_codes) >= 2:
        result["from_location"], result["to_location"] = airport_codes[:2]
    elif len(airport_codes) == 1:
        result["from_location"] = airport_codes[0]
    return result

def fetch_flight_data(from_location, to_location, from_date):
    """Fetch flight data from API dynamically."""
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "flights-sky.p.rapidapi.com",
        "content-type": "application/json",
    }
    url = f"{FLIGHT_API_URL}?fromEntityId={from_location}&toEntityId={to_location}&yearMonth={from_date}"
    print(f"📡 Sending API Request to: {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        flight_data = response.json()
        print(f"✅ API Response: {flight_data}") 
        return flight_data
    else:
        print(f" API Error: {response.status_code}, {response.text}") 
        return {"error": "Could not fetch flight data"}

def parse_flight_data(flight_data):
    parsed_flights = []
    price_mapping = {}
    
    # Extract price details
    if "data" in flight_data and "PriceGrids" in flight_data["data"]:
        for grid in flight_data["data"]["PriceGrids"]["Grid"]:
            for price_info in grid:
                if "Indirect" in price_info and "TraceRefs" in price_info["Indirect"]:
                    for trace_ref in price_info["Indirect"]["TraceRefs"]:
                        price_mapping[trace_ref] = price_info["Indirect"]["Price"]
    
    if "data" in flight_data and "Traces" in flight_data["data"]:
        for key, trace in flight_data["data"]["Traces"].items():
            parts = trace.split("*")  # Split raw string based on "*"
            
            if len(parts) >= 7:
                trace_id = key  # The key acts as the trace reference
                price = price_mapping.get(trace_id, "N/A")  # Fetch price if available
                
                flight = {
                    "date": parts[4], 
                    "airline": parts[6], 
                    "from": parts[2],  # Departure airport
                    "to": parts[3],  # Arrival airport
                    "stops": "Non-stop" if len(parts) == 7 else f"1 stop ({parts[3]})",
                    "price": price  # Add price to flight details
                }
                parsed_flights.append(flight)
    
    return parsed_flights

def format_flight_data(parsed_flights):
    if not parsed_flights:
        return "No flights available."

    # Convert the parsed flights into a text format that the LLM can understand
    flights_text = "\n".join([
        f"Flight {i+1}: Airline: {flight['airline']}, Date: {flight['date']}, From: {flight['from']}, To: {flight['to']}, Stops: {flight['stops']}, Price: ${flight['price']}"
        for i, flight in enumerate(parsed_flights)
    ])

    # Truncate the text to avoid exceeding the token limit
    truncated_text = truncate_text(flights_text, max_tokens=300)  # Adjust max_tokens as needed

    print(f"Flight data being sent to LLM: {truncated_text}")  # Debug log

    # Use the LLM to generate a response
    ai_response = llm_chain.predict(flight_data=truncated_text)

    print(f"LLM response: {ai_response}")  # Debug log

    return ai_response

def process_user_input(user_message, session_data):
    """Ensure user provides from_location, to_location, and date before responding dynamically."""
    extracted_data = extract_entities(user_message)

    # Check and update session data
    from_location = extracted_data["from_location"] or session_data.get("from_location")
    to_location = extracted_data["to_location"] or session_data.get("to_location")
    from_date = extracted_data["date"] or session_data.get("date")

    # Store updated data in session
    session_data["from_location"] = from_location
    session_data["to_location"] = to_location
    session_data["date"] = from_date

    # Identify missing details
    missing_info = []
    if not from_location:
        missing_info.append("departure location")
    if not to_location:
        missing_info.append("destination")
    if not from_date:
        missing_info.append("travel date")

    # If any information is missing, generate an AI response
    if missing_info:
        ai_prompt = f"""
        A user is trying to find a flight but hasn't provided all details.
        They are missing: {', '.join(missing_info)}.
        Politely ask them to provide the missing details.
        """
        ai_response = llm_chain.predict(user_input=truncate_text(ai_prompt))  # AI-generated response

        return {
            "response": f"  {ai_response}",
            "session_data": session_data
        }
    return {"response": None, "session_data": session_data}


def chat_view(request):
    """Handle chatbot requests dynamically based on user input."""
    if request.method == "POST":
        user_message = request.POST.get("message").strip().lower()
        print(f"User message: {user_message}")
        
        # Load session data
        session_data = request.session.get("flight_data", {})
        print(f"Before processing: {session_data}")

        # Process user input & check missing details
        info_response = process_user_input(user_message, session_data)

        # Store UPDATED session data
        request.session["flight_data"] = info_response["session_data"]
        request.session.modified = True  

        print(f"After processing: {request.session.get('flight_data')}")

        # If AI response exists, return it
        if info_response["response"]:
            return JsonResponse({"response": info_response["response"]})

        updated_session_data = request.session.get("flight_data", {})
        from_location = updated_session_data.get("from_location")
        to_location = updated_session_data.get("to_location")
        from_date = updated_session_data.get("date")

        print(f"Fetching flight data with: from={from_location}, to={to_location}, date={from_date}")
        if not from_location or not to_location or not from_date:
            return JsonResponse({"response": "   Missing flight details. Please provide them again."})

        flight_data = fetch_flight_data(from_location, to_location, from_date)

        if "error" in flight_data or not flight_data.get("data"):
            return JsonResponse({"response": "   No flights found. Please try another date."})

        parsed_flights = parse_flight_data(flight_data)

        structured_response = format_flight_data(parsed_flights) 

        return JsonResponse({"response": f"{structured_response}"})
    return render(request, "chat/chat.html")
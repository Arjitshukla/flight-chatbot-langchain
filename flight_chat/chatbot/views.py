import logging
from django.shortcuts import render
from django.http import JsonResponse
import spacy
from dateutil import parser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import random
import aiohttp
import asyncio
from asgiref.sync import sync_to_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAPIDAPI_KEY = "810cc052e8mshb5c92ca94aa209ep16e511jsn254c65a2263b"
FLIGHT_API_URL = "https://flights-sky.p.rapidapi.com/flights/price-calendar-web"
AIRPORT_LOOKUP_URL = "https://flights-sky.p.rapidapi.com/flights/auto-complete"

nlp = spacy.load("en_core_web_sm")

chat_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn", max_new_tokens=100)
chat_llm = HuggingFacePipeline(pipeline=chat_pipeline)

memory = ConversationBufferMemory(input_key="flight_data", memory_key="chat_history", max_token_limit=250)

flight_summary_template = """Summarize the following flight options:
- Departure: {from_city}
- Arrival: {to_city}
- Date: {date}
- Available Flights: {flight_data}

Provide a short, clear summary."""

prompt = PromptTemplate(input_variables=["from_city", "to_city", "date", "flight_data"], template=flight_summary_template)
llm_chain = LLMChain(llm=chat_llm, prompt=prompt, memory=memory)

def extract_flight_info(user_message):
    logger.info(f"Extracting flight info from message: {user_message}")
    result = {"from_city": None, "to_city": None, "date": None, "cheapest": False}
    
    try:
        result["date"] = parser.parse(user_message, fuzzy=True).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        logger.warning("No valid date found in the message.")
        result["date"] = None  
    
    doc = nlp(user_message)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    
    if len(cities) >= 2:
        result["from_city"], result["to_city"] = cities[:2]
        logger.info(f"Detected cities - From: {result['from_city']}, To: {result['to_city']}")
    else:
        logger.warning("Could not detect both departure and destination cities.")

    if "cheapest" in user_message.lower() or "lowest price" in user_message.lower():
        result["cheapest"] = True

    return result

async def get_airport_code(city):
    if not city:
        return None 
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "flights-sky.p.rapidapi.com"}
    async with aiohttp.ClientSession() as session:
        async with session.get(AIRPORT_LOOKUP_URL, headers=headers, params={"query": city}) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch airport code for {city}. Status: {response.status}")
                return None
            data = await response.json()
            for entry in data.get("data", []):
                if "relevantFlightParams" in entry.get("navigation", {}):
                    code = entry["navigation"]["relevantFlightParams"].get("skyId")
                    logger.info(f"Airport code for {city}: {code}")
                    return code
    return None

async def fetch_flight_data(from_city, to_city, flight_date):
    if not from_city or not to_city or not flight_date:
        logger.error("Invalid input parameters for flight search.")
        return {"error": "Invalid input parameters."}
    
    from_airport, to_airport = await asyncio.gather(
        get_airport_code(from_city),
        get_airport_code(to_city)
    )

    if not from_airport or not to_airport:
        logger.error("Invalid city names, unable to fetch airport codes.")
        return {"error": "Invalid city names."}

    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "flights-sky.p.rapidapi.com"}
    url = f"{FLIGHT_API_URL}?fromEntityId={from_airport}&toEntityId={to_airport}&yearMonth={flight_date[:7]}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"Flight API error: {response.status}")
                return {"error": "Flight API error"}
            data = await response.json()
            logger.info("Flight data fetched successfully.")
            return data
    
def format_flight_data(from_city, to_city, date, flight_data, cheapest=False):
    if "error" in flight_data:
        return flight_data["error"]

    traces = flight_data.get("data", {}).get("Traces", {})

    if not traces:
        return "No flights available."

    flights = []
    
    for key, value in traces.items():
        parts = value.split("*")
        if len(parts) >= 5:
            departure_time = f"{parts[0][6:8]}:{parts[0][8:10]}"  
            departure_airport = parts[2]  
            arrival_airport = parts[3]  
            arrival_date = f"{parts[4][:4]}-{parts[4][4:6]}-{parts[4][6:8]}"  
            airline = parts[5]  
            price = random.randint(150, 900)

            flights.append({
                "airline": airline,
                "departure_airport": departure_airport,
                "arrival_airport": arrival_airport,
                "departure_time": departure_time,
                "arrival_date": arrival_date,
                "price": price
            })

    if cheapest:
        flights = sorted(flights, key=lambda x: x["price"])[:1]

    table_rows = [
        f"""
        <tr>
            <td>{flight['airline']}</td>
            <td>{flight['departure_airport']}</td>
            <td>{flight['arrival_airport']}</td>
            <td>{flight['departure_time']}</td>
            <td>{flight['arrival_date']}</td>
            <td>${flight['price']}</td>
        </tr>
        """ for flight in flights
    ]

    table_html = f"""
        <style>
            table {{
                width: 70%;
                margin: 20px auto;
                border-collapse: collapse;
                text-align: center;
                font-family: Arial, sans-serif;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{background-color: #f2f2f2;}}
            tr:hover {{background-color: #ddd;}}
        </style>
        <table>
            <tr>
                <th>Airline</th>
                <th>Departure</th>
                <th>Arrival</th>
                <th>Time</th>
                <th>Arrival Date</th>
                <th>Price</th>
            </tr>
            {''.join(table_rows)}
        </table>
    """

    return f"🛫 <b>Flights from {from_city} to {to_city} on {date}:</b><br><br>{table_html}"


async def chat_view(request):
    if request.method == "GET":
        return render(request, "chat/chat.html") 

    elif request.method == "POST":
        user_message = request.POST.get("message", "").strip()
        flight_info = extract_flight_info(user_message)
        flight_info["date"] = flight_info["date"] or "2025-03"

        flight_data = await fetch_flight_data(flight_info["from_city"], flight_info["to_city"], flight_info["date"])
        response_text = format_flight_data(flight_info["from_city"], flight_info["to_city"], flight_info["date"], flight_data, flight_info["cheapest"])
        
        return JsonResponse({"response": response_text})

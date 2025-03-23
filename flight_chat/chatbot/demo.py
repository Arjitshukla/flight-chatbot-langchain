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
import aiohttp
import asyncio
from asgiref.sync import sync_to_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(_name_)

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
    result = {"from_city": None, "to_city": None, "date": None}
    
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
        
def format_flight_data(from_city, to_city, date, flight_data):
    if "error" in flight_data:
        return flight_data["error"]

    flight_details = []
    for key, value in flight_data.get("data", {}).get("Traces", {}).items():
        parts = value.split("*")  
        if len(parts) >= 5:
            departure_time = f"{parts[0][6:8]}:{parts[0][8:10]}"  
            departure_airport = parts[2]  
            arrival_airport = parts[3]  
            arrival_date = f"{parts[4][:4]}-{parts[4][4:6]}-{parts[4][6:8]}"  
            airline = parts[5]  

            flight_details.append(
                f"✈ *{airline}* | {departure_airport} → {arrival_airport}\n"
                f"   🕒 Departure: {departure_time} on {date}\n"
                f"   📅 Arrival: {arrival_date}\n"
            )

    formatted_data = "\n".join(flight_details) if flight_details else "No flights available."

    response_text = (
        f"🛫 *Flights from {from_city} to {to_city} on {date}:*\n\n{formatted_data}"
    )

    return response_text




async def chat_view(request):
    if request.method == "GET":
        logger.info("Rendering chat.html")
        return render(request, "chat/chat.html") 

    elif request.method == "POST":
        user_message = request.POST.get("message", "").strip()
        if not user_message:
            return JsonResponse({"response": "Please provide departure and destination cities."})

        logger.info(f"User message received: {user_message}")

        flight_info = extract_flight_info(user_message)
        if not flight_info["from_city"] or not flight_info["to_city"]:
            return JsonResponse({"response": "Please specify both departure and destination cities."})
        
        flight_info["date"] = flight_info["date"] or "2025-03"
        logger.info(f"Fetching flight data for: {flight_info}")

        flight_data = await fetch_flight_data(flight_info["from_city"], flight_info["to_city"], flight_info["date"])
        
        response_text = format_flight_data(flight_info["from_city"], flight_info["to_city"], flight_info["date"], flight_data)
        
        return JsonResponse({"response": response_text})
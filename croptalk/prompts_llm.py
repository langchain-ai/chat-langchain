RESPONSE_TEMPLATE = """\
You are an expert Crop insurance agent.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] ([${{page}}]) notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "I'm \
not sure." and request additional information. SPECIFY what information you need to be able to answer the question. Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.  Put citations where they apply rather than putting them all at the end.\

"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, identify whether the question is a follow-up question.\
If it is, rephrase the question to be a standalone question. If it is not, return the question as is.\

Use this Response Schema to rephrase the follow-up question into a standalone question:
<schema> [Question body]. Commodity: [The most recently mentioned commodity]. State: [The most recently mentioned state]. County: [The most recently mentioned county].</schema>

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone Question:"""


COMMODITY_TEMPLATE = """\
Given the following question, identify whether is it matches to any of the following commodities. 
If it is, extract the relevant commodity and return it. If it is not, return 'None'.

Commodities: \
['Wheat', 'Pecans', 'Cotton', 'Peaches', 'Corn', 'Peanuts', 'Whole Farm Revenue Protection', 'Soybeans', 'Pasture,Rangeland,Forage', 'Sesame', 'Controlled Environment', 'Apiculture', 'Hemp', 'Micro Farm', 'Blueberries', 'Oats', 'Fresh Market Sweet Corn', 'Grain Sorghum', 'Potatoes', 'Oysters', 'Triticale', 'Cucumbers', 'Canola', 'Popcorn', 'Fresh Market Tomatoes', 'Feeder Cattle', 'Fed Cattle', 'Cattle', 'Weaned Calves', 'Swine', 'Milk', 'Dairy Cattle', 'Forage Production', 'Dry Peas', 'Barley', 'Cabbage', 'Onions', 'Cotton Ex Long Staple', 'Chile Peppers', 'Dry Beans', 'Apples', 'Pistachios', 'Grapefruit', 'Lemons', 'Tangelos', 'Oranges', 'Mandarins/Tangerines', 'Rice', 'Hybrid Seed Rice', 'Grapes', 'Forage Seeding', 'Walnuts', 'Almonds', 'Prunes', 'Safflower', 'Cherries', 'Processing Cling Peaches', 'Kiwifruit', 'Olives', 'Tomatoes', 'Fresh Apricots', 'Processing Apricots', 'Pears', 'Raisins', 'Table Grapes', 'Figs', 'Plums', 'Alfalfa Seed', 'Strawberries', 'Tangelo Trees', 'Orange Trees', 'Grapefruit Trees', 'Lemon Trees', 'Fresh Nectarines', 'Processing Freestone', 'Fresh Freestone Peaches', 'Mandarin/Tangerine Trees', 'Pomegranates', 'Sugar Beets', 'Grapevine', 'Cultivated Wild Rice', 'Mint', 'Avocados', 'Caneberries', 'Millet', 'Sunflowers', 'Annual Forage', 'Nursery (NVS)', 'Silage Sorghum', 'Hybrid Sweet Corn Seed', 'Cigar Binder Tobacco', 'Cigar Wrapper Tobacco', 'Sweet Corn', 'Processing Beans', 'Green Peas', 'Flue Cured Tobacco', 'Tangors', 'Peppers', 'Sugarcane', 'Macadamia Nuts', 'Macadamia Trees', 'Banana', 'Coffee', 'Papaya', 'Banana Tree', 'Coffee Tree', 'Papaya Tree', 'Hybrid Popcorn Seed', 'Mustard', 'Grass Seed', 'Flax', 'Hybrid Corn Seed', 'Pumpkins', 'Burley Tobacco', 'Hybrid Sorghum Seed', 'Camelina', 'Dark Air Tobacco', 'Fire Cured Tobacco', 'Sweet Potatoes', 'Maryland Tobacco', 'Cranberries', 'Clams', 'Buckwheat', 'Rye', 'Fresh Market Beans', 'Clary Sage', 'Hybrid Vegetable Seed', 'Cigar Filler Tobacco', 'Tangerine Trees', 'Lime Trees']
Question: {question}
commodity: """

INS_PLAN_TEMPLATE = """\
Given the following question, identify whether is it mentions any of the following crop insurance plans. 
If it is, return the plan abbreviation. If it is not, return 'None'.

Example: "Can apples be covered under ARH?" -> "ARH"
Example: "Can apples be covered under Yield Protection?" -> "YP"
Insurance Plans table: \
Abbreviation,Full Name
YP,Yield Protection
RP,Revenue Protection
ARH,Actual Revenue History
WFRP,Whole Farm Revenue Protection
APH,Actual Production History

Question: {question}
Insurance Plan:
"""

STATE_TEMPLATE = """\
Identify whether the following text mentions a state of the US.
If it does, return the state name. Otherwise, return None.
Example: "I live in California, Ventura County" -> "California"
Text: {question}
State: """

COUNTY_TEMPLATE = """\
Identify whether the following text mentions a county of the US state.
If it does, return the county name. Otherwise, return None.
Example: "I live in California, Ventura County" -> "Ventura"
Text: {question}
County: """

DOC_CATEGORY_TEMPLATE = """\
Given the following question, identify whether it mentions any of the following document categories.
If it does, return the document category abbreviation. Otherwise, return None.

Example: "Is there a special provisions document for apples in Yakima county Washington for APH insurance plan?" -> "SP"
Example: "Show me sections of CIH related to apples" -> "CIH"
Document categories table: \
Abbreviation,Full Name
CIH,Crop Insurance Handbook
BP,Basic Provisions
CP,Crop Provisions
SP,Special Provisions

Question: {question}
Document category:
"""

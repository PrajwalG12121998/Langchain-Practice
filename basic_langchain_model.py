#Calling the open AI library directly 
import os
import openai
from dotenv import load_dotenv, find_dotenv
OPEN_AI_LLM_MODEL = 'gpt-3.5-turbo-0301'

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def get_completion(prompt, model=OPEN_AI_LLM_MODEL):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = 0
    )
    return response.choices[0].message['content']

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English in a calm and respectful tone"""

prompt = f"""Translate the text that is delimited by triple backticks into a style that is {style}.
text: ```{customer_email}```
"""

response = get_completion(prompt)


#Now lets do the same thing with the help of langchain
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0.0, model=OPEN_AI_LLM_MODEL)

#ChatOpenAI(verbose=False, callbacks=None, callback_manager=None, 
#client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, 
#model_name='gpt-3.5-turbo-0301', temperature=0.0, model_kwargs={}, 
#openai_api_key=None, openai_api_base=None, openai_organization=None, 
#request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None)


from langchain.prompts import ChatPromptTemplate

template_string = """
Translate the text that is delimited by triple backticks into a style that is {style}
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_messages = prompt_template.format_messages(
    style = style,
    text = customer_email
)

customer_response = chat(customer_messages)

#Output received
#I'm really frustrated that my blender lid flew off and made a mess of my kitchen walls with smoothie. 
#To add to my frustration, the warranty doesn't cover the cost of cleaning up my kitchen. 
#Can you please help me out, friend?

#Output Parsers
#How do you want the LLM to give the output

#Sample response
#{
#  "gift": False,
#  "delivery_days": 5,
#  "price_value": "pretty affordable!",
#  "feedback": "positive"
#}

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

feedback: What was the overall sentiment of using the product
Answer Positive if customers were happy, else Negative if the customer didn't like the product

Format the output as JSON with the following keys:
gift
delivery_days
price_value
feedback

text: {text}
"""

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)

messages = prompt_template.format_messages(text=customer_review)
chat = ChatOpenAI(temperature=0.0, model=OPEN_AI_LLM_MODEL)
response = chat(messages)
print(response.content)

#The output of the above response is a string. But let's use Parser to generate output in JSON

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

feedback_schema = ResponseSchema(name="feedback",
                                description="What was the overall sentiment of using the product\
                                             Answer Positive if customers were happy, else Negative \
                                             if the customer didn't like the product")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema,
                    feedback_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)

response = chat(messages)
output_dict = output_parser.parse(response.content)
type(output_dict) #dict
output_dict.get('delivery_days') #2






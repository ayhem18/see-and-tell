import asyncio
import os
from langchain.chat_models import ChatOpenAI as LLMOpenAI
from langchain.output_parsers import ResponseSchema, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
)

from pydantic import BaseModel, Field


__prompt = open(os.path.join(os.path.dirname(__file__), 'prompt',  'get_caption.txt')).read()

class Captions(BaseModel):
    caption: str = Field(
        ..., 
        description="List of words or phrases that can be substituted with characters."
    )

    
async def get_captions(
    caption: str,
    place: str,
) -> Captions:
    client = LLMOpenAI(
        model='gpt-4',
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_API_ORG"),
        max_retries=1,
        temperature=0.996,
        max_tokens=256,
    )

    # print(schemas)
    output_parser = PydanticOutputParser(pydantic_object=Captions)

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                __prompt +
                "\n{format_instructions}",
            )
        ],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions(),
        },
    )

    output = await client.ainvoke(
        prompt.format_prompt(
            caption=caption.replace('"', ''),
            place=place,
        ).to_messages()
    )

    return output_parser.parse(output.content.replace('\n', ''))


async def rewrite_with_llm(
    captions: list[str],
    places: list[str],
) -> list[Captions]:
    return await asyncio.gather(*[get_captions(caption, place) for caption, place in zip(captions, places)])
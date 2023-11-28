import asyncio
import os
from langchain.chat_models import ChatOpenAI as LLMOpenAI
from langchain.output_parsers import ResponseSchema, PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
)

from pydantic import BaseModel, Field


__prompt = open(os.path.join(os.path.dirname(__file__), 'prompt',  'get_message.txt')).read()

class Words(BaseModel):
    words: list[str] = Field(
        ..., 
        description="List of words or phrases that can be substituted with characters."
    )

    
async def get_words(
    sentence: str
) -> Words:
    client = LLMOpenAI(
        model='gpt-4',
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_API_ORG"),
        max_retries=1,
        temperature=0.776,
        max_tokens=1024,
    )

    # print(schemas)
    output_parser = PydanticOutputParser(pydantic_object=Words)

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
        prompt.format_prompt(sentence=sentence).to_messages()
    )

    return output_parser.parse(output.content.replace('\n', ''))


async def get_substitutions(
    sentences: str
) -> list[Words]:
    return await asyncio.gather(*[get_words(sentence) for sentence in sentences])
#!/usr/bin/env python3
from pprint import pprint as pp
import os
from typing import reveal_type

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall


def send_messages(messages):
    response = client.chat.completions.create(
        model='deepseek-chat', messages=messages, tools=tools
    )
    return response.choices[0].message


client = OpenAI(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url='https://api.deepseek.com',
)

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get weather of an location, the user shoud supply a location first',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The city and state, e.g. San Francisco, CA',
                    }
                },
                'required': ['location'],
            },
        },
    },
]

messages = [
    {
        'role': 'user',
        'content': "Say hello. Then, how's the weather in Hangzhou?",
    }
]
message = send_messages(messages)
print(f'User>\t {messages[0]["content"]}')

print(f'Model>\t {message.content}')

tool: ChatCompletionMessageToolCall = message.tool_calls[0]

reveal_type(message)
reveal_type(tool)
print(dir(tool))

print(type(tool))
messages.append(
    {
        'role': message.role,
        'content': message.content,
        'tool_calls': message.tool_calls,
    }
)

messages.append(
    {
        'role': 'tool',
        'tool_call_id': tool.id,
        'content': '24℃',
    }
)
message = send_messages(messages)
print(f'Model>\t {message.content}')
reveal_type(message)

print('-------------')

pp(messages)
reveal_type(messages)

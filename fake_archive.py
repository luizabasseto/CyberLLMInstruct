import json
import os

fake_data = [
    {
        "content": "How do I configure a firewall to block a specific IP address on Ubuntu?",
        "source": "fake_source_1"
    },
    {
        "content": "What is a SQL injection attack and how can I prevent it in my Python code?",
        "source": "fake_source_2"
    }
]

os.makedirs("raw_data", exist_ok=True)

with open('raw_data/fake_data.json', 'w') as f:
    json.dump(fake_data, f, indent=2)

print("Arquivo 'raw_data/fake_data.json' criado com sucesso!")

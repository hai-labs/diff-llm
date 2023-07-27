import os
import requests

root_url = "https://api.wikimedia.org/core/v1/wikipedia/en"
page = "Earth"
history_url = f"{root_url}/page/{page}/history"
filter_type = 'bot'
older_than = 981126172
parameters = {'filter': filter_type, 'older_than': older_than}

headers = {
  'Authorization': f'Bearer {os.environ["WIKIMEDIA_ACCESS_TOKEN"]}',
  'User-Agent': 'diff-llm (niels.bantilan@gmail.com)'
}

response = requests.get(history_url, headers=headers, params=parameters)
revisions = response.json()


old = revisions["revisions"][0]["id"]
new = revisions["revisions"][1]["id"]
compare_url = f"{root_url}/revision/{old}/compare/{new}"
response = requests.get(compare_url, headers=headers, params=parameters)
comparison = response.json()


page_url = f"{root_url}/page/{old}/compare/{new}"
import ipdb; ipdb.set_trace()

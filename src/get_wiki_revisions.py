import os

from wikiwho_wrapper import WikiWho

wikiwho = WikiWho(
    os.environ["WIKIWHO_USERNAME"],
    os.environ["WIKIWHO_PASSWORD"],
    lng='en',
)


response = wikiwho.api.all_content("Earth")
dataview = wikiwho.dv.all_content("Earth")
import ipdb; ipdb.set_trace()

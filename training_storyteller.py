#!./venv/bin/python

from openai import OpenAI
import cli
import time
import json
import re
import shlex
from vocabulary import vocabulary

import xml.etree.ElementTree

def normalize_story(story):
    if story == None:
        return None

    story = ' '.join(story.split()).lower()

    # Fix versions of say: say, .. say :, say " ", say ' ' => say ()
    story = re.sub(
        r'say\s*[:,]?\s*\(\s*([^\'"\(\)]+?)\s*\)([?.!])',
        lambda m: f"say ({m.group(1)}){m.group(2)}",
        story
    )
    story = re.sub(
        r'say\s*[:,]?\s*([^\'"\(\)]+?)\s*([?.!])',
        lambda m: f"say ({m.group(1)}){m.group(2)}",
        story
    )
    story = re.sub(
        r'say\s*[:,]?\s*\'\s*([^\'"]+?)\s*\'([?.!])',
        lambda m: f"say ({m.group(1)}){m.group(2)}",
        story
    )
    story = re.sub(
        r'say\s*[:,]?\s*"\s*([^\'"]+?)\s*"([?.!])',
        lambda m: f"say ({m.group(1)}){m.group(2)}",
        story
    )
    return story


def parse_xml_stories(xml_string):
    """
    Parse the given XML string of the form:
    <stories>
        <story summary="…">…</story>
        …
    </stories>
    and return a list of dicts: [{"summary": str, "content": str}, …].
    Whitespace in content is collapsed to single spaces.
    """
    m = re.match(r'<stories>.*</stories>',xml_string) 
    if m:
        xml_string = m.group(0)
    
    xml_stories = xml.etree.ElementTree.fromstring(xml_string)
    stories = []
    for xml_story in xml_stories.findall('story'):
        summary = xml_story.get('summary', '')  # summary attribute
        # Extract all inner text (including across newlines)
        content = normalize_story(''.join(xml_story.itertext()))
        if content != None:
            stories.append({
                'summary': summary,
                'content': content
            })
    return stories


class TrainingStoryteller:
    def __init__(self, args):
        self._args = args
        self._grammar = None
        self._universe = None
        self._client = None
        self._base_messages = None
        self._queue = []

    @property
    def num_stories(self):
        return int(self._args.get("num_stories", 1))
    @property
    def num_stories_per_imagine(self):
        return int(self._args.get("num_stories_per_imagine", 1))
    @property
    def grammar_file(self):
        return str(self._args.get("grammar_file", "pidgin.md"))
    @property
    def universe_file(self):
        return str(self._args.get("universe_file", "universe.md"))
    @property
    def grammar(self):
        if self._grammar == None:
            with open(self.grammar_file, "r") as md:
                self._grammar = md.read()
        return self._grammar
    @property
    def universe(self):
        if self._universe == None:
            with open(self.universe_file, "r") as md:
                self._universe = md.read()
        return self._universe
    @property
    def url(self):
        return str(self._args.get("url", "https://api.openai.com/v1"))
    
    @property
    def key(self):
        return str(self._args.get("key", ""))

    @property
    def client(self):
        if self._client == None:
            self._client = OpenAI(api_key=self.key, base_url=self.url)
        return self._client
    
    @property
    def temperature(self):
        return float(self._args.get("temperature", 1.0))
    
    @property
    def training_stories_file(self):
        return self._args.get("training_stories_file","training_stories.json")
    
    @property
    def max_tokens(self):
        return int(self._args.get("max_tokens", 4000))

    @property
    def stories(self):
        data = []
        try:
            with open(self.training_stories_file, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        return data
    
    @stories.setter
    def stories(self,value):
        with open(self.training_stories_file, "w") as file:
            json.dump(value, file, indent=4)

    def remember(self,story):
        stories = self.stories
        stories.append(story)
        self.stories = stories

    @property
    def model(self):
        return self._args.get("model", "gpt-3.5-turbo")

    @property
    def models(self):
        """
        Returns a list of valid model IDs available via the OpenAI API.
        """
        try:
            resp = self.client.models.list()
            return [m.id for m in resp.data]
        except Exception as e:
            print(f"Could not retrieve models: {e}")
            return []

    @property
    def base_messages(self):
        if self._base_messages is None:
            self._base_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system", "content": f"Grammar:\n{self.grammar}"},
                {"role": "system", "content": f"Universe:\n{self.universe}"},
                {"role": "system", "content": f"Vocabulary:\n{json.dumps(vocabulary)}"},
            ]
        return self._base_messages

    def tell(self):
        if len(self._queue) == 0:
            self.imagine()
        story = self._queue[0]
        self._queue = self._queue[1:]
        return story
    
    @property
    def prompt(self):
        return f"""
        Using the provided grammar, universe, and vocabulary, 
        write about {self.num_stories_per_imagine} children's stories.
        Respond with precisely the story as narrator, 
        use the XML format <stories>...</stories> with each story in <story summary=\"English Summary\">...</story> tags.
        """
    
    def imagine(self):
        client = self.client
        try:
            messages = self.base_messages + [
                {"role": "user", "content": self.prompt }
            ]
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            print(response)
            xml_string = response.choices[0].message.content
            imagined_stories = parse_xml_stories(xml_string)
            self._queue += imagined_stories
        except Exception as e:
            # retry with GPT-3.5 if model not found
            err = getattr(e, 'error', {})
            if isinstance(err, dict) and err.get('code') == 'model_not_found':
                print("Model not found, retrying with gpt-3.5-turbo...")
                self._args['model'] = 'gpt-3.5-turbo'
                return self.tell()
            print(f"An error occurred: {e}")
            return None

def main():
    args = cli.args()
    llm_args = args["training-storyteller"]
    storyteller = TrainingStoryteller(llm_args)

    if args.get("models",False):
        for model in storyteller.models:
            print(model)
        return

    if args.get("xml",None) != None:
        print(parse_xml_stories(args.get("xml")))
        return

    for i in range(storyteller.num_stories):
        print(f"Imagining story {i + 1} of {storyteller.num_stories}...")
        story = storyteller.tell()
        if story != None:
            storyteller.remember(story)

if __name__ == "__main__":
    main()
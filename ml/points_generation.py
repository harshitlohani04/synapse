import os
import requests
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from dotenv import load_dotenv
from transformers import pipeline
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed # for main prototyping will later shift to asyncio for better efficiency
import asyncio
import aiohttp
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# nlp processing class
class KeywordAndSentencesExtractor:
    def __init__(self, transcription):
        super().__init__()
        self.transcription = transcription

    def preprocess_transcription(self):
        sentences = sent_tokenize(self.transcription)
        stop_words = set(stopwords.words("english"))
        cleaned_sentence = []

        for s in sentences:
            words = word_tokenize(s)
            cleaned_arr = [w for w in words if w.isalnum() and w not in stop_words]
            cleaned_sentence.append("".join(cleaned_arr))

        return sentences, cleaned_sentence

    def _similarity_matrix(self, sentences):

        vectorizer = TfidfVectorizer()
        sim_mat = vectorizer.fit_transform(sentences)
        return cosine_similarity(sim_mat)
    
    def text_extraction(self, top_k: int = 5):
        original, cleaned = self.preprocess_transcription()
        if len(original)<top_k:
            return " ".join(original)

        sim_matrix = self._similarity_matrix(cleaned)
        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph) # {node1 : pr_value} format
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original)), reverse=True)
        summ_array = [s for _, s in ranked_sentences[:top_k]]

        return " ".join(summ_array)
    
    def keyword_extraction(self, top_n: int = 10):
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(self.transcription, 
                                             keyphrase_ngram_range=(1, 3), # Allow up to 3-word phrases
                                             stop_words='english',
                                             use_mmr=True, # using the maximal marginal relevance for better extraction
                                             diversity=0.7,
                                             top_n=top_n)
        return [kw for kw, _ in keywords]

# trello integration class
class TrelloIntegration:
    def __init__(self, name, desc, token, api_key, default_lists: bool = True, n: int=None): # board desc to be added
        super().__init__()
        # keyword extraction model
        self.zero_shot_classifier = pipeline("zero-shot-classification")
        # defining the connection creation variables
        self.token = token
        self.api_key = api_key
        self.n = n
        self.default_lists = default_lists
        self.board_desc = desc
        if not self.api_key or not self.token:
            print("🔴 ERROR: TRELLO_KEY or TRELLO_TOKEN not found in environment.")
        self.url = "https://api.trello.com/1/boards/"
        self.boardURL = ""
        # Establish connection with trello
        query = {
            "name": name,
            "key": self.api_key,
            "desc": desc,
            "token": self.token,
            "defaultLists": "true" if default_lists else "false"
        }
        try:
            # ------------ checking for already existing board by the same name pending ------------- #
            response = requests.post(
                url=self.url,
                params=query,
            )
            if response.status_code!=200:
                print(f"following is the error message from trello : {response.text}")
            response.raise_for_status() # error if board creation failed
            self.board_response = response.json()
            print(response.json())
            self.boardURL = self.board_response.get("url") # fetching the board url
            print(response.text)
        except requests.exceptions.HTTPError as e:
            self.board_response = {"status_code": 400, "message": e}
            print(f"error encountered in board creation : {e}")

    async def generate_list_names(summary: str, n: int = 3): # function to generate "n" lists for the users
        llm = ChatOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=128
        )
        promptTemplate = ChatPromptTemplate([
            ("system", "You are a Trello workflow automation agent and you are to produce {n} list names for a particular summary."),
            ("user", "Given this context {context} you are to generate the names for the lists that can be created on the trello board for the user, basically dividing the entire task into the required number of list names. Output only the list of the list names and no explanation for the same")
        ])

        promptTemplate.invoke({"n": n, "context": summary})
        try:
            response = await llm.invoke(promptTemplate)
        except Exception:
            return None
        return response.content

    @property
    def _get_board_url(self):
        # simply returns the url of the created board
        return self.boardURL

    async def list_processing(self, list_id: str, card: list):
        url = "https://api.trello.com/1/cards"
        headers = {
          "Accept": "application/json"
        }
        query = {
                  'idList': list_id,
                  "name": card,
                  'key': self.api_key,
                  'token': self.token
                }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=query, headers=headers) as response:
                response.raise_for_status()

    async def _add_cards_to_list(self, keywords: list):
        board_id = self.board_response.get("id")

        # if the user selects the option for custom lists
        if self.n and not self.default_lists:
            # Creating the lists in the board
            url_list = "https://api.trello.com/1/lists"
            names = await self.generate_list_names(self.board_desc, self.n) # supposed to be in the list format
            query = {
                "name": '{name}',
                "idBoard": board_id,
                "key": self.api_key,
                "token": self.token
            }
            async def add_lists(session, name):
                payload = query["name"].format(name=name)
                query["name"] = payload
                async with session.get(url = url_list, params=query) as response:
                    response.raise_for_status()
                    return response.text
            async with aiohttp.ClientSession() as session:
                list_processing_coroutines = [add_lists(session, names[i]) for i in range(len(names))]
                _ = await asyncio.gather(*list_processing_coroutines) # text messages from the list creation request

        url = self.url + f"{board_id}/lists"
        headers = {
            "Accept": "application/json"
        }
        query = {
            "key": self.api_key,
            "token": self.token
        }
        list_response = requests.get(url=url, headers=headers, params=query)
        board_lists = list_response.json()
        list_id_map = {l.get("name"): l.get("id") for l in board_lists} # list name mapping to the id
        list_names = [lists.get("name") for lists in board_lists]
        print(list_names)

        async def keyword_classifier(): # zero-shot classification for the keywords

            '''
            Code segment for improving the keywords of the keybert model using langchain
            '''
            prompt_template_key = ChatPromptTemplate([
                ("system", "You are a trello card creator and you are to improve the currently existing card text."),
                ('user', 'Given this context {context} and these cards values : {cards} you are to improve the current card values such that the card values are more contextually correct and to the point that are able to explain the particular task that they are meant to explain.\
                 The output should be in a strict list format and no other information should be there along with it.')
            ])

            model = ChatOpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=128
            )
            try:
                list_names = await model.invoke(prompt_template_key)
            except Exception:
                list_names = None
            '''
            classification using the zero shot learning model.
            '''
            list_key_mapping = defaultdict(list)
            try:
                if list_names:
                    for key in keywords:
                        label = self.zero_shot_classifier(key, list_names)
                        list_key_mapping[label["labels"][0]].append(key)
                return list_key_mapping
            except Exception as e:
                print(f"error in list classification : {e}")
                return None
            
        # fetch the lists and add the cards to the respective list
        listKeyMap = await keyword_classifier()
        # {"todo": ["card1", "card2"]} --> desired output from the function
        
        try: # threading for adding the cards to the lists simultaneously

            # Can be made more efficient as per gemini
            # right now the entire thread pool is destroyed and made again for different iterations -> inefficient
            # every processing can be done by creating a seperate thread for all -> more efficient
            # most efficient -> asyncio
            tasks = []
            for list_name, list_id in list_id_map.items():
                for card in listKeyMap[list_name]:
                    tasks.append(self.list_processing(list_id, card))
                _ = asyncio.gather(*tasks)
            print("Adding cards to the lists successful")
        except Exception as e:
            print(f"Error occured in adding the cards to the lists! : {e}")


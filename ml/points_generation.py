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
import asyncio
import aiohttp
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

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
    load_dotenv()
    groq_key = os.getenv("GROQ_KEY")

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
            self.boardURL = self.board_response.get("url") # fetching the board url
        except requests.exceptions.HTTPError as e:
            self.board_response = {"status_code": 400, "message": e}
            print(f"error encountered in board creation : {e}")

    def generate_list_names(self, summary: str, n: int, user_context: str): # function to generate "n" lists for the users
        llm = ChatOpenAI(
            api_key=TrelloIntegration.groq_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=128
        )
        promptTemplate = ChatPromptTemplate([
                ("system", "You are a Trello workflow automation agent. Always respond ONLY with a valid JSON array of {n}"
                " list names. No explanations, no other text."),
                ("user", "Given this context {context} and to build a {user_context}, return the Trello list names as a "
                "JSON array of strings.")
            ])
        chain = promptTemplate | llm | StrOutputParser()
        try:
            response = chain.invoke({"n": n, "context": summary, "user_context": user_context})
            print(f"this is the llm response content {response}")
            try:
                names = json.loads(response)
            except Exception as e:
                names = [line.strip("-• ") for line in response.splitlines() if line.strip()]
            return names
        except Exception as e:
            return [e]

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
                raw = await response.text()
                return raw

    @staticmethod   
    async def improve_user_context(user_context):
        '''
        The main purpose of this function is to remove the vagueness and add more relatable context to the user context.
        groq/compound
        '''
        prompt_template = ChatPromptTemplate([
            ("system", "You are a context improver for a planning assistant."
                "Your task is to take the user's short or vague request and rewrite it into a clear, descriptive, and structured context."
                "Do not change the meaning of the request.Do not add irrelevant details."
                "Output must be STRICTLY valid JSON that can be parsed with json.loads in Python."
                "The JSON must be a string, which represents the improvised user context."
                "Do NOT include explanations, keys, markdown, or any text other than the JSON string."
            ),
            ("user", "Original user request: {user_context}" 
                "Rewrite this into a clearer and more detailed context that can guide a task planner."
            )
        ])

        model = ChatOpenAI(
                api_key=TrelloIntegration.groq_key,
                base_url="https://api.groq.com/openai/v1",
                model="groq/compound",
                temperature=0.1,
                max_tokens=128
            )

        chain = prompt_template | model | StrOutputParser()
        try:
            response = await chain.ainvoke({"user_context": user_context})
            try:
                new_context = json.loads(response)
            except json.JSONDecodeError:
                new_context = user_context
        except Exception as e:
            new_context = user_context
        
        return new_context

    async def _add_cards_to_list(self, keywords: list, user_context: str):
        board_id = self.board_response.get("id")
        user_context = await TrelloIntegration.improve_user_context(user_context)

        # if the user selects the option for custom lists
        if self.n and not self.default_lists:
            # Creating the lists in the board
            url_list = "https://api.trello.com/1/lists"
            names = self.generate_list_names(self.board_desc, self.n, user_context) # supposed to be in the list format
            print(names)
            query = {
                "name": '{name}',
                "idBoard": board_id,
                "key": self.api_key,
                "token": self.token
            }

            semaphore = asyncio.Semaphore(3)

            async def add_lists(session, name):
                payload = query.copy()
                payload["name"] = name
                async with semaphore:
                    async with session.post(url = url_list, params=payload) as response:
                        response.raise_for_status()
                        return response.text
            list_addition_flag = False
            async with aiohttp.ClientSession() as session:
                for attempt in range(5):
                    list_processing_coroutines = [add_lists(session, name) for name in names]
                    try:
                        sample = await asyncio.gather(*list_processing_coroutines) # text messages from the list creation request
                        list_addition_flag = True
                        break
                    except aiohttp.ClientResponseError as e:
                        if e.status == 429:
                            print(f"Rate limit error retrying for the {attempt+1} time")
                            await asyncio.sleep(2*attempt)
                if not list_addition_flag:
                    raise Exception("Maximum retries exhausted the list cannot be created")

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
                    (
                        "system",
                        "You are a Trello card creator. "
                        "Your task is to filter and improve the given card texts. "
                        "Only keep cards that are DIRECTLY relevant to the user's context : {user_context} "
                        "Completely ignore and drop unrelated cards. "
                        "Output must be STRICTLY valid JSON that can be parsed with json.loads in Python. "
                        "The JSON must be a list of strings, where each string is an improved card value. "
                        "Do NOT include explanations, keys, markdown, or any text other than the JSON list."
                    ),
                    (
                        "user",
                        "Context: {context} Cards: {cards}"
                        "Return ONLY the JSON list of improved card values."
                    )
                ])
            model = ChatOpenAI(
                api_key=TrelloIntegration.groq_key,
                base_url="https://api.groq.com/openai/v1",
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=128
            )

            chain = prompt_template_key | model | StrOutputParser()
            try:
                raw = await chain.ainvoke({"context":self.board_desc, "cards":keywords, "user_context": user_context})
                try:
                    card_names = json.loads(raw)
                except Exception as e:
                    card_names = [line.strip("-• ") for line in raw.splitlines() if line.strip()]
            except Exception:
                card_names = None
            print(f"These are the new card names to be classified : {card_names}")
            '''
            classification using the zero shot learning model.
            '''
            list_key_mapping = defaultdict(list)
            try:
                if list_names:
                    for key in card_names:
                        label = self.zero_shot_classifier(key, list_names)
                        list_key_mapping[label["labels"][0]].append(key)
                return list_key_mapping
            except Exception as e:
                print(f"error in list classification : {e}")
                return None
            
        # fetch the lists and add the cards to the respective list
        listKeyMap = await keyword_classifier()
        # {"todo": ["card1", "card2"]} --> desired output from the function
        
        try:
            tasks = []
            for list_name, list_id in list_id_map.items():
                for card in listKeyMap[list_name]:
                    tasks.append(self.list_processing(list_id, card))
            sample = await asyncio.gather(*tasks)
            print("Adding cards to the lists successful")
        except Exception as e:
            print(f"Error occured in adding the cards to the lists! : {e}")


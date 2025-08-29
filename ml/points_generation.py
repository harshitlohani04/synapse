import os
import requests
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
from dotenv import load_dotenv
from transformers import pipeline
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed # for main prototyping will later shift to asyncio for better efficiency

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
    
    def text_rank_extraction(self, top_k: int = 5):
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
                                             top_n=top_n)
        return [kw for kw, _ in keywords]

# trello integration class
class TrelloIntegration:
    def __init__(self, name, desc, token, api_key, create_default_lists: bool = True): # board desc to be added
        super().__init__()
        # defining the connection creation variables
        self.token = token
        self.api_key = api_key
        self.url = "https://api.trello.com/1/boards/"
        # Establish connection with trello
        query = {
            "name": f"{name}",
            "key": self.api_key,
            "token": self.token,
            "defaultLists": "true" if create_default_lists else "false"
        }
        try:
            # ------------ checking for already existing board by the same name pending ------------- #
            response = requests.post(
                url=self.url,
                params=query,
            )
            response.raise_for_status() # error if board creation failed
            self.board_response = response.text
            print(response.content)
        except requests.exceptions.HTTPError as e:
            self.board_response = {"status_code": 400, "message": e}
            print(f"error encountered in board creation : {e}")

    def list_processing(list_id: str, cards: list):
        url = "https://api.trello.com/1/cards"
        headers = {
          "Accept": "application/json"
        }
        for card_name in cards:
            query = {
              'idList': list_id,
              "name": card_name,
              'key': 'APIKey',
              'token': 'APIToken'
            }
            requests.post(url, headers, params=query)    

    def _add_cards_to_list(self, keywords: list):
        # list name can also be figured out from the video transcription in some way but in later versions
        board_id = self.board_response["id"]
        url = self.url + f"{board_id}/lists"
        headers = {
            "Accept": "application/json"
        }
        query = {
            "key": self.api_key,
            "token": self.token
        }
        list_response = requests.get(url=url, headers=headers, params=query)
        board_lists = list_response.text
        list_id_map = {list["name"]: list["id"] for list in board_lists} # list name mapping to the id
        list_names = [lists["name"] for lists in board_lists]
        print(list_names)

        def keyword_classifier(): # zero-shot classification for the keywords
            # no need for initializing as non-local as only read operation is performed
            list_key_mapping = defaultdict(list)
            try:
                zero_shot_classifier = pipeline("zero-shot-classification")
                for key in keywords:
                    label = zero_shot_classifier(key, list_names)
                    list_key_mapping[label["labels"][0]].append(key)
                    # print(f"keyword {key} likely belongs to the {label["labels"][0]} class")

                    return list_key_mapping
            except Exception as e:
                print(f"error in list classification : {e}")
                return None
            
        # fetch the lists and add the cards to the respective list
        listKeyMap = keyword_classifier()
        # {"todo": ["card1", "card2"]} --> desired output from the function
        
        try: # threading for adding the cards to the lists simultaneously
            for list_name, list_id in list_id_map:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(self.list_processing, list_id, listKeyMap[list_name])]
            print("Adding cards to the lists successful")
        except Exception as e:
            print(f"Error occured in adding the cards to the lists! : {e}")


load_dotenv()
def testing_function():
    trello_key = os.getenv("TRELLO_KEY")
    trello_token = os.getenv("TRELLO_TOKEN")

    # creating connection
    trello_instance = TrelloIntegration("sample board", " ", trello_token, trello_key)

if __name__ == "__main__":
    testing_function()


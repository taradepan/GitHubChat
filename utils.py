
import git
import os
from queue import Queue
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, Language, RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore")
from pprint import pprint
from langchain.llms import Replicate
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
import chromadb
from chromadb.utils import embedding_functions
import dotenv
dotenv.load_dotenv()
REPLICATE_API_TOKEN=os.getenv("replicate_api_token")
llm = Replicate(model="meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48")
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv("hf_api_key"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        self.last_name = git_link.split('/')[-1].split('.')[0]
        self.clone_path = self.last_name
        self.hf = huggingface_ef
        self.MyQueue = Queue(maxsize=2)
        self.chromadb = chromadb.Client()
        self.fnames = []
        self.unnecessary_extensions = ['FETCH_HEAD', 'description', 'HEAD', 'exclude', 'main', 'index', 'config', 'packed-refs', 'pre-rebase.sample', 'fsmonitor-watchman.sample', 'pre-push.sample', 'prepare-commit-msg.sample', 'pre-receive.sample', 'pre-commit.sample', 'pre-applypatch.sample', 'update.sample', 'pre-merge-commit.sample', 'post-update.sample', 'push-to-checkout.sample', 'applypatch-msg.sample', 'commit-msg.sample', 'pack-abc2ae661c17b50118c186ea774574eedc69c535.idx', 'pack-abc2ae661c17b50118c186ea774574eedc69c535.pack']

    
    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                if file not in self.unnecessary_extensions:
                    self.fnames.append(file)
                    f_ext = os.path.splitext(file)[1]
                    try:
                        if f_ext in [".py", ".js"]:
                            loader = GenericLoader.from_filesystem(
                                os.path.join(dirpath, file),
                                glob="*",
                                suffixes=[".py", ".js"],
                                parser=LanguageParser(),
                            )
                            self.docs.extend(loader.load())
                        else:
                            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                            self.docs.extend(loader.load_and_split())
                    except Exception as e:
                        pass

    def preprocess(self):
        self.data = []
        self.metadata = []
        for i in range(len(self.texts)):
            self.data.append(str(self.texts[i]))
            self.metadata.append(self.texts[i].metadata)
        for i in range(len(self.data)):
            self.data[i] = self.data[i].replace("\\n", " ")
       
       
    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.preprocess()
    
    def embed_chroma(self):
        self.db = self.chromadb.get_or_create_collection(name=self.last_name)
        self.db.add(
            embeddings=self.hf(self.data),
            documents=self.data,
            metadatas=self.metadata,
            ids=[str(i) for i in range(len(self.data))]
        )
        return self.db
    
    def load_db(self):
        if self.last_name not in self.chromadb.list_collections():
            self.extract_all_files()
            self.chunk_files()
            self.embed_chroma()
        else:
            self.db = self.chromadb.get_collection(name=self.last_name)


    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def retrieve_results(self, query):
        res = self.db.query(
            query_embeddings=self.hf([query]),
            n_results=1,
        )
        return res["documents"]
    

    def chain(self, query,reprompt="",max_count=5):
        if max_count==4:
            reprompt = "you provided wrong output for the given query. try again, but DON'T repeat yourself. learn and improve from previous outputs: "
        a=llm(f"""{reprompt}
            Availabe Files:{self.fnames}
            you are given a query: {query}
            give an output (only keywords) that can be used can be used to perfrom vector search and retrieve the data from the file.
            example: if query is "what are the packages used?", then output should be :""requirenments.txt, import, etc"" or you can also mention the file they might be present as keywords.
               """)
        print("a: "+a+"\n b: ")
        b=self.retrieve_results(a)
        print(b)
        c = llm(f"""
                go through the output: {b}, List of files: {self.fnames}
                Do you think the above data contains correct answer for the given query somewhere (INCLUDING "metadata") or is related?
                query: {query}
                if yes, then type "yes" with the reason else "no"
                """)
        print("c: "+c)
        words = c.lower().split()
        if words[0] == "yes" or max_count==0:
            return str(b)+c
        else:
            reprompt=reprompt+a+" "+str(b)
            return self.chain(query,reprompt,max_count-1)



    def chat_data(self, query):
        chat_history = list(self.MyQueue.queue)
        template = f"""
        - refer to the following data: {self.chain(query)}.
        - List of files: {self.fnames}
        - 'source' is the file name with location. 'page_content' is the content of the file. carefully go through the page_content, metadata, etc and understand everything.
        - also consider previous conversation : {chat_history}
        - Answer the given Query using the data provided above.
        Query: {query}
        - Answer should be as precise as possible.
        - Also mention the file name (or source) if you think it is related to the answer.
        """
        result=llm(template)
        self.add_to_queue(result)
        return result


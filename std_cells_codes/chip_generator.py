import os
import glob
import re
import shutil
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

# Constants
FAISS_INDEX_PATH = "faiss_mag_index"
GENERATED_MAG_DIR = "generated_mag"

class MagicLayoutGenerator:
    """
    A robust RAG generator using targeted preprocessing, hierarchical generation,
    and an optimization loop.
    """
    def __init__(self, mag_files_directory: str):
        print("🚀 Initializing MagicLayoutGenerator...")
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("🔴 GOOGLE_API_KEY not found.")

        # Shared LLM instance
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.0)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        self._load_or_create_vector_store(mag_files_directory)

        # --- Synthesis Chain ---
        synthesis_prompt = PromptTemplate.from_template(
            """You are a Magic VLSI expert. Your task is to generate a .mag file for the requested QUESTION.
            - Analyze the provided CONTEXTS, which are existing .mag files.
            - If the QUESTION is for a complex cell (like a MUX), you MUST use the provided component contexts to build the new layout by instantiating them with the 'use' command and adding routing.
            - If the QUESTION is for a simple gate, REPLICATE the best context and RENAME its labels.
            - Your output MUST BE ONLY the raw .mag file content.
            CONTEXTS: {context}
            QUESTION: {question}
            ANSWER (A complete .mag file):"""
        )
        self.synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()

        # --- Optimization Chain ---
        optimization_prompt = PromptTemplate.from_template(
            """You are an expert physical design engineer. You previously created the layout in 'EXISTING_LAYOUT' based on the 'ORIGINAL_REQUEST'.
            The user now has a new 'MODIFICATION_REQUEST'.
            Your task is to generate a new, improved .mag file that incorporates the user's feedback.
            Focus on implementing the requested changes while maintaining a valid layout.
            Your output MUST BE ONLY the new, raw .mag file content.
            
            ORIGINAL_REQUEST: {original_query}
            EXISTING_LAYOUT:
            {original_mag_content}
            
            MODIFICATION_REQUEST:
            {optimization_query}
            
            ANSWER (The new, improved .mag file):"""
        )
        self.optimization_chain = optimization_prompt | self.llm | StrOutputParser()
        print("\n🎉 MagicLayoutGenerator initialized successfully!")

    def _preprocess_mag_file(self, file_path: str):
        with open(file_path, 'r') as f: content = f.read()
        cell_name = os.path.splitext(os.path.basename(file_path))[0]
        keyword_part = f"cell name is {cell_name}. function is {cell_name}. " * 5
        description = f"Magic layout for standard cell. {keyword_part}"
        match = re.search(r"#\s*this is\s+(.*)", content, re.IGNORECASE)
        if match:
            description += f"It is described as a {match.group(1).strip()}. "
        return Document(page_content=description + content, metadata={"source": file_path})

    def _load_or_create_vector_store(self, path: str):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        
        print("🔍 Creating new vector store with targeted preprocessing...")
        mag_files = glob.glob(os.path.join(path, '**', '*.mag'), recursive=True)
        if not mag_files: raise FileNotFoundError(f"🔴 No .mag files found in '{path}'.")
        
        documents = [self._preprocess_mag_file(file_path) for file_path in mag_files]
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(FAISS_INDEX_PATH)
        print("✅ New vector store saved.")
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def _parse_dependencies(self, mag_content: str) -> set:
        return set(re.findall(r"^\s*use\s+([\w\d_]+)", mag_content, re.MULTILINE))

    def generate_and_save(self, initial_query: str, top_level_filename: str):
        """
        A generator function that yields the path of each generated file,
        allowing for step-by-step visualization.
        """
        os.makedirs(GENERATED_MAG_DIR, exist_ok=True)
        top_level_cell_name = os.path.splitext(top_level_filename)[0]
        generation_queue = [(initial_query, top_level_cell_name)]
        completed_cells = set()

        while generation_queue:
            current_query, current_cell_name = generation_queue.pop(0)
            if current_cell_name in completed_cells: continue
            
            print(f"\n▶️ Generating cell: '{current_cell_name}'")
            
            retrieved_docs = self.retriever.get_relevant_documents(current_query)
            if not retrieved_docs: continue

            context_str = "\n".join([f"--- CONTEXT: {os.path.basename(doc.metadata.get('source'))} ---\n{doc.page_content}" for doc in retrieved_docs])
            
            mag_content = self.synthesis_chain.invoke({"context": context_str, "question": current_query})
            if not mag_content or not mag_content.strip().startswith("magic"): continue

            file_path = os.path.join(GENERATED_MAG_DIR, f"{current_cell_name}.mag")
            with open(file_path, "w") as f: f.write(mag_content)
            
            completed_cells.add(current_cell_name)
            yield file_path # Yield the path for visualization

            dependencies = self._parse_dependencies(mag_content)
            for dep_cell_name in dependencies:
                if dep_cell_name not in completed_cells:
                    generation_queue.append((f"a {dep_cell_name.replace('_', ' ')} layout", dep_cell_name))

    def optimize_layout(self, original_query: str, original_mag_content: str, optimization_query: str) -> str:
        """
        Takes an existing layout and an optimization query to generate a new, improved layout.
        """
        return self.optimization_chain.invoke({
            "original_query": original_query,
            "original_mag_content": original_mag_content,
            "optimization_query": optimization_query
        })
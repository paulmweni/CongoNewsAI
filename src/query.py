"""Retrieval-augmented question answering for the News AI pipeline.

Exposes :class:`NewsQA`, which retrieves relevant article chunks from the
Chroma vector store and asks an OpenAI chat model to answer the user's
question, grounded in those sources.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from typing import List, Optional
import os

class NewsQA:
    """Answers questions about the news corpus using retrieval + an LLM."""

    def __init__(self,
                 persist_directory: str = "chroma_db",
                collection_name: str = "news",
                model: str = "gpt-4o-mini",
                api_key: Optional[str] = None,

):
        """Initialise the vector store, embedding model, and OpenAI client.

        Args:
            persist_directory: Directory of the persisted Chroma database.
            collection_name: Name of the Chroma collection to query.
            model: OpenAI chat model used to generate answers.
            api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
                environment variable when not provided.
        """
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function
        )

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

        self.model = model 

        self.prompt_system = """
Vous êtes un historien-journaliste qui analyse les actualités et les tendances en synthétisant les informations provenant d'articles de presse. Votre rôle est de combiner intelligemment les données de plusieurs sources pour offrir une compréhension claire et globale d'un sujet, tout en restant rigoureux sur les sources.

Vous recevrez :
- Une requête en français
- Plusieurs articles de presse avec leurs métadonnées (titre, date, URL, source, auteur si disponible)

Votre réponse doit :
- Être naturelle et accessible — comme si vous expliquiez le sujet à quelqu'un d'intelligent mais non-spécialiste
- Tisser les informations des articles ensemble pour créer une narrative cohérente plutôt que de les présenter séparément
- Dégager les tendances globales, les points communs et les désaccords entre les sources
- Rester strict sur la traçabilité : systématiquement citer vos sources

**Format des citations :**
- Mentionner le titre de l'article et la date entre parenthèses lors du premier point de référence
- Ajouter l'URL à la fin si vous la mentionnez
- Exemple : "Selon le rapport de RFI (15 mars 2024), la situation s'aggrave..."

**Structure attendue :**
- Pas de présentation rigide article par article
- Plutôt : synthèse thématique où vous combinez les informations de plusieurs sources
- Commencez par une observation générale, puis soutenez-la avec des citations précises
- Signalez les contradictions ou divergences entre sources si pertinent
- Terminez avec une perspective : ce qui se dessine, les questions ouvertes

**Tone :**
- Conversationnel mais rigoureux
- Analytique sans être pédant
- Engageant — comme si vous racontiez une histoire vérifiée
- Honnête sur les limites : "Les sources ne disent pas..." ou "On ne sait pas encore..."

**À faire :**
- Combiner et synthétiser (pas lister)
- Contextualiser les informations
- Citer précisément vos sources à chaque affirmation factuelle

**À éviter :**
- Présenter les articles un par un
- Une formalité excessive
- De vagues références ("selon certains")
"""

    def _compose_prompt(self, query: str, documents: List[str]) -> str:
        """Format the query and retrieved documents into a user prompt.

        Args:
            query: The user's natural-language question.
            documents: Retrieved Documents with text and metadata.

        Returns:
            A prompt string combining the query and formatted article excerpts.
        """
        formatted_documents = []
        for doc in documents:
            text = doc.page_content.strip()
            metadata = doc.metadata

            title = metadata.get('title')
            date = metadata.get('date')
            url = metadata.get('url')
            
            parts = [f"**ARTICLE {title.upper()}**"]
            if date:
                parts.append(f"**{date.strip()}**")
    
            if url:
                parts.append(f"**{url.strip()}**")

            parts.append(f"\n{text}\n")
            parts.append("---")

            formatted_documents.append('\n'.join(parts))

        return f"Requête : {query}\nArticles : {formatted_documents}"

    def answer_query(self, query: str, k: int = 6) -> str:
        """Answer a question using retrieved articles and the LLM.

        Args:
            query: The user's natural-language question.
            k: Number of most-similar chunks to retrieve as context.

        Returns:
            The model's answer, or a fallback message if nothing relevant is found.
        """
        results = self.vectorstore.similarity_search(query, k=k)

        if not results:
            return "Pas assez de donner pour repondre a la question"

        prompt = self._compose_prompt(query, results)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()
    
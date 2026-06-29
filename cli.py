"""Command-line interface for the News AI assistant.

Provides an ``ask`` command that answers a natural-language question about the
news corpus using the :class:`~src.query.NewsQA` retrieval-augmented pipeline.

Example:
    python cli.py ask "Comment la situation avec les M23 a-t-elle évolué ?"
"""

from src.query import NewsQA
import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()
@app.callback()

def main():
    """CLI de questions-réponses en lien avec la politique congolaise."""

@app.command()
def ask(question: str):
    """Ask a question and print the assistant's grounded answer."""
    qa = NewsQA()
    response = qa.answer_query(question)
    typer.echo(response)

if __name__ == "__main__":
    app()
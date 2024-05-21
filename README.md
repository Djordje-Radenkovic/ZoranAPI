# Retrieval Augmented Generation Model

Implemented a model which uses embedded information from an external code repository as truth source.

## File Structure
- **proof_of_concept.ipynb** - full workflow implemented, useful to understand.
- **generate_data.py** - generates and saves vector embeddings of an external codebase - run only if codebase updates or you want to change codebase.
- **vector_store** - contains the vector embeddings
- **app.py** - runs the flask pipeline to ask questions about codebase
- **templates** - HTML for the website
- **requirements.txt** - dependencies

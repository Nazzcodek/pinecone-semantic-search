# Semantic Search Application
This is a simple application that demonstrates semantic search using Pinecone and Cohere. It allows users to enter a query and retrieve the most relevant results from a pre-indexed set of questions.

## Installation

1. Clone the repository:
`https://github.com/Nazzcodek/pinecone-semantic-search.git`
2. Install the required dependencies:
`pip install -r requirements.txt`
3. Set up your API keys:

- Get a Cohere API key from the [Cohere website](https://www.cohere.ai/).
- Get a Pinecone API key from the [Pinecone website](https://www.pinecone.io/).

4. Update the environment variables:

- Set the `COHERE_API_KEY` environment variable with your Cohere API key.
- Set the `PINECONE_API_KEY` environment variable with your Pinecone API key.

## Usage

1. Run the application:
`streamlit run search.py`
2. Enter a query in the input field and click the "Search" button.
3. The application will retrieve and display the top search results based on the query.

## Project Ideas

The application uses a pre-defined list of project ideas for indexing and searching. You can customize the list by modifying the `project_ideas` variable in the script.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.





from langchain_core.prompts import ChatPromptTemplate

from llm.llm_services import LANGUAGE, get_llm

# --- Initialize LLM Client ---
# Using a specific LLM instance for these quick, utility tasks
LLM_CLIENT = get_llm()


def translate_name(text: str) -> str:
    """
    Uses the LLM to translate a name or entity from its source language
    to the target language (e.g., Greek), useful for database fallback.

    Args:
        text (str): The name to translate (e.g., 'LeBron James').
    Returns:
        str: The translated name (e.g., 'Λεμπρόν Τζέιμς').
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert linguistic assistant. Your task is to provide the most common and accurate translation 
        of a sports entity's proper noun into the specified target language. 
        You MUST ONLY return the translated name, with no extra commentary or characters.

        Target Language: {target_lang}
        Name to Translate: {text}
        
        Example: 
        If text is 'LeBron James' and target_lang is 'Greek', you return 'Λεμπρόν Τζέιμς'.
        If text is 'Messi' and target_lang is 'Greek', you return 'Μέσι'.
        """
    )
    chain = prompt | LLM_CLIENT
    try:
        # We strip the result to ensure only the name is returned
        result = chain.invoke({"text": text, "target_lang": LANGUAGE}).content.strip()
        print(f"✅ Translated '{text}' to '{result}' for database lookup.")
        return result
    except Exception as e:
        print(f"❌ Error during translation: {e}")
        return text  # Return original text on failure


def improve_vector_query(original_query: str) -> str:
    """
    Uses the LLM to refine a conversational or ambiguous natural language query
    into concise, optimized search keywords for vector store retrieval.

    Args:
        original_query (str): The user's original conversational query.

    Returns:
        str: An optimized, keyword-rich query string.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are a search query optimization engine for a vector database of sports articles. 
        Your goal is to extract the most critical keywords and concepts from the user's conversational query 
        and restructure them into a concise, keyword-rich string optimized for semantic search. 
        You MUST ONLY return the optimized query string, with no extra commentary.

        Original Query: {original_query}
        
        Example: 
        If the query is 'What is the news on Messi's last game and his injury status?'
        You should return: 'Messi last game summary injury status latest news'
        """
    )
    chain = prompt | LLM_CLIENT
    try:
        # We strip the result to ensure only the optimized query is returned
        result = chain.invoke({"original_query": original_query}).content.strip()
        print(f"✅ Optimized query: '{result}'")
        return result
    except Exception as e:
        print(f"❌ Error during query improvement: {e}")
        return original_query  # Return original query on failure


# --- Optional main execution block for testing ---
if __name__ == "__main__":
    print(f"Running query processors in language: {LANGUAGE}")

    # Test 1: Translation
    english_name = "Stephen Curry"
    greek_translation = translate_name(english_name, target_lang="Greek")
    print(f"Translation result: {english_name} -> {greek_translation}\n")

    # Test 2: Query Improvement
    vague_query = "Can you tell me how the new coach is doing at the Lakers?"
    optimized = improve_vector_query(vague_query)
    print(f"Improvement result: {optimized}")

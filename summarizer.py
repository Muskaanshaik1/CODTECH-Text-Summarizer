import re
from collections import defaultdict
from heapq import nlargest
import sys

# Ensure NLTK data is available. If not, you might need to download it:
# import nltk
# nltk.download('punkt') # For sentence tokenization
# nltk.download('stopwords') # For common words to ignore

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def summarize_text(text, num_sentences=5):
    """
    Summarizes the given text by extracting the most important sentences.

    Args:
        text (str): The input article or lengthy text.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: A concise summary of the input text.
    """
    if not text:
        return "Input text cannot be empty."

    # Clean the text: remove special characters and extra spaces
    # Using regex to replace non-alphanumeric characters (except spaces and periods)
    # and then multiple spaces with a single space.
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Tokenize the text into sentences
    sentences = sent_tokenize(cleaned_text)
    if not sentences:
        return "Could not tokenize sentences. Please check input text format."

    # Tokenize words and remove stopwords
    # Using a set for faster lookup of stopwords
    stop_words = set(stopwords.words('english'))
    word_frequencies = defaultdict(int)

    # Calculate word frequencies, ignoring stopwords and making words lowercase
    for word in word_tokenize(cleaned_text):
        word = word.lower()
        if word not in stop_words and word.isalpha(): # Only consider alphabetic words
            word_frequencies[word] += 1

    if not word_frequencies:
        return "No significant words found to create a summary."

    # Calculate maximum frequency to normalize
    max_frequency = max(word_frequencies.values())
    if max_frequency == 0: # Avoid division by zero if all frequencies are zero (unlikely but safe)
        return "Could not determine word frequencies for summarization."

    # Normalize word frequencies
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Score sentences based on word frequencies
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[i] += word_frequencies[word]

    # Get the top 'num_sentences' sentences based on their scores
    # nlargest returns a list of tuples (score, index)
    # We want the original sentence order, so we sort by index after getting the top scores
    summarized_sentences_indices = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Sort the selected sentence indices to maintain original order
    sorted_summarized_indices = sorted(summarized_sentences_indices)

    # Reconstruct the summary
    final_summary = [sentences[idx] for idx in sorted_summarized_indices]

    return ' '.join(final_summary)

if __name__ == "__main__":
    print("--- Text Summarization Tool ---")
    print("Enter the article text below. Press Enter twice to finish input.")
    print("-------------------------------")

    # Read multiline input from the user
    lines = []
    while True:
        try:
            line = input()
            if not line: # Empty line signifies end of input
                break
            lines.append(line)
        except EOFError: # Handle EOF (Ctrl+D or Ctrl+Z) for non-interactive input
            break

    article_text = "\n".join(lines)

    if not article_text.strip():
        print("\nNo text provided. Exiting.")
    else:
        # You can adjust the number of sentences in the summary here
        summary = summarize_text(article_text, num_sentences=3)
        print("\n--- Original Text ---")
        print(article_text)
        print("\n--- Generated Summary ---")
        print(summary)
        print("-----------------------")

    # Instructions for NLTK download if needed
    print("\nNote: If you encounter errors related to NLTK data (e.g., 'punkt' or 'stopwords'),")
    print("you might need to download them by running these lines once in a Python interpreter:")
    print("import nltk")
    print("nltk.download('punkt')")
    print("nltk.download('stopwords')")

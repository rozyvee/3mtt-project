import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

load_dotenv(find_dotenv())

# Set up Azure OpenAI credentials
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    api_key=azure_openai_api_key,
    azure_endpoint=azure_openai_endpoint,
    azure_deployment="gpt-35-turbo",
    api_version="2023-05-15",
    temperature=0.7
)


def generate_lullaby(forest, title, story, language):
    # Generate the original story
    story_prompt = PromptTemplate.from_template("""
    As a children's book author, write a short story about a young person's journey.
    Setting: {forest}
    Title: {title}
    Base story idea: {story}
    
    Write an engaging story suitable for children aged 8-12 years old.
    """)
    
    story_text = story_prompt.format(
        forest=forest,
        title=title,
        story=story
    )
    
    original_story = llm.invoke(story_text).content

    # Translate the story
    translation_prompt = PromptTemplate.from_template("""
    Translate the following story into {language}:
    
    {original_story}
    """)
    
    translation_text = translation_prompt.format(
        language=language,
        original_story=original_story
    )
    
    translated_story = llm.invoke(translation_text).content

    return {
        "original_story": original_story,
        "translated_story": translated_story
    }
    
def main():
    st.set_page_config(page_title="Lullaby Generator", page_icon=":bird:", layout="centered")
    st.title('Let AI write a lullaby for you...')
    st.header('Enter the details below to generate a lullaby.')
    
    forest_input = st.text_input('Enter the forest name:')
    title_input = st.text_input('Enter the title of the book:')
    story_input = st.text_area('Enter the story:')
    language_input = st.text_input('Enter the language for translation:')
    
    submit_button = st.button('Generate Lullaby')
    
    if forest_input and title_input and story_input and language_input:
        if submit_button:
            with st.spinner('Generating Lullaby...'):
                response = generate_lullaby(forest=forest_input, 
                                            title=title_input, 
                                            story=story_input, 
                                            language=language_input)
                with st.expander('See Story'):
                    st.write(response['original_story'])
                with st.expander('See Story Update'):
                    st.write(response['translated_story'])
                st.success('Lullaby generated successfully!')

if __name__ == "__main__":
    main()
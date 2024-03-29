import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Lyzr Interview Preparation Agent",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Interview Preparation Agent")
st.markdown("### Welcome to the Lyzr Interview Preparation Agent!")

query=st.text_area("Enter your Job Description: ")

open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

def interview_questions(query):

    interview_agent = Agent(
            role="toxicity expert",
            prompt_persona=f"You are an Expert INTERVIEW COACH. Your task is to FORMULATE 30 interview questions that are DIRECTLY TAILORED to a {query}."
        )

    prompt=f"""Your task is to FORMULATE 30 interview questions that are DIRECTLY TAILORED to a {query}.
        1.ANALYZE the job description CAREFULLY, identifying KEY SKILLS, RESPONSIBILITIES, and QUALIFICATIONS required for the role.
        
        2. DEVELOP a list of questions that ADDRESS each of these key areas, ensuring you probe into the candidate's RELEVANT EXPERIENCE and EXPERTISE.
        
        3. CREATE behavioral interview questions to ASSESS how candidates have previously HANDLED situations similar to those they might encounter in the target role.
        
        4. FORMULATE questions that allow candidates to DEMONSTRATE their problem-solving abilities and how they align with the company's VALUES and CULTURE.
        
        5. ENSURE that your questions also invite candidates to SHARE their professional aspirations and how they see themselves CONTRIBUTING to the company's GROWTH and SUCCESS.
        
        6. STRUCTURE your questions in a way that they build upon each other, creating a COHERENT and ENGAGING interview flow.
        
        You MUST craft these questions with the intention of GAUGING both technical competencies and SOFT SKILLS.
        
        I'm going to tip $300K for a BETTER SOLUTION!
        
        Now Take a Deep Breath."""

    question_task = Task(
        name="Maths Problem Solver",
        model=open_ai_text_completion_model,
        agent=interview_agent,
        instructions=prompt,
    )

    output = LinearSyncPipeline(
        name="Question Pipline",
        completion_message="pipeline completed",
        tasks=[
              question_task
        ],
    ).run()

    answer = output[0]['task_output']

    return answer

if st.button("Generate"):
    solution = interview_questions(query)
    st.markdown(solution)


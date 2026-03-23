from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Arvind Krishna (born November 23, 1962)[2] is an Indian-American business executive, and the chairman and CEO of IBM. He has been CEO of IBM since April 2020 and chairman since January 2021.[3][4] Krishna began his career at IBM in 1990, at its Thomas J. Watson Research Center,[5] and was promoted to senior vice president in 2015, managing IBM Cloud & Cognitive Software and IBM Research divisions. He was a principal architect of the acquisition of Red Hat, the largest acquisition in the company's history.[6][7]

Early life and education
Krishna was born in a Telugu family in West Godavari District, Andhra Pradesh, India.[1][8][9] His father, Major General Vinod Krishna, was an army officer who served in the Indian Army and his mother, Aarathi Krishna, worked for the welfare of Army widows.[10][11] Krishna studied at Stanes Anglo Indian Higher Secondary School in Coonoor, Tamil Nadu, and at St Joseph's Academy, Dehradun.[12]

Krishna received a Bachelor of Technology degree in electrical engineering from Indian Institute of Technology, Kanpur in 1985 and a Doctor of Philosophy in electrical engineering from the University of Illinois Urbana-Champaign in 1991.[13][14][15][16]
    """

    summary_template = """
    Given the info {information} I want you to create:
    1. A summary
    2. Two interesting facts
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm=  ChatOpenAI(temperature=0, model="gpt-5")
    llm=  ChatOllama(temperature=0, model="gpt-oss:20b")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()

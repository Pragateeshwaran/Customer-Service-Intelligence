from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

class HF_LLM:
    def __init__(self, model_name, llm):
        self.llm = llm  # LangChain ChatGroq object
        self.model_name = model_name

    def generate_response(self, transcripts, emotions, system_input="You are a Customer Service expert!"):
        example = "Communication: z/10 Resolution: y/10 Emotion Handling: x/10. So, the overall Customer Satisfaction Index can be calculated as the average of these three scores, which is approximately x+y+z/10."
        user_input = f"""
        I will provide you with the transcripts of a customer service call. I will also provide you the tone of the voices at each timestamp.('a': Anger 'h': Happy 'n': Neutral) You have to analyse both and come up with a Customer Satisfaction Index. You should also give reason why..
<Transcripts of the talks>
{transcripts}<Transcripts of the talks>
<Tone and emotion of the voice>
{emotions}<Tone and emotion of the voice>
<Example>
{example}<Example>
        """

        prompt_template = PromptTemplate(
            input_variables=["system_input", "user_input"],
            template="### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"
        )
        
        chain = prompt_template | self.llm
        response = chain.invoke({"system_input": system_input, "user_input": user_input})
        return response.content.split("### Assistant:\n")[-1]

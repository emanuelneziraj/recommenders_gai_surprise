# import the OpenAI Python library for calling the OpenAI API
import openai

class OpenAI:
    def __init__(self, model, api_key):
        self.model = model
        openai.api_key = api_key

    def request(self, message):
        try:
            # Reference:
            # https://platform.openai.com/docs/api-reference/chat/create
            #
            response = openai.chat.completions.create(
                model=self.model, # gpt-3.5-turbo / gpt-4-turbo
                messages=[
                    # experiment 1
                    {"role": "system", "content": "Given a user, as a Recommender System, please provide only the top 50 recommendations before 09/2018."},
                    # experiment 2 and 3
                    #{"role": "system", "content": "Given a user, act like a Recommender System."},
                    {"role": "user", "content": message}
                ],
                temperature=0,
                max_tokens=750,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        except openai.BadRequestError as e:
            print(f"Error: {e}")
            return ""

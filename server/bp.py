import google.generativeai as genai

genai.configure(api_key="AIzaSyBWL00-84wRaC1QyaVJs42ee3jCY7KBfIE")

for model in genai.list_models():
    print(model.name)
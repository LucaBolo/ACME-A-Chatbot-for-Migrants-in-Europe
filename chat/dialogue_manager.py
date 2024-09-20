from flask import Blueprint
from flask import request
import os
import replicate
import re
import json

from queue import Queue
from chat.new_controller import Controller
from chat.argumentation import ArgumentationManager

dialogue_blueprint = Blueprint('dialogue_manager', __name__)
arg_manager = ArgumentationManager()
controller = Controller(Queue(), None)

# Reads the prompt for Llama from a txt file
def get_system_prompt():
    with open('prompt.txt') as f:
        prompt = f.read()
    return prompt


# Returns a dictionary where the keys are the nodes and the values are the first sentence of each node, reading it all from the questions JSON file
def get_nodes_questions_dict():
    # Opening JSON file
    f = open('./db/questions.json')

    # returns JSON object as a dictionary
    data = json.load(f)

    # Creating the dictionary (key = node name, value = JSON obj with class, question and sentences
    result = {}

    # Iterating through the data
    for i in data:
        # We only need to identify argumentative nodes -> filtering out the reply nodes
        if 'questions' in i.keys():
            result[i['node'].lower()] = i['sentences'][0]

    # Adding the Unidentified option
    result['unidentified'] = 'unidentified'

    return result


# Identifies the argumentative node using Llama2 (used when the user input is not covered by the sentences in the JSON file)
def get_node_from_llama2(sentence: str):
    # Reading the prompt from the txt file
    system_prompt = get_system_prompt()

    # Initializing the API key
    os.environ["REPLICATE_API_TOKEN"] = "r8_bIcAUWQBzTamWWrEcClxnx0Ndznp0Mq1bqThu"

    # Creating the input for Llama2
    input = {
        "top_p": 1,
        "prompt": sentence,
        "temperature": 0.5,
        "system_prompt": system_prompt,
        "max_new_tokens": 500
    }

    # The output is word for word, so it creates a list
    nodes = []
    for event in replicate.stream(
            "meta/llama-2-70b-chat",
            input=input
    ):
        nodes.append(str(event).strip())

    # Stitching together the output and creating a string
    node = ''
    for n in nodes:
        node += n

    # Removing the junk (sometimes the previous nodes remain in the output)
    node = re.sub("'", "", node)
    node = node.split('\n')[-1]
    node = node.strip()

    # Retrieving the dictionary that relates the nodes to their corresponding sentences
    dict = get_nodes_questions_dict()

    if node.lower() in dict.keys():
        return dict[node.lower()]
    else:
        return 'Unidentified'


@dialogue_blueprint.route("/", methods=("GET",))
def start_conversation():
    
    initial_msg ={"data": """Hello, I will help you decide whether you can apply for some form of protection on the italian territory. 
                                You may share with me any information about yourself and your story. All the information will never leave your device.\n
                                Start by telling me something about you and then I will ask you some questions. Tell me if you don't understand the question!""",
            }

    controller.start_conversation(initial_msg)

    nodes_info = {
            "uomo": "Male",
            "donna": "Female",
            "woodo": "Victim of Voodoo ritual",
            "livIstrBasso": "Low education level",
            "condEconPr": "Precarious economic conditions",
            "condViaggio": "Issues during the journey",
            "minacce": "Threats",
            "provNigeria": "Nigerian",
            "violenza": "Victim of violence",
            "madame": "Victim of human trafficking",
            "orientamentoOmos": "Homosexuality",
            "nonProtezioneStato": "Protection denied in home country",
            "lavoroAttuale": "Has a job",
            "documentazioneLavorativa": "Has employment documentation",
            "vulnerabilita": "Vulnerable subject"
        }


    initial_msg['nodes'] = [{"id": id, "name": name, "status":0} for id,name in nodes_info.items()]


    return initial_msg



@dialogue_blueprint.route("/sentences", methods=("GET",))
def get_kb_sentences():
    return {"data": [sentence for sentence in arg_manager.arg_graph.get_arg_sentences()]}


@dialogue_blueprint.route("/chat", methods=("GET",))
def chat():
    
    usr_msg, usr_intent = controller.closest_embeddings(request.args["message"])
    reply = None
    
    if usr_intent == 'yes':
        # user responded affirmatively to question
        # we look for a sentence in the node containing the question with positive class
        usr_msg_to_send = arg_manager.arg_graph.get_sentence_corresponding_question(usr_msg[0], 'p')

        if usr_msg_to_send is None:
            usr_msg_to_send = get_node_from_llama2(str(usr_msg))
            print("FOUND WITH LLAMA")
            print(usr_msg_to_send)
            print("END LLAMA")
            if usr_msg_to_send.lower() == 'unidentified':
                reply = "I didn't understand your answer, could you repeat?"
        usr_msg = [usr_msg_to_send]

    elif usr_intent == 'no':

        # user responded negatively to question
        # we look for a sentence in the node containing the question with negative class

        usr_msg_to_send = arg_manager.arg_graph.get_sentence_corresponding_question(usr_msg[0], 'n')

        if usr_msg_to_send is None:
            usr_msg_to_send = get_node_from_llama2(str(usr_msg))
            print("FOUND WITH LLAMA")
            print(usr_msg_to_send)
            print("END LLAMA")
            if usr_msg_to_send.lower() == 'unidentified':
                reply = "I didn't understand your answer, could you repeat?"
        usr_msg = [usr_msg_to_send]

    print("message before choose_reply", usr_msg)
    
    if reply == None:
        reply = arg_manager.choose_reply(usr_msg)

    answer = {"data": reply, 
              "history_args": arg_manager.history_args, 
              "history_replies": arg_manager.history_replies,
              "activated_node": arg_manager.history_args[-1]
              }

    controller.post_chat_process(answer)

    return answer


@dialogue_blueprint.route("/close", methods=("GET",))
def clear_history():
    res = {"data": "QUIT"}
    arg_manager.clear()
    controller.stop_conversation(res)

    return res
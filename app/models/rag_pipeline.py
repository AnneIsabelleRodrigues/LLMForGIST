from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import ast
import re

from ..config import COLLECTION_NAME, EMBEDDING_MODEL
from ..constants import feature_descriptions

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

encoder = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# 1. RAG: Recupera documentos relevantes
# ============================================================
def retrieve_relevant_docs(client, query, top_k=3):
    query_vector = encoder.encode(query, convert_to_tensor=False).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    docs = [hit.payload["text"] for hit in results]
    return docs


# ============================================================
# 2. Parser do retorno do modelo
# ============================================================

def extract_list_from_text(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if not match:
        print("Nenhuma lista encontrada no texto.")
        return None

    list_str = match.group(0)

    try:
        parsed_list = ast.literal_eval(list_str)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            print("O conteúdo encontrado não é uma lista.")
            return None
    except Exception as e:
        print("Erro ao interpretar a lista:", e)
        try:
            list_str_fixed = re.sub(r"“|”", '"', list_str)
            return ast.literal_eval(list_str_fixed)
        except Exception as e2:
            print("Ainda não foi possível decodificar:", e2)
            return None


def extract_json_from_text(text):
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print("Nenhum bloco JSON encontrado no texto.")
        return None

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("Erro ao decodificar JSON:", e)
        try:
            json_str_fixed = re.sub(r',(\s*[\]}])', r'\1', json_str)
            return json.loads(json_str_fixed)
        except Exception as e2:
            print("Ainda não foi possível decodificar:", e2)
            return None


def parse_llm_answer(answer_content, valid_nodes, independent_nodes, current_node):
    try:
        answer_list = answer_content.split('<Answer>')[1].split('</Answer>')[0].split(',')
    except IndexError:
        print(f"AVISO: Não foi possível encontrar tags <Answer>...</Answer>. Tentando análise alternativa.")
        answer_list = []
        for node in valid_nodes:
            if node in answer_content and node != current_node:
                answer_list.append(node)

    cleaned_answer = []
    for node in answer_list:
        node = node.strip()

        if len(node) == 0:
            continue

        if node not in valid_nodes:
            print(f"  AVISO: '{node}' não é uma variável válida (ignorado)")
            continue

        if node in independent_nodes:
            print(f"  AVISO: '{node}' é um nó independente (ignorado)")
            continue

        if node == current_node:
            print(f"  AVISO: '{node}' é o próprio nó atual (ciclo ignorado)")
            continue

        cleaned_answer.append(node)

    print(f"  Efeitos identificados: {cleaned_answer if cleaned_answer else 'Nenhum'}")
    return cleaned_answer


# ============================================================
# 3. Execução BFS com LLM (descoberta causal)
# ============================================================
def llm_bfs_local(var_names_and_desc, independent_nodes, df, model, tokenizer, prompt_format):
    print("=" * 60)
    print("INICIANDO LLM-BFS LOCAL")
    print("=" * 60)

    system_instruction = (
        "You are a causal reasoning assistant helping an oncologist specializing in Gastrointestinal Stromal Tumor (GIST). "
        "You will receive a list of biomedical or clinical variables potentially related to GIST. "
        "Your goal is to identify, for each candidate variable, whether it is causally affected by another variable. "
        "\n\n"
        f"The target variable(s) are defined as interventions or exposures that are *not caused* by any other variable. "
        "Using causal reasoning based on medical domain knowledge, you must determine which variables are *directly* caused by them. "
        "\n\n"
        "For each variable you identify as being caused by the target, you **must provide a short causal justification** explaining *why* that relationship exists "
        "(e.g., physiological mechanism, clinical pathway, or logical dependency). "
        "If no evidence or plausible mechanism exists, state that the causal link is weak or absent. "
        #        "For each variable with causal affect, explain your reasoning with the format [CAUSE] → [EFFECT]: [Justification]\n"
        "\n\n"
        "Output format requirements:\n"
        "- Only include variables that have plausible causal links.\n"
        "- End your answer with a final summary line listing all effects separated by commas and between the tag <Answer>...</Answer> .\n"
    )

    nodes = list(var_names_and_desc.keys())
    unvisited_nodes = set(nodes)
    for indep in independent_nodes:
        if indep in unvisited_nodes:
            unvisited_nodes.remove(indep)

    predict_graph = {}
    message_history = []
    frontier = list(independent_nodes)

    level = 0
    while frontier:
        level += 1
        #        print(f"\n{'=' * 60}")
        #        print(f"NÍVEL {level}")
        #        print(f"{'=' * 60}")
        #        print(f"Nós neste nível: {frontier}")

        next_frontier = []

        for current_node in frontier:
            #            print(f"\n>>> Processando: {current_node}")

            prompt = (
                f"Given {', '.join(independent_nodes)} as root causes not affected by any variable, "
                f"and the causal relationships already found:\n"
            )

            for head, tails in predict_graph.items():
                if tails:
                    prompt += f"{head} causes {', '.join(tails)}.\n"

            prompt += (
                f"\nNow, analyze which of the following variables {nodes} are directly or indirectly caused by {current_node}. "
                f"For each link {current_node} → X, explain briefly the reasoning or mechanism. "
            )

            query_text = prompt + "\n\n" + prompt_format

            answer_content = generate_chat_answer(
                system_instruction=system_instruction,
                model=model,
                tokenizer=tokenizer,
                query=query_text,
                docs=[],
                history=message_history
            )

            #            print(f"Resposta do LLM:\n{answer_content}\n")

            message_history.append({"role": "user", "content": query_text})
            message_history.append({"role": "assistant", "content": answer_content})

            cleaned_answer = parse_llm_answer(answer_content, nodes, independent_nodes, current_node)
            predict_graph[current_node] = cleaned_answer

            for node in cleaned_answer:
                if node in unvisited_nodes:
                    unvisited_nodes.remove(node)
                    next_frontier.append(node)

        if not next_frontier and unvisited_nodes:
            #            print(f"\nNenhum novo nó encontrado no nível {level}.")
            #            print(f"Verificando nós não causados: {list(unvisited_nodes)}")
            new_indep = list(unvisited_nodes)
            #            print(f"⚙️  Reclassificando como independentes temporários: {new_indep}")
            next_frontier.extend(new_indep)
            independent_nodes.extend(new_indep)
            for node in new_indep:
                unvisited_nodes.remove(node)

        frontier = next_frontier

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    for head, tails in predict_graph.items():
        if tails:
            print(f"{head} → {', '.join(tails)}")

    return predict_graph


# ============================================================
# 4. Instruções auxiliares
# ============================================================

def causal_validation_instruction(variables, model, tokenizer, docs=None):
    system_instruction = (
        "You are a **Senior Expert in Causal Inference, Oncology, and Molecular Biology**, assisting a GIST oncologist. "
        "Your role is to rigorously **validate hypothesized causal links**.\n\n"

        "**INPUT:** You will receive a Python dictionary (`variables`) defining features and their descriptions, followed by a list of hypothesized causal relationships (KEY -> [LIST OF EFFECTS]).\n"

        "**TASK:** For EACH hypothesized causal link (Cause → Effect), perform a three-step validation:\n"
        "1. **Plausibility:** Evaluate the causal direction (Cause leads to Effect) based on established medical understanding of GIST, including known pathophysiological mechanisms, genetic drivers (e.g., KIT/PDGFRA mutations), or epidemiological evidence.\n"
        "2. **Justification:** Provide a concise, clear explanation (medical/biological/clinical) for why the link is plausible or, conversely, implausible.\n"
        "3. **Alternative:** If the link is weak, discuss a highly likely confounder, mediator, or an alternative, more plausible causal direction.\n\n"

        "**OUTPUT FORMAT REQUIREMENTS:** Your answer **MUST BE ONLY** a single, valid Python dictionary that maps the validated relationships to their justifications. "
        "The key must be the original hypothesized cause. The value must be a list of dictionaries, one for each effect.\n\n"

        "**STRICT FORMAT:**\n"
        "```python\n"
        "{\n"
        "  \"original_cause_variable\": [\n"
        "    { \"effect\": \"affected_variable\", \"plausibility\": \"High\" | \"Medium\" | \"Low\", \"justification\": \"A clear, concise explanation based on GIST biology.\" },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "```\n\n"

        "**BEGIN PYTHON DICTIONARY OUTPUT NOW. DO NOT INCLUDE ANY PREAMBLE OR ADDITIONAL TEXT.**"
    )

    query = f"Here are the features and their description, and the causal graph to validate:\n {variables}."

    if "mistral" in getattr(model.config, "name_or_path", "").lower():
        return generate_chat_answer(system_instruction, model, tokenizer, query, docs)
    elif "gemma" in getattr(model.config, "name_or_path", "").lower():
        return generate_gemma_answer_text_only(system_instruction, model, tokenizer, query, docs)
    else:
        return generate_llama_answer(system_instruction, model, tokenizer, query, docs)


def independent_validation_instruction(independent_nodes, model, tokenizer, docs=None):
    system_instruction = (
        "You are an **Expert Biostatistician and Oncologist specializing in GIST**. "
        "Your goal is to identify the **Root Causal Variables** (causally independent variables) "
        "from a provided list of medical features.\n\n"

        "**ROOT CAUSE DEFINITION:**\n"
        "A variable is considered a *Root Cause* if it is **not caused by any other variable** "
        "within the provided list. Typically, such variables include demographics, genetic drivers, "
        "or external exposures that precede biological or clinical outcomes.\n\n"

        "**INPUT:**\n"
        f"{independent_nodes}\n\n"

        "**TASK:**\n"
        "1. Analyze the variables using your domain knowledge in oncology, genetics, and biostatistics. Remember that variables like 'treatment_response', death and survival related variables can NEVER be independent.\n"
        "2. Identify only the variables that are most likely to be true *Root Causes* — that is, "
        "variables that do not depend on any others in the list.\n"
        "3. Do not include derived or dependent variables (e.g., tumor grade, response, or recurrence).\n"
        "4. The reasoning must be purely biological and clinical, not statistical.\n"
        "5. Do not repeat or list any variable twice.\n"
        "6. Return **no more than 10 variables**.\n\n"
        "7. You have to be reasonably. Remember that treatment_response, can only be caused by treatment.\n\n"

        "**OUTPUT FORMAT:**\n"
        "Return your answer as a single valid Python list containing only the string names of the identified variables.\n"
        "No explanation, no comments, no markdown fences.\n"
        "Example:\n"
        "[\"AGE\", \"KIT_mutation\", \"SEX\"]\n"
    )

    query = (
        "From the provided feature list, output only the unique, causally independent variables. "
        "Do not repeat names, and do not explain your reasoning. Return only a valid Python list."
    )

    if "mistral" in getattr(model.config, "name_or_path", "").lower():
        return generate_chat_answer(system_instruction, model, tokenizer, query, docs)
    elif "gemma" in getattr(model.config, "name_or_path", "").lower():
        return generate_gemma_answer_text_only(system_instruction, model, tokenizer, query, docs)
    else:
        return generate_llama_answer(system_instruction, model, tokenizer, query, docs)


def scm_function_generation_instruction(relations, model, tokenizer, docs=None):
    system_instruction = (
        "You are an expert in causal inference and mathematical modeling. "
        "You will receive a dictionary representing a causal graph (DAG), "
        "where each variable maps to a list of its direct causes (parents). "
        "For each variable, write a Python-like function definition that expresses "
        "how it depends on its parents, including an exogenous noise term u_i. "
        "If the variable has no parents, model it as an exogenous variable drawn from a distribution. "
        "Use realistic functional forms (linear or nonlinear) and ensure that each function name matches the variable name. "
        "Return the result as a valid JSON object mapping each variable to its function code as a string."
    )

    query = f"Here is the causal graph: {relations}. Define all structural equations f_i."

    if "mistral" in getattr(model.config, "name_or_path", "").lower():
        return generate_chat_answer(system_instruction, model, tokenizer, query, docs)
    elif "gemma" in getattr(model.config, "name_or_path", "").lower():
        return generate_gemma_answer_text_only(system_instruction, model, tokenizer, query, docs)
    else:
        return generate_llama_answer(system_instruction, model, tokenizer, query, docs)


def variable_expansion_instruction(idpdt, causal_rel, variables, model, tokenizer, query, docs):
    system_instruction = (
        "You are an expert assistant supporting an oncologist specializing in Gastrointestinal Stromal Tumor (GIST). "
        f"The target variable {idpdt} is an intervention/exposure that is not caused by any other variable. "
        f"You are given a set of directed causal relationships: {causal_rel} and a candidate variable list: {variables}. "
        "Task: Identify all variables from the candidate list that are direct or indirect descendants of "
        f"the target variable {idpdt}, based on the provided causal relationships. "
        "Requirements: Perform internal reasoning but respond only with valid JSON.\n"
        "Output format:\n"
        "{ \"affected\": [\"var1\", \"var2\"], \"explanation\": \"descendants of idpdt via provided edges\" }"
    )

    if "mistral" in getattr(model.config, "name_or_path", "").lower():
        return generate_chat_answer(system_instruction, model, tokenizer, query, docs)
    elif "gemma" in getattr(model.config, "name_or_path", "").lower():
        return generate_gemma_answer_text_only(system_instruction, model, tokenizer, query, docs)
    else:
        return generate_llama_answer(system_instruction, model, tokenizer, query, docs)


    # --- Formato de Prompt Otimizado para Qwen (ChatML) ---
    # É fundamental que este formato seja EXATAMENTE o que o modelo Qwen foi treinado.

    system_prompt = "Você é um assistente útil e preciso. Responda à pergunta do usuário estritamente baseado nas informações fornecidas no contexto."

    user_prompt = f"Contexto:\n{context}\n\nPergunta: {query}"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.6,
            top_p=0.8,
            #            top_k=10,
            max_new_tokens=4096
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def generate_gemma_answer_text_only(
        system_instruction,
        model,
        processor,
        query,
        docs=None,
        specific_document=None,
        history=None,
        max_new_tokens=1024
):
    user_content = []

    if docs and len(docs) > 0:
        rag_text = "\n\n".join(
            [f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(docs)]
        )
        user_content.append({
            "type": "text",
            "text": f"Documents retrieved from RAG:\n{rag_text}\n"
        })

    if specific_document:
        user_content.append({
            "type": "text",
            "text": f"Specific Document:\n{specific_document}\n"
        })

    user_content.append({
        "type": "text",
        "text": query
    })

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():

        generation = model.generate(
            **inputs,
            max_new_tokens=7800,
            do_sample=False
        )

    decoded = processor.decode(generation[0][input_len:], skip_special_tokens=True)

    #    print(f"Resposta do MedGemma: {decoded}")

    return decoded


def generate_llama_answer(
        system_instruction,
        model,
        processor,
        query,
        docs=None,
        specific_document=None,
        history=None,
        max_new_tokens=1024
):
    user_content = []

    if docs and len(docs) > 0:
        rag_text = "\n\n".join(
            [f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(docs)]
        )
        user_content.append({
            "type": "text",
            "text": f"Documents retrieved from RAG:\n{rag_text}\n"
        })

    if specific_document:
        user_content.append({
            "type": "text",
            "text": f"Reference Document:\n{specific_document}\n"
        })

    user_content.append({
        "type": "text",
        "text": query
    })

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction}]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    response = outputs[0][input_ids.shape[-1]:]
    decoded = tokenizer.decode(response, skip_special_tokens=True)

    return decoded

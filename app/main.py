from constants import feature_descriptions
from config import QDRANT_CLUSTER_URL, QDRANT_API_KEY
import os
import json
from models.dataset_loader import load_local_dataset, read_document
from models.base_model import load_base_model, load_gemma_model, load_llama_model
from models.fine_tune import fine_tune
from models.vector_db import init_qdrant, add_documents, load_documents_for_embedding, chunk_texts
from models.rag_pipeline import retrieve_relevant_docs, variable_discovery_instruction, \
    variable_initialization_instruction, llm_bfs_local, extract_list_from_text, extract_json_from_text, \
    independent_validation_instruction, causal_validation_instruction

if __name__ == '__main__':
    tokenizer, model = load_base_model(model_name="mistralai/Ministral-8B-Instruct-2410")
    tokenizer_gemma, model_gemma = load_gemma_model(model_name="google/medgemma-4b-it")

    # dataset = load_local_dataset("/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/data/dataset")
    #
    # model_ft = fine_tune(model, tokenizer, dataset)

    client = init_qdrant(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)

    #    texts = load_documents_for_embedding("/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/data/documents/")
    #    chunks = chunk_texts(texts, chunk_size=1024, chunk_overlap=100)
    #    add_documents(client, texts)
    #

    CASE_REPORTS_DIR = "/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/data/casereports/"
    OUTPUT_FILE = "/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/results/causal_verifications.json"

    results = []
    pdf_files = [
        f for f in os.listdir(CASE_REPORTS_DIR)
        if f.lower().endswith((".md", ".txt"))
    ]

    for pdf_name in pdf_files:
        pdf_path = os.path.join(CASE_REPORTS_DIR, pdf_name)
        print(f"\n===== PROCESSANDO {pdf_name} =====\n")

        # ---- 1. Ler o case report ----
        new_article = read_document(pdf_path)

        # ---- 2. Variable Discovery ----

        variablesdiscovered = variable_discovery_instruction(model, tokenizer, new_article)
        vd = extract_json_from_text(variablesdiscovered)

        # ---- 3. Initial Independent Variables ----
        independentvariables = variable_initialization_instruction(str(vd), model, tokenizer, query="", docs=None)
        independent_nodes = extract_list_from_text(independentvariables)

        print(independent_nodes)

        # ---- 4. Verification (RAG) ----
        query = (
            "From the provided feature list, output only the unique, causally independent variables. "
            "Do not repeat names, and do not explain your reasoning. Return only a valid Python list."
            f"{independent_nodes}"
        )

        rag = retrieve_relevant_docs(client, query)
        independent_nodes_verification = independent_validation_instruction(
            independent_nodes, model_gemma, tokenizer_gemma, docs=rag
        )
        independent_nodes_verification = extract_list_from_text(independent_nodes_verification)

        prompt_format = """
            Analyze the provided causal relationships carefully and output your result.

            Guidelines:
            - Do not repeat variables.
            - If the target variable does not cause any other variable, output exactly:
              <Answer>None</Answer>
            - Consider only *direct* causal effects (not correlations or indirect effects).
            - Apply biological and medical plausibility when deciding causal direction.
            - Respect temporal ordering and the established causal relationships provided above.
            - Do not include explanations or reasoning outside the <Answer> tag.

            Example valid responses:
            <Answer>tumor_size, stage_at_diagnosis</Answer>
            <Answer>None</Answer>
            """

        # ---- 5. Generate causal graph ----
        adj_matrix = llm_bfs_local(
            feature_descriptions,
            independent_nodes_verification,
            None,
            model,
            tokenizer,
            prompt_format
        )

        # ---- 6. Validate causal graph ----
        query = f"Here are the features and their description, and the causal graph to validate:\n {adj_matrix}."
        rag = retrieve_relevant_docs(client, query)

        causal_verification = causal_validation_instruction(
            adj_matrix, model_gemma, tokenizer_gemma, docs=rag
        )

        # ---- 7. Guardar no array ----
        results.append(causal_verification)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False)

    print(f"\n? Resultados salvos em: {OUTPUT_FILE}\n")




#  new_article = read_document("data/casereports/22045229.pdf")
   #  variablesdiscovered = variable_discovery_instruction(model, tokenizer, new_article)
   #
   #  print("\nVariaveis :\n", variablesdiscovered)
   #
   #  print(extract_json_from_text(variablesdiscovered))
   #
   #  independentvariables = variable_initialization_instruction(str(feature_descriptions), model, tokenizer, query="", docs=None)
   #
   #  print("\Independentes:\n", independentvariables)
   #
   #  independent_nodes = extract_list_from_text(independentvariables)
   #
   #  query = f"Analyze and verify statistical and clinical independence between the following medical features:\n  {feature_descriptions}"
   #
   #  #rag = retrieve_relevant_docs(client, query)
   #
   #  independent_nodes_verification = independent_validation_instruction(query, independent_nodes, model_gemma, tokenizer_gemma, docs=None)
   #
   #  print("gemma indpt:", independent_nodes_verification)
   #
   #  #    independent_nodes = ['patient_id', 'gender', 'race', 'source', 'sample_type', 'mutated_genes', 'variant_type', 'exon_number', 'primary_site', 'metastatic_site', 'tumor_grade', 'ped_ind', 'ethnic', 'msi_score', 'tmb_nonsynonymous', 'loh', 'molecular_and_ihc_data', 'collection_date', 'MutationEffect', 'existing_variant']
   #
   #  prompt_format = """
   #  Analyze the provided causal relationships carefully and output your result.
   #
   #  Guidelines:
   #  - Do not repeat variables.
   #  - If the target variable does not cause any other variable, output exactly:
   #    <Answer>None</Answer>
   #  - Consider only *direct* causal effects (not correlations or indirect effects).
   #  - Apply biological and medical plausibility when deciding causal direction.
   #  - Respect temporal ordering and the established causal relationships provided above.
   #  - Do not include explanations or reasoning outside the <Answer> tag.
   #
   #  Example valid responses:
   #  <Answer>tumor_size, stage_at_diagnosis</Answer>
   #  <Answer>None</Answer>
   #  """
   #
   #  adj_matrix = llm_bfs_local(feature_descriptions, independent_nodes, None, model, tokenizer, prompt_format)
   #
   #  causal_verification = causal_validation_instruction(adj_matrix, model_gemma, tokenizer_gemma, docs=None)
   #
   #  print(adj_matrix)
   #
   #  '''print(extract_json_from_text(variablesdiscovered))
   #
   #  query = ""
   # #rag = retrieve_relevant_docs(client, query)
   #
   #  #doc = read_document()
   #
   #  answer = variable_initialization_instruction(str(feature_descriptions), model, tokenizer, query, docs=None)
   #
   #  print("\Independentes:\n", answer)
   #
   #  #independent_nodes = extract_list_from_text(answer)
   #
   #  independent_nodes = ['patient_id', 'gender', 'race', 'source', 'sample_type', 'mutated_genes', 'variant_type', 'exon_number', 'primary_site', 'metastatic_site', 'tumor_grade', 'ped_ind', 'ethnic', 'msi_score', 'tmb_nonsynonymous', 'loh', 'molecular_and_ihc_data', 'collection_date', 'MutationEffect', 'existing_variant']
   #
   #
   #  prompt_format = """
   #  Analyze the provided causal relationships carefully and output your result.
   #
   #  Guidelines:
   #  - Do not repeat variables.
   #  - If the target variable does not cause any other variable, output exactly:
   #    <Answer>None</Answer>
   #  - Consider only *direct* causal effects (not correlations or indirect effects).
   #  - Apply biological and medical plausibility when deciding causal direction.
   #  - Respect temporal ordering and the established causal relationships provided above.
   #  - Do not include explanations or reasoning outside the <Answer> tag.
   #
   #  Example valid responses:
   #  <Answer>tumor_size, stage_at_diagnosis</Answer>
   #  <Answer>None</Answer>
   #  """
   #
   #  adj_matrix = llm_bfs_local(feature_descriptions, independent_nodes, None, model, tokenizer, prompt_format)
   #
   #  print(adj_matrix)'''

#    print("="*80)
#    print(f"MISTRAL BASE WITH GEMMA VALIDATION, WITH RAG")
#    print("-"*80)
#
#    new_article = read_document("/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/data/casereports/22045229.pdf")
#    variablesdiscovered = variable_discovery_instruction(model, tokenizer, new_article)
#
#    client = init_qdrant(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)
#
##    texts = load_documents_for_embedding("/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/data/documents/")
##    chunks = chunk_texts(texts, chunk_size=1024, chunk_overlap=100)
##    add_documents(client, texts)
#
#    print("-"*80)
#    print(f"variable discovery")
#    print("-"*80)
#
#    vd = extract_json_from_text(variablesdiscovered)
#    print(vd)
#
#    print("-"*80)
#    print(f"independent variables")
#    print("-"*80)
#
#    independentvariables = variable_initialization_instruction(str(vd), model, tokenizer, query="", docs=None)
#
#    independent_nodes = extract_list_from_text(independentvariables)
#
#    print(independent_nodes)
#
#    print("-"*80)
#    print(f"independent variables verification")
#    print("-"*80)
#
#    query = (
#        "From the provided feature list, output only the unique, causally independent variables. "
#        "Do not repeat names, and do not explain your reasoning. Return only a valid Python list."
#        f"{independent_nodes}"
#    )
#
#    rag = retrieve_relevant_docs(client, query)
#
#    independent_nodes_verification = independent_validation_instruction(independent_nodes, model_gemma, tokenizer_gemma, docs=rag)
#
#    independent_nodes_verification = extract_list_from_text(independent_nodes_verification)
#    print(independent_nodes_verification)
#
#
#    prompt_format = """
#    Analyze the provided causal relationships carefully and output your result.
#
#    Guidelines:
#    - Do not repeat variables.
#    - If the target variable does not cause any other variable, output exactly:
#      <Answer>None</Answer>
#    - Consider only *direct* causal effects (not correlations or indirect effects).
#    - Apply biological and medical plausibility when deciding causal direction.
#    - Respect temporal ordering and the established causal relationships provided above.
#    - Do not include explanations or reasoning outside the <Answer> tag.
#
#    Example valid responses:
#    <Answer>tumor_size, stage_at_diagnosis</Answer>
#    <Answer>None</Answer>
#    """
#
#    print("-"*80)
#    print(f"causal graph")
#    print("-"*80)
#
#    adj_matrix = llm_bfs_local(feature_descriptions, independent_nodes_verification, None, model, tokenizer, prompt_format)
#
#    query = f"Here are the features and their description, and the causal graph to validate:\n {adj_matrix}."
#
#    rag = retrieve_relevant_docs(client, query)
#
#    causal_verification = causal_validation_instruction(adj_matrix, model_gemma, tokenizer_gemma, docs=rag)
#
#    print(adj_matrix)
#
#    print("-"*80)
#    print(f"causal graph verification")
#    print("-"*80)
#
#    print(causal_verification)

from constants import feature_descriptions
from config import QDRANT_CLUSTER_URL, QDRANT_API_KEY
import json
from models.dataset_loader import load_local_dataset, read_document
from models.base_model import load_base_model, load_gemma_model, load_llama_model
from models.fine_tune import fine_tune
from models.vector_db import init_qdrant, add_documents, load_documents_for_embedding, chunk_texts
from models.rag_pipeline import retrieve_relevant_docs, variable_discovery_instruction, \
    variable_initialization_instruction, llm_bfs_local, extract_list_from_text, extract_json_from_text, \
    independent_validation_instruction, causal_validation_instruction

if __name__ == '__main__':

    tokenizer, model = load_base_model(model_name="mistralai/Ministral-8B-Instruct-2410")
    tokenizer_gemma, model_gemma = load_gemma_model(model_name="google/medgemma-4b-it")

    client = init_qdrant(url=QDRANT_CLUSTER_URL, api_key=QDRANT_API_KEY)

    CASE_REPORTS_DIR = "/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/data/casereports/"
    OUTPUT_FILE = "/home_cerberus/disk3/annecarvalho/code/SYNTHETICGIST/app/results/causal_verifications.json"

    results = []
    pdf_files = [
        f for f in os.listdir(CASE_REPORTS_DIR)
        if f.lower().endswith((".md", ".txt"))
    ]

    for pdf_name in pdf_files:
        pdf_path = os.path.join(CASE_REPORTS_DIR, pdf_name)
        print(f"\n===== PROCESSANDO {pdf_name} =====\n")

        # ---- 1. Ler o case report ----
        new_article = read_document(pdf_path)

        # ---- 2. Variable Discovery ----

        variablesdiscovered = variable_discovery_instruction(model, tokenizer, new_article)
        vd = extract_json_from_text(variablesdiscovered)

        # ---- 3. Initial Independent Variables ----
        independentvariables = variable_initialization_instruction(str(vd), model, tokenizer, query="", docs=None)
        independent_nodes = extract_list_from_text(independentvariables)

        print(independent_nodes)

        # ---- 4. Verification (RAG) ----
        query = (
            "From the provided feature list, output only the unique, causally independent variables. "
            "Do not repeat names, and do not explain your reasoning. Return only a valid Python list."
            f"{independent_nodes}"
        )

        rag = retrieve_relevant_docs(client, query)
        independent_nodes_verification = independent_validation_instruction(
            independent_nodes, model_gemma, tokenizer_gemma, docs=rag
        )
        independent_nodes_verification = extract_list_from_text(independent_nodes_verification)

        prompt_format = """
        Analyze the provided causal relationships carefully and output your result.

        Guidelines:
        - Do not repeat variables.
        - If the target variable does not cause any other variable, output exactly:
          <Answer>None</Answer>
        - Consider only *direct* causal effects (not correlations or indirect effects).
        - Apply biological and medical plausibility when deciding causal direction.
        - Respect temporal ordering and the established causal relationships provided above.
        - Do not include explanations or reasoning outside the <Answer> tag.

        Example valid responses:
        <Answer>tumor_size, stage_at_diagnosis</Answer>
        <Answer>None</Answer>
        """

        # ---- 5. Generate causal graph ----
        adj_matrix = llm_bfs_local(
            feature_descriptions,
            independent_nodes_verification,
            None,
            model,
            tokenizer,
            prompt_format
        )

        # ---- 6. Validate causal graph ----
        query = f"Here are the features and their description, and the causal graph to validate:\n {adj_matrix}."
        rag = retrieve_relevant_docs(client, query)

        causal_verification = causal_validation_instruction(
            adj_matrix, model_gemma, tokenizer_gemma, docs=rag
        )

        # ---- 7. Guardar no array ----
        results.append(causal_verification)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False)

    print(f"\n✔ Resultados salvos em: {OUTPUT_FILE}\n")



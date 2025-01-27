# Importa as bibliotecas necessárias
from transformers import AutoTokenizer, AutoModelForCausalLM

# Nome do modelo a ser utilizado
# GPT-2 é pré-treinado para geração de texto em várias tarefas
model_name = "gpt2"

# Carregar o tokenizador e o modelo
# O tokenizador converte texto em IDs que o modelo pode entender
print("Baixando e carregando o modelo GPT-2...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Modelo carregado com sucesso!")

# Texto de entrada para o modelo
# Aqui você define o prompt para orientar a geração de texto
input_text = "Explique, de forma simples, o conceito de inteligência artificial."
print(f"\nTexto de entrada: {input_text}")

# Tokenizar o texto de entrada
# Gera IDs e uma máscara de atenção para o modelo processar os dados
input_ids = tokenizer.encode(input_text, return_tensors="pt")
attention_mask = input_ids.new_ones(input_ids.shape)  # Cria uma máscara de atenção para os tokens válidos

# Configuração da geração de texto
# Aqui ajustamos os parâmetros para controlar o comportamento do modelo
print("\nGerando texto...")
output = model.generate(
    input_ids,               # IDs do texto de entrada
    attention_mask=attention_mask,  # Máscara de atenção para o modelo focar nos tokens certos
    max_length=100,          # Tamanho máximo da resposta gerada
    temperature=0.7,         # Controla a criatividade (valores menores tornam a saída mais conservadora)
    top_p=0.9,               # Filtro de palavras menos prováveis para maior fluidez
    do_sample=True,          # Ativa o modo criativo
    num_return_sequences=1,  # Quantidade de respostas geradas
    pad_token_id=tokenizer.eos_token_id  # Token para completar a resposta
)

# Decodificar o texto gerado
# Transforma os IDs gerados pelo modelo de volta em texto legível
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nTexto gerado pelo GPT-2:")
print(generated_text)

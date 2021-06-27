
# Análise de Comentários de E-commerce

Este é um sistema de análise de sentimentos de comentários de e-commerce. Com ele, é possível enviar avaliações de usuários e classificar automaticamente se o comentário traz uma opinião positiva ou negativa a respeito do produto em questão.

## Agradecimento

Os dados para treinar o modelo foram disponibilizados pela B2W e podem ser acessados no [repositório do Github](https://github.com/b2wdigital/b2w-reviews01).


## Features

- Embedding de texto com TF-IDF;
- Regressão logística para classificação;
- API para consultar os resultados do modelo com FastAPI;
- App desenvolvido com Streamlit para o usuário final. 

  
## Como rodar localmente

Clone o projeto

```bash
# Usando HTTPS
git clone https://github.com/iago/analise-sentimento-ecommerce.git

# ou usando SSH
git clone git@github.com:iago/analise-sentimento-ecommerce.git
```

Acesse o diretório do projeto

```bash
cd analise-sentimento-ecommerce
```

Instale as dependências

```bash
conda env create -f environment.yml
```

Rode o servidor da API

```bash
# Rodando o arquivo Python
python api.py
# ou diretamente pelo Uvicorn
uvicorn api:app
```

Rode o servidor do Streamlit

```bash
streamlit run front.py
```


## Roadmap

- Utilizar MLFlow para trackear resultados de experimentos;

- Aprimorar o app front-end.


## Contribuições

Contribuições são mais do que bem-vindas. Se você encontrar algum bug ou tiver qualquer ideia de feature nova, por favor, abra uma issue para conversarmos.
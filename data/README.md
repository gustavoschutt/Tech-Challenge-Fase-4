# Dados

## Origem e licença

**Brazilian E-Commerce Public Dataset by Olist**, publicado pela Olist no Kaggle:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Licença: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)** —
https://creativecommons.org/licenses/by-nc-sa/4.0/

Os arquivos abaixo são redistribuídos aqui **sem alteração de conteúdo** (apenas compactados com gzip),
para fins acadêmicos e não comerciais, com atribuição à Olist e sob a mesma licença.

## Arquivos incluídos (`data/raw/`)

| Arquivo | Uso no pipeline |
|---|---|
| `olist_order_reviews_dataset.csv.gz` | texto, nota e data das avaliações (fonte principal) |
| `olist_orders_dataset.csv.gz` | status, data da compra, entrega real × estimada (atraso) |
| `olist_order_items_dataset.csv.gz` | itens do pedido (categoria, nº de itens/vendedores, valores) |
| `olist_products_dataset.csv.gz` | categoria do produto |
| `olist_customers_dataset.csv.gz` | UF do cliente |
| `product_category_name_translation.csv.gz` | tradução das categorias (não obrigatório) |

O código lê `.csv` ou `.csv.gz` indistintamente (o pandas descompacta sozinho); **não é preciso descompactar**.
Se existirem as duas versões de um arquivo, a `.csv` tem preferência.

Os demais arquivos do dataset (geolocalização, pagamentos, vendedores) não são usados e não foram incluídos.

## Usando outra cópia dos dados

Para usar os CSVs baixados diretamente do Kaggle, extraia-os em `data/raw/` ou aponte `DATA_DIR`
no `.env` para a pasta onde eles estão.

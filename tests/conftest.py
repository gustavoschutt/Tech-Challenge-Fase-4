"""Fixtures: base Olist SINTÉTICA e mínima (mesmo esquema dos CSVs reais),
para testar o pipeline de ponta a ponta sem baixar dados nem modelos
(backend LSA + modo extrativo)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEXTOS = [
    (1, "Não recebi o produto", "Comprei há um mês e o produto ainda não chegou, ninguém responde"),
    (1, None, "Não recebi meu pedido até agora, péssimo"),
    (1, "Atrasou", "A entrega atrasou muito, chegou depois do prazo"),
    (2, None, "Produto chegou quebrado, a caixa estava amassada"),
    (2, None, "Veio quebrado e com defeito, não funciona"),
    (1, None, "Recebi apenas um dos dois itens do pedido, faltou o outro"),
    (2, None, "Veio a cor errada, diferente do anunciado"),
    (3, None, "Produto ok, mas a entrega demorou bastante"),
    (5, "Recomendo", "Chegou antes do prazo, produto de ótima qualidade"),
    (5, None, "Entrega rápida e produto conforme anunciado, recomendo"),
    (4, None, "Muito bom, bem embalado e chegou no prazo"),
    (5, None, "Excelente produto, recomendo a loja"),
    (5, None, "Ótimo atendimento, vendedor atencioso"),
    (4, None, "Bom custo benefício, vale a pena"),
    (5, None, "muito bom"),
    (5, None, "muito bom"),
    (1, None, "Produto falsificado, não é original"),
    (2, None, "A nota fiscal não veio junto com o produto"),
    (1, None, "Os Correios extraviaram minha encomenda"),
    (3, None, "Ligue para (11) 98888-7777 sobre a troca do produto"),
] * 3  # 60 avaliações


@pytest.fixture(scope="session")
def base_sintetica(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    n = len(TEXTOS)
    rid = [f"r{i:04d}" for i in range(n)]
    oid = [f"o{i:04d}" for i in range(n)]
    cid = [f"c{i:04d}" for i in range(n)]
    pd.DataFrame({
        "review_id": rid, "order_id": oid, "review_score": [t[0] for t in TEXTOS],
        "review_comment_title": [t[1] for t in TEXTOS], "review_comment_message": [t[2] for t in TEXTOS],
        "review_creation_date": "2018-03-10 00:00:00", "review_answer_timestamp": "2018-03-11 00:00:00",
    }).to_csv(raw / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": oid, "customer_id": cid, "order_status": "delivered",
        "order_purchase_timestamp": ["2017-05-01 10:00:00" if i % 2 else "2018-02-01 10:00:00" for i in range(n)],
        "order_approved_at": "2018-02-01 11:00:00", "order_delivered_carrier_date": "2018-02-02 11:00:00",
        "order_delivered_customer_date": ["2018-02-20 10:00:00" if i % 3 == 0 else "2018-02-08 10:00:00" for i in range(n)],
        "order_estimated_delivery_date": "2018-02-15 00:00:00",
    }).to_csv(raw / "olist_orders_dataset.csv", index=False)
    pd.DataFrame({
        "order_id": oid, "order_item_id": 1, "product_id": [f"p{i % 4}" for i in range(n)], "seller_id": "s1",
        "shipping_limit_date": "2018-02-03 00:00:00", "price": 100.0, "freight_value": 10.0,
    }).to_csv(raw / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame({"product_id": ["p0", "p1", "p2", "p3"],
                  "product_category_name": ["beleza_saude", "moveis_decoracao", "informatica_acessorios", None]}
                 ).to_csv(raw / "olist_products_dataset.csv", index=False)
    pd.DataFrame({"customer_id": cid, "customer_unique_id": cid, "customer_zip_code_prefix": 1000,
                  "customer_city": "x", "customer_state": ["SP" if i % 2 else "RJ" for i in range(n)]}
                 ).to_csv(raw / "olist_customers_dataset.csv", index=False)
    pd.DataFrame({"product_category_name": ["beleza_saude"], "product_category_name_english": ["health_beauty"]}
                 ).to_csv(raw / "product_category_name_translation.csv", index=False)
    return raw


@pytest.fixture(scope="session")
def settings_sinteticas(base_sintetica, tmp_path_factory):
    from voc_rag.config import get_settings

    art = tmp_path_factory.mktemp("artifacts")
    return get_settings(data_dir=base_sintetica, artifacts_dir=art, embedding_backend="lsa",
                        reranker_model="", llm_provider="extractive", relevance_threshold=0.05,
                        scope_threshold=0.0, min_evidences=2, min_base_analitico=10)

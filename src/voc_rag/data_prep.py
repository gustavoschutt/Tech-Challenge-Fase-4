"""Construção da base de conhecimento a partir do dataset Olist.

Saídas (em artifacts/base/):
- avaliacoes_todas.parquet : TODAS as avaliações (com e sem texto) + metadados.
                             Usada para estatísticas (camada analítica), para
                             que as proporções não fiquem enviesadas.
- documentos.parquet       : apenas avaliações com texto -> documentos do RAG.
- relatorio_preparacao.json: contagens de cada etapa (linhagem dos dados).

Unidade de documento: 1 avaliação = 1 documento (título + comentário).
Os textos são curtos (mediana ~9 palavras, máx. 208 caracteres no
comentário), então não há chunking: cortar destruiria a unidade de sentido.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .text_utils import clean_for_embedding, dedup_key, has_alpha

RAW_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

SHORT_TEXT_MAX_WORDS = 3


def raw_path(data_dir: Path, filename: str) -> Path:
    """Localiza um arquivo bruto aceitando .csv ou .csv.gz (o repositório
    versiona os CSVs compactados; quem baixar do Kaggle terá os .csv)."""
    data_dir = Path(data_dir)
    for candidate in (data_dir / filename, data_dir / f"{filename}.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Arquivo não encontrado: {data_dir / filename} (nem {filename}.gz)\n"
        "Os CSVs compactados vêm no repositório em data/raw/. Se não estiverem lá, baixe o dataset em "
        "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce e aponte DATA_DIR (no .env) "
        "para a pasta com os arquivos."
    )


def read_raw(data_dir: Path, filename: str, **kwargs) -> pd.DataFrame:
    """Lê um CSV bruto (.csv ou .csv.gz; o pandas descompacta automaticamente)."""
    return pd.read_csv(raw_path(data_dir, filename), encoding="utf-8", **kwargs)


def _read(data_dir: Path, key: str) -> pd.DataFrame:
    return read_raw(data_dir, RAW_FILES[key])


def _build_text(title, message) -> str:
    """Título + comentário. O título só entra se acrescentar algo
    (não é vazio e não está contido no comentário)."""
    t = clean_for_embedding(title) if isinstance(title, str) else ""
    m = clean_for_embedding(message) if isinstance(message, str) else ""
    if t and m:
        if dedup_key(t) and dedup_key(t) in dedup_key(m):
            return m
        sep = "" if t.endswith((".", "!", "?")) else "."
        return f"{t}{sep} {m}"
    return t or m


def _order_metadata(orders: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame,
                    customers: pd.DataFrame) -> pd.DataFrame:
    o = orders.copy()
    for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        o[c] = pd.to_datetime(o[c], errors="coerce")
    # Atraso em dias: entrega real - entrega estimada (positivo = atrasou).
    o["atraso_dias"] = (o["order_delivered_customer_date"].dt.normalize()
                        - o["order_estimated_delivery_date"].dt.normalize()).dt.days
    o["prazo_entrega_dias"] = (o["order_delivered_customer_date"] - o["order_purchase_timestamp"]).dt.days
    o["entregue"] = o["order_delivered_customer_date"].notna()

    it = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    it["product_category_name"] = it["product_category_name"].fillna("sem_categoria")
    # Categoria principal = categoria do item de maior valor do pedido
    # (pedidos multi-categoria são raros: ~0,8%; a lista completa também é guardada).
    main_cat = (it.sort_values("price", ascending=False)
                  .drop_duplicates("order_id")[["order_id", "product_category_name"]]
                  .rename(columns={"product_category_name": "categoria"}))
    agg = it.groupby("order_id").agg(
        categorias=("product_category_name", lambda s: sorted(set(s))),
        n_itens=("order_item_id", "count"),
        n_vendedores=("seller_id", "nunique"),
        valor_itens=("price", "sum"),
        valor_frete=("freight_value", "sum"),
    ).reset_index()
    meta = (o.merge(main_cat, on="order_id", how="left")
             .merge(agg, on="order_id", how="left")
             .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left"))
    meta["categoria"] = meta["categoria"].fillna("sem_itens")
    meta["categorias"] = meta["categorias"].apply(lambda v: v if isinstance(v, list) else ["sem_itens"])
    return meta[[
        "order_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "atraso_dias", "prazo_entrega_dias", "entregue",
        "categoria", "categorias", "n_itens", "n_vendedores", "valor_itens", "valor_frete",
        "customer_state",
    ]]


def build_knowledge_base(settings: Settings, verbose: bool = True) -> dict:
    data_dir = settings.data_dir
    out_dir = settings.base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    reviews = _read(data_dir, "reviews")
    report["avaliacoes_brutas"] = int(len(reviews))
    report["ausentes_titulo"] = int(reviews["review_comment_title"].isna().sum())
    report["ausentes_comentario"] = int(reviews["review_comment_message"].isna().sum())

    # 1) review_id repetido: a mesma avaliação (mesmo texto e nota) aparece
    #    associada a 2-3 pedidos. Mantemos 1 documento e guardamos os pedidos.
    report["linhas_review_id_repetido"] = int(reviews["review_id"].duplicated().sum())
    orders_per_review = reviews.groupby("review_id")["order_id"].apply(list)
    reviews = reviews.drop_duplicates("review_id", keep="first").copy()
    reviews["pedidos"] = reviews["review_id"].map(orders_per_review)
    report["avaliacoes_unicas"] = int(len(reviews))

    # 2) Metadados do pedido (entrega, categoria, UF).
    meta = _order_metadata(_read(data_dir, "orders"), _read(data_dir, "items"),
                           _read(data_dir, "products"), _read(data_dir, "customers"))
    df = reviews.merge(meta, on="order_id", how="left")
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")

    # 3) Texto do documento.
    df["texto"] = [_build_text(t, m) for t, m in zip(df["review_comment_title"], df["review_comment_message"])]
    df["tem_texto"] = df["texto"].map(has_alpha)
    df.loc[~df["tem_texto"], "texto"] = ""
    df["n_palavras"] = df["texto"].str.split().str.len().fillna(0).astype(int)
    df["texto_curto"] = df["tem_texto"] & (df["n_palavras"] <= SHORT_TEXT_MAX_WORDS)
    df["chave_dedup"] = df["texto"].map(dedup_key)
    df["pii_mascarada"] = df["texto"].str.contains(r"\[(?:TELEFONE|EMAIL|CPF|URL)\]", regex=True)

    # 4) Campos derivados úteis para filtros e para a exibição das evidências.
    df["ano_mes"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str).replace("NaT", "")
    df["faixa_nota"] = pd.cut(df["review_score"], bins=[0, 2, 3, 5], labels=["negativa", "neutra", "positiva"]).astype(str)
    df["situacao_entrega"] = np.select(
        [~df["entregue"].fillna(False).astype(bool), df["atraso_dias"] > 0],
        ["nao_entregue", "atrasada"], default="no_prazo",
    )

    keep = [
        "review_id", "order_id", "pedidos", "review_score", "faixa_nota", "review_creation_date",
        "texto", "tem_texto", "texto_curto", "n_palavras", "chave_dedup", "pii_mascarada",
        "order_status", "order_purchase_timestamp", "ano_mes", "atraso_dias", "prazo_entrega_dias",
        "situacao_entrega", "categoria", "categorias", "n_itens", "n_vendedores",
        "valor_itens", "valor_frete", "customer_state",
    ]
    all_reviews = df[keep].reset_index(drop=True)
    docs = all_reviews[all_reviews["tem_texto"]].reset_index(drop=True)

    report.update({
        "avaliacoes_sem_texto": int((~all_reviews["tem_texto"]).sum()),
        "documentos_indexados": int(len(docs)),
        "documentos_curtos_ate_3_palavras": int(docs["texto_curto"].sum()),
        "documentos_com_pii_mascarada": int(docs["pii_mascarada"].sum()),
        "textos_distintos": int(docs["chave_dedup"].nunique()),
        "taxa_comentario_por_nota": {
            int(k): round(float(v), 4)
            for k, v in all_reviews.groupby("review_score")["tem_texto"].mean().items()
        },
        "periodo_compras": [str(docs["order_purchase_timestamp"].min()), str(docs["order_purchase_timestamp"].max())],
    })

    all_reviews.to_parquet(out_dir / "avaliacoes_todas.parquet", index=False)
    docs.to_parquet(out_dir / "documentos.parquet", index=False)
    (out_dir / "relatorio_preparacao.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def load_documents(settings: Settings) -> pd.DataFrame:
    path = settings.base_dir / "documentos.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe. Rode: python scripts/01_preparar_base.py")
    return pd.read_parquet(path)


def load_all_reviews(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(settings.base_dir / "avaliacoes_todas.parquet")

"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

_DIRECTIONS = "\n".join(
    f"{subject} -{kind}-> {target}" for kind, (subject, target) in RELATION_SIGNATURES.items()
)
_SYSTEM_PROMPT = f"""Извлеки факты для графа знаний из одного фрагмента сервисной документации.
Типы сущностей: {", ".join(ENTITY_TYPES)}. Допустимые направления связей:
{_DIRECTIONS}

Используй только факты фрагмента. В списке состава каждая строка связывает модель из заголовка,
деталь, вид узла и указанного поставщика. В таблице ошибок связывай каждый код с его узлом.
FAILED означает фактический отказ в отчёте о выезде. FIXED_BY связывает ошибку с регламентом,
который применяется при этой ошибке. REPLACED_BY направлена от старой детали к новой.
Сохраняй все подтверждённые связи, включая отдельные строки списков. Не угадывай коды.
Приводи названия к словарным формам. Номера отчётов и бюллетеней не являются сущностями.
Для фрагмента без фактов графа, например планового регламента без кода ошибки, верни пустые массивы.

Пример: «Регламент SP-101 применяется при ошибке E-11».
{{"entities":[{{"type":"ErrorCode","name":"E-11"}},{{"type":"Procedure","name":"SP-101"}}],
"relations":[{{"subject_type":"ErrorCode","subject":"E-11","type":"FIXED_BY",
"object_type":"Procedure","object":"SP-101"}}]}}
Ответь только JSON с массивами entities и relations."""


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    names = vocabulary()
    wordlist = "\n".join(f"{kind}: {', '.join(values)}" for kind, values in names.items())
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Словарь названий:\n{wordlist}\n\nФрагмент документа:\n{chunk.text}"},
    ]
    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    for attempt in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=3072)
        try:
            raw = _parse_reply(response.content)
        except (ValueError, TypeError, RecursionError):
            if attempt == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": "Верни полный JSON по схеме с обоими массивами и всеми обязательными полями.",
                    }
                )
        else:
            break
    else:
        return result

    codes = set(find_codes(chunk.text))

    def grounded_entity(kind: str, raw_name: str) -> Entity | None:
        name = canonical_name(kind, raw_name, names)
        if name is None or (kind in CODE_TYPES and name not in codes):
            return None
        return Entity(kind, name)

    entities: dict[str, Entity] = {}
    relations: dict[Relation, None] = {}
    for item in raw["entities"]:
        entity = grounded_entity(item["type"], item["name"])
        if entity is None:
            result.dropped += 1
        else:
            entities[entity.node_id] = entity

    for item in raw["relations"]:
        subject = grounded_entity(item["subject_type"], item["subject"])
        target = grounded_entity(item["object_type"], item["object"])
        if subject is None or target is None:
            result.dropped += 1
            continue
        relation = Relation(subject, item["type"], target)
        if not relation.matches_schema():
            result.dropped += 1
            continue
        entities[subject.node_id] = subject
        entities[target.node_id] = target
        relations[relation] = None

    result.entities = list(entities.values())
    result.relations = list(relations)
    return result


def _parse_reply(content: str) -> dict[str, list[dict[str, str]]]:
    """Проверяет форму JSON отдельно от проверки фактов по тексту."""
    reply = json.loads(content)
    if not isinstance(reply, dict):
        raise ValueError("Ожидался JSON-объект")
    for name, array_schema in EXTRACTION_SCHEMA["properties"].items():
        items = reply.get(name)
        if not isinstance(items, list):
            raise ValueError(f"Ожидался массив {name}")
        item_schema = array_schema["items"]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Ожидался объект в массиве {name}")
            for field in item_schema["required"]:
                value = item.get(field)
                allowed = item_schema["properties"][field].get("enum")
                if not isinstance(value, str) or (allowed is not None and value not in allowed):
                    raise ValueError(f"Недопустимое поле {field} в массиве {name}")
    return reply

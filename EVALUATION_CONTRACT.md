# Контракт оценки AuditGym

Здесь точно описано: какие файлы составляют эпизод, как вызывать среду, что такое ответ агента и как он оценивается. Код в `auditgym/` и тесты в `tests/jury/` обязаны совпадать с этим файлом.

Бухгалтерская модель и правила, по которым запись считается ошибкой, — в [DESIGN.md](DESIGN.md).

Версия контракта: `1.0` (поле `schema_version` в файлах эпизода).

## 1. Точка входа: `judge_adapter.py`

Файл лежит в корне репозитория и в рабочей папке контейнера.

```python
def generate_episode(seed: int, config: dict, output_dir: str) -> None
def make_env() -> AuditEnv
def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict
```

- `generate_episode` пишет в `output_dir` восемь файлов эпизода (раздел 2). `config` — это `team_config` (раздел 8). Если передан объект с ключом `team_config`, берётся его значение. Одинаковые `seed` и `config` дают одинаковые файлы.
- `make_env` возвращает объект с методами `reset` и `step` (раздел 3). Работает в одном процессе, подпроцессы не запускает.
- `score_episode` читает `truth.json` и открытые файлы из `episode_dir` и возвращает метрики (раздел 6). Это единственное место, где читается эталон в закрытом режиме.

## 2. Файлы эпизода

Одна папка, восемь файлов. Кодировка UTF-8.

| Файл | Кто видит | Содержимое |
|---|---|---|
| `manifest.json` | агент | паспорт эпизода |
| `ledger.csv` | агент | проводки |
| `invoices.jsonl` | агент | счета |
| `payments.jsonl` | агент | платежи и возвраты |
| `fx.csv` | агент | курсы валют |
| `policy.json` | агент | учётная политика |
| `truth.json` | только оценщик | эталон |
| `provenance.json` | только оценщик | история генерации и внесённых изменений |

Общие правила значений:

- Деньги — целые числа в минимальных единицах валюты. Разрядность валюты задана в `manifest.currency_exponents` (JPY 0, KZT/USD/EUR 2, KWD 3).
- Время — ISO-8601 со смещением, с точностью до секунды: `2026-09-30T23:30:00+05:00`. Смещение записи может отличаться от смещения компании.
- Курсы — десятичные строки: `"470.25"`.
- Отсутствующая ссылка — `null`. В CSV это пустое поле; слово `null` загрузчик тоже понимает.
- Суммы в разных валютах без пересчёта не сравниваются.

### manifest.json

```json
{
  "schema_version": "1.0",
  "episode_id": "ep-7c1e90ab",
  "base_currency": "KZT",
  "period_start": "2026-07-01T00:00:00+05:00",
  "period_end": "2026-09-30T23:59:59+05:00",
  "timezone": "Asia/Almaty",
  "utc_offset": "+05:00",
  "record_counts": {"ledger": 612, "invoices": 58, "payments": 49, "fx": 152},
  "document_count": 107,
  "currency_exponents": {"KZT": 2, "USD": 2},
  "rounding": {"mode": "half_up", "unit": "minor"}
}
```

- Обе границы периода входят в период.
- `utc_offset` — постоянное смещение компании. Местная дата любого времени считается по нему. Базы часовых поясов код не использует.
- `document_count = invoices + payments`.
- Выгрузка шире периода: в неё попадают записи примерно на 10 дней раньше и позже.

### ledger.csv

Столбцы в этом порядке: `row_id, timestamp, account, debit_minor, credit_minor, currency, entity_id, invoice_id, payment_id`.

- В каждой строке ровно одно из `debit_minor`, `credit_minor` больше нуля, второе равно 0.
- `invoice_id` и `payment_id` ссылаются на бизнес-ключи документов, не на `doc_id`.
- Счета учёта: `AR`, `REVENUE`, `CASH`, `CUSTOMER_ADVANCES`, `FX_CLEARING` и фоновые `RENT_EXPENSE`, `PAYROLL_EXPENSE`, `UTILITIES_EXPENSE`.

### invoices.jsonl

Одна строка — один JSON-объект.

| Поле | Тип | Примечание |
|---|---|---|
| `doc_id` | строка | ключ документа |
| `invoice_id` | строка | бизнес-ключ счёта |
| `entity_id` | строка | клиент |
| `issued_at`, `due_at` | время | |
| `currency` | строка | |
| `total_minor` | целое | равно сумме `line_items[].amount_minor` |
| `line_items` | список | `{"description", "quantity", "unit_price_minor", "amount_minor"}`, порядок значим |
| `po_ref` | строка или `null` | наше поле: номер заказа клиента |
| `service_period` | строка или `null` | наше поле: за какой месяц услуга, `"2026-08"` |

### payments.jsonl

| Поле | Тип | Примечание |
|---|---|---|
| `doc_id` | строка | ключ документа |
| `payment_id` | строка | бизнес-ключ платежа |
| `timestamp` | время | |
| `currency` | строка | |
| `amount_minor` | целое | больше нуля — входящий платёж, меньше нуля — возврат клиенту |
| `invoice_refs` | список строк или `null` | `invoice_id` счетов, которые закрывает платёж |

Сколько именно ушло на каждый счёт, видно в книге: строки `Cr AR` с парой `invoice_id` + `payment_id`. Клиент платежа — `entity_id` в его строках книги.

### fx.csv

Столбцы: `date, source_currency, target_currency, rate`. Смысл: 1 единица `source_currency` = `rate` единиц `target_currency`. Курсы есть только на рабочие дни.

Ключ записи — `date:source_currency:target_currency`, например `2026-09-30:USD:KZT`. Он собирается из открытых столбцов и при замене ID не меняется. Если в файле встретились одинаковые ключи, второй и следующие получают суффикс `#2`, `#3` в порядке строк.

### policy.json

```json
{
  "fx_rate_date": "document_date",
  "fx_missing_rate": "previous_available",
  "fx_direction": "source_to_target",
  "rounding": "half_up",
  "period_bounds": "inclusive",
  "cutoff_basis": "issued_at_local",
  "allocation_rule": "open_invoices_first",
  "duplicate_key": ["entity_id", "po_ref", "service_period", "currency", "total_minor"],
  "fx_materiality": {"min_relative": "0.005", "min_minor": 100},
  "amount_tolerance_minor": {
    "duplicate_invoice": 0, "unallocated_payment": 0, "fx_error": 0, "cutoff_error": 0
  }
}
```

Что значит каждое правило — в [DESIGN.md](DESIGN.md), раздел «Правила, по которым решается ошибка». Загрузчик принимает `policy.json` без любого из этих ключей; отсутствующий допуск равен 0.

### truth.json

```json
{
  "schema_version": "1.0",
  "episode_id": "ep-7c1e90ab",
  "issues": [
    {
      "issue_id": "ISS-0001",
      "type": "duplicate_invoice",
      "affected_ids": ["doc-a31f09c2", "doc-5e77b410"],
      "expected_amount_minor": 12500000,
      "currency": "KZT",
      "evidence_sets": [["doc-a31f09c2", "doc-5e77b410"]]
    }
  ]
}
```

- В чистом эпизоде `issues` — пустой список.
- `evidence_sets` — список допустимых наборов. В каждом не больше 10 ID.
- Oracle строит ответ так: по одному выводу на ошибку, поля из эталона, `evidence_ids` — первый набор.
- Загрузчик принимает и «голый» список ошибок без обёртки.

### provenance.json

```json
{
  "schema_version": "1.0",
  "episode_id": "ep-7c1e90ab",
  "generator_version": "0.1.0",
  "seed": 20261120,
  "attempt": 0,
  "team_config": {},
  "injections": [
    {
      "issue_ids": ["ISS-0001"],
      "kind": "duplicate_invoice",
      "scenario": null,
      "description": "Счёт проведён второй раз под новым номером",
      "changes": [
        {"file": "invoices.jsonl", "op": "add", "key": "doc-5e77b410", "before": null, "after": {}},
        {"file": "ledger.csv", "op": "add", "key": "row-91c0de55", "before": null, "after": {}}
      ],
      "effects": [
        {"control": "C5", "currency": "KZT", "delta_minor": 12500000}
      ]
    }
  ],
  "distractors": [
    {"kind": "recurring_amount", "ids": ["doc-0b6e2a17", "doc-c4419d03"]}
  ]
}
```

- `op` — `add`, `modify` или `remove`. `before` и `after` — запись целиком в формате своего файла. `key` — `row_id` для книги, `doc_id` для документов.
- **Как получить книги до ошибки:** пройти `injections` с конца, в каждой пройти `changes` с конца и применить обратную правку (`add` → удалить запись `key`; `remove` → вернуть `before`; `modify` → заменить на `before`). Результат должен пройти контрольные проверки C1–C6 из `DESIGN.md`.
- `effects` — какую контрольную проверку нарушает правка и на какую сумму.
- `attempt` — номер принятой попытки генерации (см. `DESIGN.md`, «Откуда берётся эталон»).

## 3. Среда: `reset` и `step`

### reset(seed, config) → (observation, info)

```json
{"mode": "closed", "episode_dir": "/path/to/episode", "team_config": {}}
```

- `mode`: `"closed"` или `"train"`.
- С `episode_dir` среда загружает шесть открытых файлов этой папки. Ничего не генерирует, ID не меняет. `seed` при этом не используется.
- Без `episode_dir` (только `train`) эпизод создаёт генератор из `seed` и `team_config`.
- Повторный `reset` начинает эпизод с нуля. Состояние прошлого эпизода не сохраняется.
- Отсчёт времени начинается в момент входа в `reset`.

Первое наблюдение:

```json
{"task": "…", "manifest": {}, "policy": {}, "remaining_steps": 100}
```

`task` — строка. В ней: что нужно сдать, четыре значения `type`, какие ID канонические для `affected_ids`, формула пересчёта валют, допуск суммы, лимиты ответа. `info` — пустой объект.

### step(action) → (observation, reward, terminated, truncated, info)

Действие: `{"name": "<имя>", "args": {…}}`.

Наблюдение: `{"result": …, "error": null, "remaining_steps": 97}` или `{"result": null, "error": {"code": "…", "message": "…"}, "remaining_steps": 97}`.

| Действие | Аргументы | `result` |
|---|---|---|
| `list_records` | `table`, `cursor`, `limit` | `{"records": […], "next_cursor": null \| строка}` |
| `get_record` | `table`, `id` | объект записи |
| `get_document` | `doc_id` | объект счёта или платежа |
| `calculator` | `expression` | десятичная строка |
| `submit` | `findings` | `{"accepted": true}` |

**list_records.** Таблицы: `ledger`, `invoices`, `payments`, `fx`. Первый `cursor` — `null`. `limit` — целое от 1; больше 50 обрезается до 50; по умолчанию 50. Записи идут в порядке строк файла. Курсор — непрозрачная строка; один и тот же курсор всегда возвращает ту же страницу. После последней страницы `next_cursor` равен `null`.

**get_record.** Каждая таблица ищет в своём пространстве ID:

| Таблица | Ключ | Если не найдено |
|---|---|---|
| `ledger` | `row_id` | ошибка |
| `invoices` | `doc_id` | ищем по `invoice_id` |
| `payments` | `doc_id` | ищем по `payment_id` |
| `fx` | `date:SRC:TGT` | ошибка |

Сравнение строк точное. Запись `fx` возвращается с добавленным полем `id` (её ключ).

**get_document.** Ищет `doc_id` среди счетов и платежей. Поиск идёт по словарю в памяти; строка не превращается в путь.

**calculator.** Грамматика: десятичные числа, `+ - * /`, скобки, унарный минус. Длина выражения до 200 символов. Счёт в `Decimal`. Результат — строка без экспоненты и лишних нулей: `0.1+0.2` → `"0.3"`. Всё остальное (имена, вызовы, `**`, строки) — ошибка `calc_error`. Деление на ноль — тоже `calc_error`.

**submit.** Аргумент `findings` — объект из раздела 4. Завершает эпизод: `terminated = true`. При верной структуре `result` равен `{"accepted": true}`, при неверной возвращается ошибка `invalid_submission`. Ни в том, ни в другом случае ответ не говорит, совпали ли выводы с эталоном.

### Коды ошибок

| Код | Когда |
|---|---|
| `unknown_action` | неизвестное `name` или действие не объект |
| `unknown_table` | неизвестная таблица |
| `unknown_id` | записи или документа с таким ID нет |
| `bad_arguments` | нет нужного аргумента или он не того типа, неверный курсор |
| `calc_error` | выражение вне грамматики или деление на ноль |
| `invalid_submission` | неверная структура ответа |
| `time_limit` | вышло время эпизода |
| `episode_over` | эпизод уже завершён |

Текст ошибки короткий и постоянный для каждого кода. В нём нет путей, трассировок, seed и данных эталона. Среда от ошибки не падает.

### Бюджет и завершение

- 100 действий на эпизод, включая `submit`. Каждый вызов `step` до конца эпизода тратит одно действие, в том числе ошибочный.
- `submit` сотым действием принимается.
- Сотое действие, не являющееся `submit`, выполняется и возвращает `truncated = true`. Итоговый ответ считается пустым.
- Время: 60 секунд от входа в `reset`. Действие, пришедшее позже, возвращает ошибку `time_limit` и `truncated = true`; итоговый ответ пустой, даже если это был `submit`. Лимит можно уменьшить через `team_config.time_limit_s`.
- После конца эпизода любой `step` возвращает ошибку `episode_over` с теми же `terminated` и `truncated`. Результат эпизода не меняется.

### Награда и режимы

| | `closed` | `train` |
|---|---|---|
| Награда за шаг исследования | 0 | 0 |
| Награда на последнем шаге | 0 | итоговая награда из раздела 6 |
| `info` на последнем шаге | `{}` | `{"metrics": {…}}` с полями раздела 6 |
| Читает ли среда `truth.json` | нет | да, один раз при завершении |

В закрытом режиме настоящую награду считает `score_episode` на стороне оценщика.

## 4. Ответ агента: findings

```json
{
  "episode_id": "ep-7c1e90ab",
  "findings": [
    {
      "type": "duplicate_invoice",
      "affected_ids": ["doc-a31f09c2", "doc-5e77b410"],
      "claimed_amount_minor": 12500000,
      "currency": "KZT",
      "evidence_ids": ["doc-a31f09c2", "doc-5e77b410"],
      "explanation": "Один заказ и один месяц услуги выставлены дважды."
    }
  ]
}
```

### Значения `type`

| Наш `type` | Тип ошибки в кейсе |
|---|---|
| `duplicate_invoice` | дубль счёта |
| `unallocated_payment` | пропущенное или частичное распределение платежа |
| `fx_error` | неверная дата или курс конвертации |
| `cutoff_error` | ошибка отнесения к отчётному периоду |

### Канонические ID и сумма по типам

| `type` | `affected_ids` | `claimed_amount_minor`, `currency` | Допустимые наборы доказательств |
|---|---|---|---|
| `duplicate_invoice` | `doc_id` обоих счетов | `total_minor` счёта, валюта счёта | оба `doc_id` |
| `unallocated_payment` | `doc_id` платежа и `doc_id` счёта | нераспределённая часть платежа, валюта платежа | оба `doc_id` |
| `fx_error` | `doc_id` счёта | модуль разницы между проведённой и правильной суммой, базовая валюта | `doc_id` счёта + `row_id` его строки `REVENUE` + ключ `fx` правильной даты |
| `cutoff_error` | `doc_id` счёта | `total_minor` счёта, валюта счёта | `doc_id` счёта + `row_id` его строки `AR`; либо `doc_id` счёта + `row_id` его строки `REVENUE` |

Документы в `affected_ids` указываются только по `doc_id`.

## 5. Проверка структуры

Проверяется до любой нормализации. Ошибка структуры завершает эпизод с наградой −1 и записью в журнал оценщика.

Структура неверна, если:

- `findings` не объект; нет `episode_id` или он не строка; нет списка `findings`;
- выводов больше 30;
- в выводе нет одного из шести полей или поле не того типа: `type` и `currency` — строки, `affected_ids` и `evidence_ids` — списки строк, `explanation` — строка;
- `claimed_amount_minor` не целое число JSON: строка, дробное или `true`/`false`;
- в `evidence_ids` больше 10 элементов, считая повторы;
- `explanation` длиннее 1000 символов Unicode (code points).

Не являются ошибкой структуры и дают обычный FP: неизвестная строка в `type`, неизвестная валюта, несуществующие ID, пустой `affected_ids`.

`episode_id`, не совпадающий с `manifest.episode_id`, записывается в журнал как предупреждение и на оценку не влияет.

## 6. Сопоставление и награда

### Когда вывод подходит к ошибке эталона

Все пять условий сразу:

1. `type` равен точно.
2. Множество `affected_ids` равно множеству из эталона. Порядок и повторы внутри списка не важны. Строки сравниваются точно.
3. `currency` равна точно.
4. `abs(claimed_amount_minor − expected_amount_minor) ≤ τ`, где τ — `policy.amount_tolerance_minor[type]`, по умолчанию 0.
5. Множество `evidence_ids` целиком содержит хотя бы один набор из `evidence_sets`, и каждый указанный ID существует.

ID **существует**, если его возвращает инструмент: это `row_id` строки книги, `doc_id`, `invoice_id` или `payment_id` документа, либо ключ записи `fx`. Лишние существующие доказательства не мешают. Один несуществующий ID делает вывод несопоставимым.

Текст `explanation` в сопоставлении не участвует.

### Выбор пар

Строится двудольный граф: слева каждый вывод (повторы — отдельные вершины), справа каждая ошибка эталона, ребро — «подходит». Берётся паросочетание наибольшего размера.

Если таких несколько, выбор детерминирован: ошибки перебираются по `issue_id` в порядке code points; для каждой берётся наименьший индекс вывода (с нуля), при котором паросочетание наибольшего размера всё ещё достижимо. Это даёт лексикографически наименьшую последовательность пар `(issue_id, индекс)`.

- TP — размер паросочетания.
- FP — число выводов минус TP. Повторный верный вывод — FP.
- FN — число ошибок эталона минус TP.

### Метрики

- `P = TP / (TP + FP)`, `R = TP / (TP + FN)`. При нулевом знаменателе значение равно 0.
- `F1 = 2·TP / (2·TP + FP + FN)`.
- Эталон и ответ пусты: `F1 = 1`, при этом `P = R = 0`.
- Эталон пуст, выводы есть: `F1 = 0`.

### Награда

```
return = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)
```

`inspection_steps` — число вызовов `step` до `submit`, включая ошибочные. Сам `submit` не считается. При обрыве по шагам или времени оценивается пустой ответ с числом сделанных шагов.

Дроби считаются точно (`Fraction`), наружу отдаются как `float`.

### Что возвращает `score_episode`

```json
{"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "return": 0.98}
```

При ошибке структуры: `{"tp": null, "fp": null, "fn": null, "precision": null, "recall": null, "f1": null, "return": -1}`.

### Журнал оценщика

`score_episode` дописывает одну JSON-строку в файл из переменной окружения `AUDITGYM_GRADER_LOG`. Если она не задана — в `auditgym/grader_log.jsonl` во временной папке системы. В папку эпизода ничего не пишется.

В записи: `episode_id`, метрики, выбранные пары, для каждого несопоставленного вывода — причина по лучшему кандидату (`wrong_type`, `wrong_ids`, `wrong_currency`, `amount_off`, `evidence_insufficient`, `evidence_unknown`), а для ошибки структуры — её описание. Агент этот файл не видит.

## 7. Поля с идентификаторами

Нужно тому, кто согласованно меняет ID в копии эпизода. Наш инструмент `auditgym/verify/transform.py` использует тот же список.

| Пространство ID | Где определяется | Где встречается как ссылка |
|---|---|---|
| `row_id` | `ledger.csv` | `truth.evidence_sets`, `provenance…key`, записи в `before`/`after` |
| `doc_id` | `invoices.jsonl`, `payments.jsonl` | `truth.affected_ids`, `truth.evidence_sets`, `provenance…key`, `provenance.distractors[].ids`, `before`/`after` |
| `invoice_id` | `invoices.jsonl` | `ledger.invoice_id`, `payments.invoice_refs`, `before`/`after` |
| `payment_id` | `payments.jsonl` | `ledger.payment_id`, `before`/`after` |
| `entity_id` | `invoices.jsonl`, `ledger.csv` | `before`/`after` |
| `issue_id` | `truth.json` | `provenance.injections[].issue_ids` |

Не являются ID и не заменяются: ключ `fx` (`date:SRC:TGT`), `episode_id`, `po_ref`, `service_period`, названия счетов учёта.

Безопасно переставлять: строки `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`. Нельзя переставлять `line_items` внутри счёта.

## 8. Настройки генератора: `team_config`

```json
{
  "config_version": "1",
  "n_ledger_rows": 600,
  "n_documents": 90,
  "base_currency": "KZT",
  "currencies": ["KZT", "USD"],
  "timezone": "Asia/Almaty",
  "errors": {
    "duplicate_invoice": 0,
    "unallocated_payment": {"missing": 0, "partial": 0},
    "fx_error": {"wrong_date": 0, "wrong_pair": 0},
    "cutoff_error": {"late": 0, "early": 0}
  },
  "scenarios": {"S1": 0, "S2": 0},
  "distractors": {"split_payment": 0, "refund": 0, "recurring_amount": 0, "boundary": 0, "advance": 0},
  "rounding": "half_up",
  "fx_rate_date": "document_date",
  "time_limit_s": 60
}
```

- Любой ключ можно опустить, тогда берётся значение по умолчанию. Пустой объект даёт чистую компанию без двойников.
- Каждая ошибка, каждый её вид, каждый сценарий и каждый двойник включаются отдельно. Число — сколько таких случаев внести.
- Для вида ошибки вместо объекта можно дать число; тогда вид выбирает генератор.
- Готовые рецепты для двадцати сценариев жюри лежат в `configs/matrix/D01.json … D20.json`.
- Неизвестный ключ или недопустимое значение — понятная ошибка `ValueError` при генерации, а не молчаливый пропуск.

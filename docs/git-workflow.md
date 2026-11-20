# Как работаем с git

```
твой компьютер  →  GitLab (рабочий, приватный)  →  друг  →  GitHub друга (ссылка для сдачи)
```

Рабочий репозиторий живёт на GitLab. GitHub — зеркало, оно даёт ссылку для сдачи. Друг забирает изменения из GitLab и пушит их на свой GitHub. История коммитов и авторство при этом сохраняются.

## Один раз: ты

Сделано 20.11: SSH-ключ добавлен в GitLab, приватный проект создан, папка подключена.

- Проект: https://gitlab.com/nazarakanov/cookedbooks
- Адрес для git: `git@gitlab.com:nazarakanov/cookedbooks.git` (remote `origin`)

Осталось пригласить людей: **Manage → Members → Invite members**. Участникам команды — роль Maintainer, другу с GitHub — Reporter (этого хватает, чтобы забирать код).

Для нового участника команды:

```bash
git clone git@gitlab.com:nazarakanov/cookedbooks.git
```

## Один раз: друг

1. На GitHub: **New repository**, имя `cookedbooks`, **Private** (до сдачи), без README.
2. Заведи аккаунт на GitLab и добавь туда свой SSH-ключ (как в шагах 1–4 выше), чтобы читать приватный проект.
3. Склонируй проект из GitLab и добавь GitHub вторым адресом:
   ```bash
   git clone git@gitlab.com:nazarakanov/cookedbooks.git && cd cookedbooks
   ```
   ```bash
   git remote add github git@github.com:ЛОГИН_ДРУГА/cookedbooks.git
   ```
   ```bash
   git push github main
   ```

## Каждый раз: синхронизация (друг)

Каждый вечер и обязательно перед сдачей:

```bash
cd cookedbooks && git pull origin main && git push github main
```

## Перед сдачей (22 ноября)

1. Дедлайн — 23:59 по Астане. Последний пуш в GitLab — до 21:00.
2. Друг синхронизирует (команда выше).
3. Друг делает GitHub-репозиторий публичным: **Settings → General → Danger Zone → Change visibility → Public**.
4. Открыть ссылку в окне инкогнито и убедиться, что код виден.
5. Сделать архив кода и положить его в папку сдачи на Google Диске вместе с текстовым файлом, где записана ссылка на GitHub. Сдаётся ссылка на эту папку, через форму в личном кабинете.
   ```bash
   git archive --format=zip -o cookedbooks.zip HEAD
   ```
6. Запасной вариант: сделать публичным и GitLab-проект — вторая ссылка на случай проблем с GitHub.

## Чего не делать

- Не входить в аккаунт друга со своего компьютера, не брать его пароль или токен.
- Не пушить в GitHub с других компьютеров: зеркало обновляет один человек.

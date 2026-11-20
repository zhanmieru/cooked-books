# Как работаем с git

```
твой компьютер  →  GitLab (рабочий, приватный)  →  друг  →  GitHub друга (ссылка для сдачи)
```

Рабочий репозиторий живёт на GitLab. GitHub — зеркало, оно даёт ссылку для сдачи. Друг забирает изменения из GitLab и пушит их на свой GitHub. История коммитов и авторство при этом сохраняются.

## Один раз: ты

1. Зарегистрируйся на https://gitlab.com
2. Создай SSH-ключ (на все вопросы можно нажать Enter):
   ```bash
   ssh-keygen -t ed25519 -C "gitlab"
   ```
3. Скопируй публичный ключ и добавь его в GitLab: аватар → **Preferences → SSH Keys → Add new key**, вставить, сохранить.
   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub
   ```
4. Проверь, что ключ работает. Ответ должен быть `Welcome to GitLab, @твой_логин!`
   ```bash
   ssh -T git@gitlab.com
   ```
5. Создай проект: **New project → Create blank project**, имя `auditgym`, Visibility **Private**, галочку «Initialize repository with a README» **сними**.
6. Подключи папку к GitLab и запушь (или скажи Claude логин, он сделает сам):
   ```bash
   git remote add origin git@gitlab.com:ТВОЙ_ЛОГИН/auditgym.git
   ```
   ```bash
   git push -u origin main
   ```
7. Добавь людей: проект → **Manage → Members → Invite members**.
   - Тиммейты, которые пишут код, — роль **Maintainer** (иначе GitLab не даст пушить в `main`).
   - Друг, который только зеркалит, — роль **Reporter** (только чтение).

## Один раз: друг

1. На GitHub: **New repository**, имя `auditgym`, **Private** (до сдачи), без README.
2. Заведи аккаунт на GitLab и добавь туда свой SSH-ключ (как в шагах 1–4 выше), чтобы читать приватный проект.
3. Склонируй проект из GitLab и добавь GitHub вторым адресом:
   ```bash
   git clone git@gitlab.com:ЛОГИН_ВЛАДЕЛЬЦА/auditgym.git && cd auditgym
   ```
   ```bash
   git remote add github git@github.com:ЛОГИН_ДРУГА/auditgym.git
   ```
   ```bash
   git push github main
   ```

## Каждый раз: синхронизация (друг)

Каждый вечер и обязательно перед сдачей:

```bash
cd auditgym && git pull origin main && git push github main
```

## Перед сдачей (22 ноября)

1. Дедлайн — 23:59 по Астане. Последний пуш в GitLab — до 21:00.
2. Друг синхронизирует (команда выше).
3. Друг делает GitHub-репозиторий публичным: **Settings → General → Danger Zone → Change visibility → Public**.
4. Открыть ссылку в окне инкогнито и убедиться, что код виден.
5. Запасной вариант: сделать публичным и GitLab-проект — вторая ссылка на случай проблем с GitHub.

## Чего не делать

- Не входить в аккаунт друга со своего компьютера, не брать его пароль или токен.
- Не пушить в GitHub с других компьютеров: зеркало обновляет один человек.
